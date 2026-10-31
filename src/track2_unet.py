"""Convolutional U-Net masker (magnitude spectrogram in, bounded gain mask out) trained on waveform SI-SDR.

The predicted gain is mapped into [floor, 1], so no bin is ever zeroed. In this release it only produces the
enhancement-residual input channel (mixture minus estimated speech) of one synthetic specialist.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from synthetic_aug import HOP, N_FFT

N_FREQ = N_FFT // 2 + 1


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride=(1, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, (3, 5), stride=stride, padding=(1, 2)),
            nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, (3, 3), padding=1),
            nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, (3, 4), stride=(1, 2), padding=(1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(cout + skip, cout, (3, 3), padding=1),
            nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, x, s):
        x = self.up(x)
        if x.shape[-1] != s.shape[-1]:
            x = torch.nn.functional.interpolate(x, size=s.shape[-2:], mode="nearest")
        return self.net(torch.cat([x, s], dim=1))


class UNetMasker(nn.Module):
    """(B, 1, T, F) log-magnitude -> (B, T, F) gain in [floor, 1]."""

    def __init__(self, ch: tuple[int, ...] = (32, 64, 96, 128), floor: float = 0.1):
        super().__init__()
        self.floor = floor
        self.inp = nn.Sequential(nn.Conv2d(1, ch[0], (3, 3), padding=1),
                                 nn.BatchNorm2d(ch[0]), nn.GELU())
        self.d1 = ConvBlock(ch[0], ch[1])
        self.d2 = ConvBlock(ch[1], ch[2])
        self.d3 = ConvBlock(ch[2], ch[3])
        # widen the temporal receptive field cheaply with dilated convs at the bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(ch[3], ch[3], (5, 3), padding=(4, 1), dilation=(2, 1)),
            nn.BatchNorm2d(ch[3]), nn.GELU(),
            nn.Conv2d(ch[3], ch[3], (5, 3), padding=(8, 1), dilation=(4, 1)),
            nn.BatchNorm2d(ch[3]), nn.GELU(),
        )
        self.u3 = UpBlock(ch[3], ch[2], ch[2])
        self.u2 = UpBlock(ch[2], ch[1], ch[1])
        self.u1 = UpBlock(ch[1], ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], 1, (3, 3), padding=1)

    def forward(self, log_mag: torch.Tensor) -> torch.Tensor:
        x0 = self.inp(log_mag)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x3 = self.mid(x3)[:, :, : x3.shape[2], :]   # dilation pads time; crop back
        y = self.u3(x3, x2)
        y = self.u2(y, x1)
        y = self.u1(y, x0)
        g = torch.sigmoid(self.out(y)).squeeze(1)
        # never zero a bin: spectral holes are what break ASR
        return self.floor + (1.0 - self.floor) * g


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative SI-SDR, the quantity the leaderboard scores on the synthetic subset."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    ratio = proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()


class TorchSTFT(nn.Module):
    """Differentiable STFT/iSTFT pair matching the numpy one used elsewhere."""

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(x, self.n_fft, self.hop, window=self.window,
                          return_complex=True, center=True)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(spec, self.n_fft, self.hop, window=self.window,
                           length=length, center=True)
