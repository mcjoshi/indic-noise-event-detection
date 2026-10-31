"""Convolutional U-Net masker for Track 2, trained directly on SI-SDR.

Why replace the frame-context MLP that is currently deployed:

  * it sees +-2 frames (~16 ms) of context, so it cannot use the temporal shape of an
    event at all - a horn's decay or a bark's envelope is invisible to it;
  * it was trained on CPU because a vLLM process occupied the GPU, so it saw 6,000 clips
    for 25 epochs rather than what an A100 allows;
  * it regresses the ideal ratio mask, a proxy, while the leaderboard scores SI-SDR.

This model is a small 2-D U-Net over the magnitude spectrogram (no recurrence, which also
sidesteps the cuDNN bf16 RNN trap recorded in CLAUDE.md), and it is trained on the
**waveform SI-SDR** that the competition actually measures.

Two properties are built in for the ASR side of the score, because `Combined = SI-SDR +
100 * dWER` and our deployed masker currently gives back 1.46 points in transcription
damage:

  * a **mask floor**: the predicted gain is mapped into [floor, 1], so a bin is attenuated
    at most `-20*log10(floor)` dB and never zeroed.  Spectral holes are the classic cause
    of enhancement wrecking ASR, and a floor removes them by construction.
  * the mask is applied only inside detected event spans by the existing span-blend
    wrapper, so speech outside events is bit-exact.
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
