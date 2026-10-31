"""CRNN sound event detector at the metric's own 10 ms resolution, plus decoding.

Frequency is pooled away across the convolutional stack; time is never pooled, so the
output frame grid is exactly the 10 ms grid the segment-Dice metric uses and boundaries
are never quantised coarser than the ~90 ms collar.
"""

from __future__ import annotations
from paths import DATA, PRETRAINED, WORK  # noqa: F401

import contextlib
import numpy as np
import math

import torch
import torch.nn as nn
import torchaudio

from dataset import CATEGORIES, FRAME_SEC, HOP, SR
from epn_head import EPNHead


class LogMel(nn.Module):
    def __init__(self, n_mels: int = 128, n_fft: int = 1024):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, hop_length=HOP, n_mels=n_mels,
            f_min=0.0, f_max=SR / 2, power=2.0, center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, wav: torch.Tensor, n_frames: int | None = None) -> torch.Tensor:
        x = self.to_db(self.mel(wav))          # (B, n_mels, T)
        x = x.transpose(1, 2)                  # (B, T, n_mels)
        if n_frames is not None:
            if x.shape[1] > n_frames:
                x = x[:, :n_frames]
            elif x.shape[1] < n_frames:
                pad = n_frames - x.shape[1]
                x = torch.cat([x, x[:, -1:].expand(-1, pad, -1)], dim=1)
        return x


class PCEN(nn.Module):
    """Per-channel energy normalisation (Wang et al., ICASSP 2017; Lostanlen et al., SPL 2019) on mel
    power: gain-invariant, and it amplifies quiet content relative to a slow running mean, which is
    where the far-field events that sit below the speech live.  alpha, delta, r are per-band and
    trainable (log-parameterised); the smoother is a fixed first-order IIR."""

    def __init__(self, n_mels: int = 128, tau_sec: float = 0.4, alpha: float = 0.8,
                 delta: float = 10.0, r: float = 0.25, eps: float = 1e-6):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((n_mels,), math.log(alpha)))
        self.log_delta = nn.Parameter(torch.full((n_mels,), math.log(delta)))
        self.log_r = nn.Parameter(torch.full((n_mels,), math.log(r)))
        self.s = 1.0 - math.exp(-HOP / SR / tau_sec)
        self.eps = eps

    def forward(self, power: torch.Tensor) -> torch.Tensor:   # (B, n_mels, T) -> (B, T, n_mels)
        with torch.autocast("cuda", enabled=False):
            e = power.float()
            m = torch.empty_like(e)
            acc = e[..., 0]
            m[..., 0] = acc
            for t in range(1, e.shape[-1]):
                acc = (1.0 - self.s) * acc + self.s * e[..., t]
                m[..., t] = acc
            alpha = self.log_alpha.exp()[None, :, None]
            delta = self.log_delta.exp()[None, :, None]
            r = self.log_r.exp()[None, :, None]
            out = (e / (self.eps + m).pow(alpha) + delta).pow(r) - delta.pow(r)
        return out.transpose(1, 2)


def freq_mixstyle(x: torch.Tensor, p: float = 0.4, alpha: float = 0.3) -> torch.Tensor:
    """Frequency-wise MixStyle on log-mel features.

    Each clip's per-frequency-band statistics (computed over time) carry the recording
    channel: microphone response, room, device.  Swapping those statistics between clips
    in a batch, while keeping the normalised content, presents the same acoustic content
    under another recording's channel - so the model cannot rely on channel cues.

    This is the DCASE Task 1 device-mismatch remedy, where it took unseen-device accuracy
    from 62.0% to 67.7%.  Our failure is the same shape: submission A showed a model
    trained without 7 districts lost 0.0145 natural F1 on a test set drawn from unseen
    districts, i.e. it had been leaning on channel characteristics that did not transfer.

    Applied to the *input* features only, so frame targets stay valid - unlike waveform
    mixup, which the same literature found harmful for unseen-device generalisation.
    """
    if not x.requires_grad and not torch.is_grad_enabled():
        return x
    if torch.rand(()).item() > p:
        return x
    b = x.shape[0]
    if b < 2:
        return x
    # statistics over time, per frequency bin
    mu = x.mean(dim=2, keepdim=True)
    sig = x.std(dim=2, keepdim=True) + 1e-5
    x_norm = (x - mu) / sig

    lam = torch.distributions.Beta(alpha, alpha).sample((b,)).to(x.device)
    lam = lam.view(b, 1, 1, 1)
    perm = torch.randperm(b, device=x.device)
    mu_mix = lam * mu + (1 - lam) * mu[perm]
    sig_mix = lam * sig + (1 - lam) * sig[perm]
    return x_norm * sig_mix + mu_mix


def random_eq(x: torch.Tensor, max_db: float = 6.0, n_ctrl: int = 8) -> torch.Tensor:
    """Smooth random spectral tilt, simulating a different microphone response.

    The features are log-mel in dB, so a smooth additive curve across frequency is a
    multiplicative EQ on the spectrum.  Smooth rather than per-bin noise, because a device
    response is smooth; per-bin jitter would just be noise the model learns to ignore.
    """
    b, _, t, f = x.shape
    ctrl = (torch.rand(b, 1, 1, n_ctrl, device=x.device) * 2 - 1) * max_db
    curve = torch.nn.functional.interpolate(
        ctrl.view(b, 1, n_ctrl), size=f, mode="linear", align_corners=True
    ).view(b, 1, 1, f)
    return x + curve


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool_f: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
        # pooling only over frequency - time resolution is the whole point
        self.pool = nn.AvgPool2d((1, pool_f)) if pool_f > 1 else nn.Identity()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


def linear_softmax_pool(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum(p^2) / sum(p) over valid frames.

    Wang, Li & Metze 2019 compared five MIL pooling functions and linear-softmax gave
    the best *frame-level* F1 (21.8% vs 16.0% for attention), which is what matters here
    - Bronze exists to sharpen localisation, not to improve clip tagging.
    """
    mask = mask.unsqueeze(-1) if probs.dim() == 3 else mask
    num = (probs * probs * mask).sum(dim=1)
    den = (probs * mask).sum(dim=1).clamp(min=1e-6)
    return num / den


class CRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        channels: tuple[int, ...] = (32, 64, 128, 128, 128, 128, 128),
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        extra_dim: int = 0,
        fms_prob: float = 0.0,
        fms_alpha: float = 0.3,
        eq_prob: float = 0.0,
        eq_max_db: float = 6.0,
        epn_hidden: int = 128,
        in_ch: int = 1,
        unet_ckpt: str = str(PRETRAINED / "unet_v4.pt"),
        conv_heads: bool = False,
        bdist_k: int = 0,
        epn_bins: int = 0,
        frontend: str = "logmel",
        occ_hidden: int = 0,
    ):
        super().__init__()
        self.frontend = frontend
        self.res_ch = in_ch == 2
        if frontend == "both":
            self.pcen = PCEN(n_mels)
            in_ch = in_ch + 1
        self.conv_heads = conv_heads
        self.bdist_k = bdist_k
        self.logmel = LogMel(n_mels=n_mels)
        # in_ch 2 = (mixture, enhancement residual): the noise view without the speech.  The
        # residual is precomputed for training and computed on the fly (same U-Net) otherwise.
        self.in_ch = in_ch
        self.unet_ckpt = unet_ckpt
        self._unet = None
        self.input_norm = nn.BatchNorm2d(in_ch)
        self.extra_dim = extra_dim
        # domain-generalisation augmentation, applied to log-mel inputs during training
        self.fms_prob, self.fms_alpha = fms_prob, fms_alpha
        self.eq_prob, self.eq_max_db = eq_prob, eq_max_db

        blocks, cin = [], in_ch
        for cout in channels:
            blocks.append(ConvBlock(cin, cout, pool_f=2, dropout=dropout))
            cin = cout
        self.conv = nn.Sequential(*blocks)

        freq_out = n_mels // (2 ** len(channels))
        assert freq_out >= 1, "too many frequency pools for this many mel bins"
        feat_dim = channels[-1] * freq_out

        self.conv_out_dim = feat_dim
        self.rnn = nn.GRU(
            feat_dim + extra_dim, rnn_hidden, num_layers=rnn_layers, batch_first=True,
            bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * rnn_hidden, 1)
        self.head_cat = nn.Linear(2 * rnn_hidden, len(CATEGORIES))
        # dedicated boundary heads: Event-F1 is decided by onset/offset placement inside
        # a ~90 ms collar, which a frame-activity target never supervises directly
        self.head_bnd = nn.Linear(2 * rnn_hidden, 2)
        # annotation convention as a latent: Silver/pseudo tiers mark ~2x longer events than Gold,
        # so they supervise a second pair of activity/boundary heads and inference uses the Gold pair
        if conv_heads:
            self.head_alt = nn.Linear(2 * rnn_hidden, 1)
            self.head_bnd_alt = nn.Linear(2 * rnn_hidden, 2)
        # distributional boundary head: each frame emits a categorical over where the nearest onset
        # (and offset) lies within +-bdist_k frames; the per-frame distributions are aggregated into
        # onset/offset curves so every downstream consumer of the boundary heads is unchanged
        if bdist_k > 0:
            self.head_bdist = nn.Linear(2 * rnn_hidden, 2 * (2 * bdist_k + 1))
        # 30.6% of event pairs in this corpus overlap and a single binary activity curve
        # cannot represent two simultaneous events - one contiguous run is its correct
        # output.  This second channel carries the layer-2 (overlapping) events, which
        # 96.5% of overlapping clips need only one of.
        self.head_l2 = nn.Linear(2 * rnn_hidden, 1)
        # event proposal network (Schmid et al. 2026): every frame regresses the interval
        # of the event covering it, decoded by greedy selection instead of thresholding
        self.head_epn = EPNHead(2 * rnn_hidden + 3, hidden=epn_hidden, n_bins=epn_bins)
        # low-rate occupancy branch: 100 ms pooled inputs through a clip-long BiGRU, added to the activity
        # logit; the zero-initialised output makes a warm start exactly the parent model
        self.occ_hidden = occ_hidden
        if occ_hidden > 0:
            self.occ_rnn = nn.GRU(feat_dim + extra_dim, occ_hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.occ_out = nn.Linear(2 * occ_hidden, 1)
            nn.init.zeros_(self.occ_out.weight)
            nn.init.zeros_(self.occ_out.bias)

    def conv_features(self, wav: torch.Tensor, n_frames: int | None = None,
                      res: torch.Tensor | None = None) -> torch.Tensor:
        x = self.logmel(wav, n_frames).unsqueeze(1)          # (B, 1, T, M)
        if self.res_ch:
            if res is None:
                res = self.residual(wav)
            x = torch.cat([x, self.logmel(res, n_frames).unsqueeze(1)], dim=1)   # (B, 2, T, M)
        if self.frontend == "both":
            p = self.pcen(self.logmel.mel(wav))
            if p.shape[1] > x.shape[2]:
                p = p[:, : x.shape[2]]
            elif p.shape[1] < x.shape[2]:
                p = torch.cat([p, p[:, -1:].expand(-1, x.shape[2] - p.shape[1], -1)], dim=1)
            x = torch.cat([x, p.unsqueeze(1)], dim=1)
        if self.training:
            # channel-level augmentation goes before input normalisation, so the
            # normaliser sees the augmented distribution rather than undoing it
            if self.eq_prob > 0 and torch.rand(()).item() < self.eq_prob:
                x = random_eq(x, self.eq_max_db)
            if self.fms_prob > 0:
                x = freq_mixstyle(x, self.fms_prob, self.fms_alpha)
        x = self.input_norm(x)
        x = self.conv(x)                        # (B, C, T, M')
        b, c, t, m = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, c * m)

    @torch.no_grad()
    def residual(self, wav: torch.Tensor) -> torch.Tensor:
        """Mixture minus the Track 2 U-Net's speech estimate (identical to make_residuals.py)."""
        if self._unet is None:
            from track2_unet import TorchSTFT, UNetMasker
            ck = torch.load(self.unet_ckpt, map_location="cpu")
            ch = tuple(int(c) for c in ck["args"]["channels"].split(","))
            unet = UNetMasker(floor=ck.get("floor", 0.05), ch=ch)
            unet.load_state_dict(ck["model"])
            self._unet = unet.to(wav.device).eval()
            self._stft = TorchSTFT().to(wav.device)
        with torch.autocast("cuda", enabled=False):
            w = wav.float()
            spec = self._stft.stft(w)
            lm = torch.log1p(spec.abs()).transpose(1, 2).unsqueeze(1)
            g = self._unet(lm)
            g01 = ((g - self._unet.floor) / (1.0 - self._unet.floor)).clamp(0, 1)
            speech = self._stft.istft(spec * g01.transpose(1, 2), length=w.shape[-1])
            return w - speech

    def heads(self, x: torch.Tensor):
        # cuDNN has no bf16 GRU kernel: a bf16 input silently falls back to unrolling
        # every timestep, which measured 2.611 s vs 0.011 s per step at T=1000 - a 237x
        # penalty.  Run the recurrence in fp32 with autocast disabled.
        with torch.autocast("cuda", enabled=False):
            x_in = x.float()
            x, _ = self.rnn(x_in)
            act = self.head(x).squeeze(-1)
            if self.occ_hidden > 0:
                z = torch.nn.functional.avg_pool1d(x_in.transpose(1, 2), 10, 10, ceil_mode=True).transpose(1, 2)
                z, _ = self.occ_rnn(z)
                occ = self.occ_out(z).transpose(1, 2)
                act = act + torch.nn.functional.interpolate(occ, size=x_in.shape[1], mode="linear", align_corners=False).squeeze(1)
            bnd = self.head_bnd(x)
            if self.bdist_k > 0:
                K = self.bdist_k
                lg = self.head_bdist(x).view(x.shape[0], x.shape[1], 2, 2 * K + 1)
                self._bdist = lg
                p = lg.softmax(-1) * torch.sigmoid(act).detach()[..., None, None]
                agg = torch.zeros_like(p[..., 0])
                for k in range(2 * K + 1):
                    d = k - K
                    if d >= 0:
                        agg[:, d:, :] = agg[:, d:, :] + p[:, : p.shape[1] - d, :, k]
                    else:
                        agg[:, :d, :] = agg[:, :d, :] + p[:, -d:, :, k]
                a = (agg / (K + 1)).clamp(1e-4, 1 - 1e-4)
                bnd = torch.log(a) - torch.log1p(-a)
            if self.conv_heads:
                bnd_alt = self.head_bnd_alt(x)
                self._alt = (self.head_alt(x).squeeze(-1), bnd_alt[..., 0], bnd_alt[..., 1])
            curves = torch.sigmoid(torch.stack([act, bnd[..., 0], bnd[..., 1]], dim=-1))
            epn = self.head_epn(torch.cat([x, curves], dim=-1))
        return (act, self.head_cat(x), bnd[..., 0], bnd[..., 1],
                self.head_l2(x).squeeze(-1), epn)

    def forward(self, wav: torch.Tensor, n_frames: int | None = None, res: torch.Tensor | None = None):
        return self.heads(self.conv_features(wav, n_frames, res))


class CRNNBeats(CRNN):
    """CRNN with a frozen AudioSet-pretrained BEATs branch fused before the recurrence.

    This is the DCASE 2024 winner's shape: semantics from an audio-event transformer,
    time resolution from the convolutional stack.  BEATs alone could not be used as the
    frontend here - it emits 20 ms frames against a ~90 ms collar, and the diagnosis
    showed our losses are boundary placement and short events, both of which need the
    10 ms grid.  As a second branch that objection disappears: BEATs supplies "what this
    sounds like" at 20 ms and is interpolated onto the CRNN's 10 ms grid, while the conv
    stack keeps supplying "exactly when".

    BEATs is frozen by default (the winner's stage 1), so it costs a forward pass only
    and its 90 M parameters never receive gradients.
    """

    def __init__(self, *args, beats_ckpt: str = "", beats_dim: int = 256,
                 freeze_beats: bool = True, **kwargs):
        super().__init__(*args, extra_dim=beats_dim, **kwargs)
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).parent / "beats"))
        from BEATs import BEATs, BEATsConfig

        ck = torch.load(beats_ckpt, map_location="cpu", weights_only=False)
        self.beats = BEATs(BEATsConfig(ck["cfg"]))
        self.beats.load_state_dict(ck["model"])
        self.freeze_beats = freeze_beats
        if freeze_beats:
            self.beats.eval()
            for p in self.beats.parameters():
                p.requires_grad_(False)

        self.beats_proj = nn.Sequential(
            nn.LayerNorm(self.beats.cfg.encoder_embed_dim),
            nn.Linear(self.beats.cfg.encoder_embed_dim, beats_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_beats:
            self.beats.eval()   # keep frozen norms in inference mode
        return self

    def beats_features(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        # nullcontext, not enable_grad: an unfrozen encoder must still respect an outer no_grad at inference
        ctx = torch.no_grad() if self.freeze_beats else contextlib.nullcontext()
        with ctx:
            feats, _ = self.beats.extract_features(
                wav, padding_mask=torch.zeros_like(wav, dtype=torch.bool)
            )
        # BEATs runs at ~20 ms; interpolate onto the CRNN's 10 ms grid
        feats = torch.nn.functional.interpolate(
            feats.transpose(1, 2).float(), size=n_frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.beats_proj(feats)

    def forward(self, wav: torch.Tensor, n_frames: int | None = None, res: torch.Tensor | None = None):
        conv = self.conv_features(wav, n_frames, res)
        b = self.beats_features(wav, conv.shape[1])
        return self.heads(torch.cat([conv, b], dim=-1))

class CRNNBeatsSpeech(CRNNBeats):
    """CRNNBeats plus a frozen Vaani-finetuned Whisper encoder branch (Indic speech knowledge:
    what the speaker's speech sounds like, so speaker-produced non-speech becomes separable).
    20 ms states are interpolated onto the 10 ms grid exactly like the BEATs branch."""

    def __init__(self, *args, speech_ckpt: str = "", speech_dim: int = 128, beats_dim: int = 256,
                 **kwargs):
        super().__init__(*args, beats_dim=beats_dim, **kwargs)
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
        self.speech_ckpt = speech_ckpt
        self.extra_dim = beats_dim + speech_dim
        self.rnn = nn.GRU(self.conv_out_dim + self.extra_dim, self.rnn.hidden_size,
                          num_layers=self.rnn.num_layers, batch_first=True, bidirectional=True,
                          dropout=self.rnn.dropout)
        fe = WhisperFeatureExtractor.from_pretrained(speech_ckpt)
        self.register_buffer("speech_mel", torch.tensor(fe.mel_filters, dtype=torch.float32))
        self.register_buffer("speech_window", torch.hann_window(fe.n_fft))
        self.speech_n_fft, self.speech_hop, self.speech_n_frames = fe.n_fft, fe.hop_length, fe.nb_max_frames
        self.speech = WhisperForConditionalGeneration.from_pretrained(speech_ckpt).model.encoder
        self.speech.eval()
        for p in self.speech.parameters():
            p.requires_grad_(False)
        self.speech_proj = nn.Sequential(
            nn.LayerNorm(self.speech.config.d_model),
            nn.Linear(self.speech.config.d_model, speech_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.speech.eval()
        return self

    def speech_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """Whisper's own log-mel front end, on the GPU, padded to its fixed 30 s input."""
        with torch.autocast("cuda", enabled=False):
            wav = wav.float()
            spec = torch.stft(wav, self.speech_n_fft, self.speech_hop, window=self.speech_window,
                              return_complex=True)
            mag = spec[..., :-1].abs() ** 2                              # (B, F, T)
            mel = self.speech_mel.T @ mag                                # (B, M, T)
            log_spec = torch.clamp(mel, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            pad = self.speech_n_frames - log_spec.shape[-1]
            if pad < 0:
                log_spec = log_spec[..., :self.speech_n_frames]
            elif pad > 0:
                log_spec = torch.nn.functional.pad(log_spec, (0, pad))
        return log_spec

    def speech_features(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        with torch.no_grad():
            feats = self.speech(self.speech_logmel(wav)).last_hidden_state   # (B, 1500, D)
            keep = min(feats.shape[1], max(1, -(-wav.shape[1] // (2 * self.speech_hop))))
            feats = feats[:, :keep]
        feats = torch.nn.functional.interpolate(
            feats.transpose(1, 2).float(), size=n_frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.speech_proj(feats)

    def forward(self, wav: torch.Tensor, n_frames: int | None = None):
        conv = self.conv_features(wav, n_frames)
        b = self.beats_features(wav, conv.shape[1])
        sp = self.speech_features(wav, conv.shape[1])
        return self.heads(torch.cat([conv, b, sp], dim=-1))


class CRNNAtst(CRNN):
    """CRNN with 40 ms ATST-Frame embeddings fused before the recurrence."""

    def __init__(self, *args, atst_ckpt: str = "", atst_dim: int = 256,
                 freeze_atst: bool = True, **kwargs):
        super().__init__(*args, extra_dim=atst_dim, **kwargs)
        from atst import FrameAtstEncoder

        self.atst = FrameAtstEncoder()
        self.atst.load_checkpoint(atst_ckpt)
        self.freeze_atst = freeze_atst
        if freeze_atst:
            self.atst.eval()
            for p in self.atst.parameters():
                p.requires_grad_(False)
        self.atst_proj = nn.Sequential(
            nn.LayerNorm(self.atst.embed_dim),
            nn.Linear(self.atst.embed_dim, atst_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_atst:
            self.atst.eval()
        return self

    def atst_features(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_atst else contextlib.nullcontext()
        with ctx:
            feats = self.atst(wav.float())
        feats = torch.nn.functional.interpolate(
            feats.transpose(1, 2).float(), size=n_frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.atst_proj(feats)

    def forward(self, wav: torch.Tensor, n_frames: int | None = None, res: torch.Tensor | None = None):
        conv = self.conv_features(wav, n_frames, res)
        atst = self.atst_features(wav, conv.shape[1])
        return self.heads(torch.cat([conv, atst], dim=-1))


class CRNNSedEnc(CRNN):
    """CRNN with frozen AudioSet-Strong PretrainedSED embeddings fused before the GRU."""

    def __init__(self, *args, sed_encoder: str, sed_ckpt: str, sed_dim: int = 256,
                 freeze_sed: bool = True, **kwargs):
        super().__init__(*args, extra_dim=sed_dim, **kwargs)
        from sed_encoders import PretrainedSedEncoder

        self.sed = PretrainedSedEncoder(sed_encoder, sed_ckpt)
        self.freeze_sed = freeze_sed
        if freeze_sed:
            self.sed.eval()
            for name, p in self.sed.named_parameters():
                p.requires_grad_(name == "layer_weights")
        self.sed_proj = nn.Sequential(
            nn.LayerNorm(self.sed.embed_dim),
            nn.Linear(self.sed.embed_dim, sed_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_sed:
            self.sed.eval()
        return self

    def sed_features(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_sed and self.sed.encoder != "wavlm" else contextlib.nullcontext()
        with ctx:
            feats = self.sed(wav)
        feats = torch.nn.functional.interpolate(
            feats.transpose(1, 2).float(), size=n_frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.sed_proj(feats)

    def forward(self, wav: torch.Tensor, n_frames: int | None = None, res: torch.Tensor | None = None):
        conv = self.conv_features(wav, n_frames, res)
        sed = self.sed_features(wav, conv.shape[1])
        return self.heads(torch.cat([conv, sed], dim=-1))


def tta_forward(model, wav: torch.Tensor, n_frames: int | None = None, res: torch.Tensor | None = None,
                shift: int = HOP // 2):
    """Half-hop test-time augmentation: run the model on the clip and on the clip delayed by `shift`
    samples, align the delayed outputs back onto the frame centres (a half-frame interpolation) and
    average the two passes, which doubles the effective temporal resolution of every head."""
    out = model(wav, n_frames, res)
    wav2 = torch.nn.functional.pad(wav, (shift, 0))[:, : wav.shape[1]]
    res2 = torch.nn.functional.pad(res, (shift, 0))[:, : res.shape[1]] if res is not None else None
    out2 = model(wav2, n_frames, res2)

    def align(x):
        if x is None or not torch.is_tensor(x) or x.ndim < 2:
            return x
        nxt = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        return 0.5 * (x + nxt)

    return tuple(0.5 * (a + align(b)) if torch.is_tensor(a) else a for a, b in zip(out, out2))


def load_checkpoint(path: str, device: str, use_ema: bool = True) -> tuple[CRNN, dict]:
    """Rebuild the exact architecture recorded in the checkpoint, then load weights.

    The architecture must come from the checkpoint's own args rather than from defaults,
    otherwise any run with a non-default width silently fails to load here.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    for k in ("sed_ckpt", "beats_ckpt", "atst_ckpt", "speech_ckpt", "unet_ckpt"):
        if isinstance(cargs.get(k), str) and cargs[k].startswith("pretrained/"):
            cargs[k] = str(PRETRAINED / cargs[k][len("pretrained/"):])
    channels = tuple(
        int(c) for c in str(cargs.get("channels", "32,64,128,128,128,128,128")).split(",")
    )
    common = dict(channels=channels, rnn_hidden=int(cargs.get("rnn_hidden", 128)),
                  dropout=float(cargs.get("dropout", 0.1)),
                  fms_prob=float(cargs.get("fms_prob", 0.0)),
                  fms_alpha=float(cargs.get("fms_alpha", 0.3)),
                  eq_prob=float(cargs.get("eq_prob", 0.0)),
                  eq_max_db=float(cargs.get("eq_max_db", 6.0)),
                  epn_hidden=int(cargs.get("epn_hidden", 128)),
                  in_ch=2 if cargs.get("residual_root") else 1,
                  conv_heads=bool(cargs.get("conv_heads", False)),
                  bdist_k=int(cargs.get("bdist_k", 0) or 0),
                  epn_bins=int(cargs.get("epn_bins", 0) or 0),
                  frontend=cargs.get("frontend") or "logmel",
                  occ_hidden=int(cargs.get("occ_hidden", 0) or 0),
                  unet_ckpt=cargs.get("unet_ckpt") or str(PRETRAINED / "unet_v4.pt"))
    if cargs.get("sed_ckpt"):
        model = CRNNSedEnc(sed_encoder=cargs["sed_encoder"], sed_ckpt=cargs["sed_ckpt"],
                           sed_dim=int(cargs.get("sed_dim", 256)),
                           freeze_sed=not cargs.get("unfreeze_sed", False),
                           **common).to(device)
    elif cargs.get("atst_ckpt"):
        model = CRNNAtst(atst_ckpt=cargs["atst_ckpt"],
                         atst_dim=int(cargs.get("atst_dim", 256)),
                         freeze_atst=not cargs.get("unfreeze_atst", False),
                         **common).to(device)
    elif cargs.get("beats_ckpt") and cargs.get("speech_ckpt"):
        model = CRNNBeatsSpeech(beats_ckpt=cargs["beats_ckpt"], speech_ckpt=cargs["speech_ckpt"],
                                speech_dim=int(cargs.get("speech_dim", 128)),
                                beats_dim=int(cargs.get("beats_dim", 256)),
                                freeze_beats=not cargs.get("unfreeze_beats", False),
                                **common).to(device)
    elif cargs.get("beats_ckpt"):
        model = CRNNBeats(beats_ckpt=cargs["beats_ckpt"],
                          beats_dim=int(cargs.get("beats_dim", 256)),
                          freeze_beats=not cargs.get("unfreeze_beats", False),
                          **common).to(device)
    else:
        model = CRNN(**common).to(device)
    state = ckpt["ema"] if (use_ema and "ema" in ckpt) else ckpt["model"]

    # Checkpoints trained before the boundary heads existed have no head_bnd weights.
    # They load with the head left at its random init, which is harmless *only* because
    # such a model is used for its activity output alone - the flag makes that explicit
    # so a caller can refuse to use its boundary predictions.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {path}: {unexpected}")
    has_boundary_heads = True
    has_epn = not any(k.startswith("head_epn.") for k in missing)
    missing = [k for k in missing if not k.startswith("head_epn.")]
    if missing:
        if all(k.startswith("head_l2.") for k in missing):
            print(f"note: {path} predates the layer-2 head; its overlap channel is "
                  f"untrained and must not be used")
        elif all(k.startswith(("head_bnd.", "head_l2.")) for k in missing):
            has_boundary_heads = False
            print(f"note: {path} predates the boundary heads; "
                  f"its onset/offset outputs are untrained and must not be used")
        else:
            raise RuntimeError(f"missing keys in {path}: {missing}")

    model.eval()
    ckpt["has_boundary_heads"] = has_boundary_heads
    ckpt["has_epn"] = has_epn
    return model, ckpt


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    padded = np.pad(x, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=-1)


def decode(
    probs: np.ndarray,
    threshold: float = 0.5,
    offset_threshold: float | None = None,
    median_k: int = 1,
    merge_gap: float = 0.0,
    min_dur: float = 0.0,
    onset_bias: float = 0.0,
    offset_bias: float = 0.0,
    duration: float | None = None,
) -> list[tuple[float, float]]:
    """Frame probabilities -> event spans.

    `offset_threshold` below `threshold` gives hysteresis: a run must rise above the
    onset threshold to start but only has to fall below the lower threshold to end.  It
    preserves an event's extent without bridging arbitrary gaps the way a long median
    filter does - which matters because 48.5% of clips hold more than one event and
    merging two references into one prediction costs two Event-F1 matches while only
    helping Dice.
    """
    if probs.size == 0:
        return []

    smoothed = median_filter(probs.astype(np.float64), median_k)
    lo = offset_threshold if offset_threshold is not None else threshold

    active = np.zeros(smoothed.shape[0], dtype=bool)
    on = False
    for i, p in enumerate(smoothed):
        if not on and p >= threshold:
            on = True
        elif on and p < lo:
            on = False
        active[i] = on
    if not active.any():
        return []

    padded = np.concatenate([[False], active, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    spans = [[float(s) * FRAME_SEC, float(e) * FRAME_SEC]
             for s, e in zip(edges[0::2], edges[1::2])]

    if merge_gap > 0:
        merged = [spans[0]]
        for onset, offset in spans[1:]:
            if onset - merged[-1][1] <= merge_gap:
                merged[-1][1] = offset
            else:
                merged.append([onset, offset])
        spans = merged

    out = []
    for onset, offset in spans:
        onset += onset_bias
        offset += offset_bias
        if duration is not None:
            onset = max(0.0, min(onset, duration))
            offset = max(0.0, min(offset, duration))
        if offset - onset >= max(min_dur, 1e-3):
            out.append((round(onset, 4), round(offset, 4)))
    return out


def snap_to_boundaries(
    spans: list[tuple[float, float]],
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,
    search_sec: float = 0.12,
    min_peak: float = 0.10,
) -> list[tuple[float, float]]:
    """Move each span's edges onto the nearest confident boundary-head peak.

    Thresholding an activity curve puts the edge where the activity crosses a level,
    which depends on how fast the event fades in - not on where it began.  The boundary
    heads are trained to fire at the transition itself, so a short local search around
    each edge recovers the placement Event-F1 is actually scoring.  The search window is
    kept near the collar; a peak below `min_peak` is ignored so an unconfident head never
    drags a good edge away.
    """
    if onset_probs.size == 0 or not spans:
        return spans
    n = onset_probs.shape[0]
    w = max(1, int(round(search_sec / FRAME_SEC)))

    def refine(t: float, probs: np.ndarray) -> float:
        i = int(round(t / FRAME_SEC))
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if hi <= lo:
            return t
        seg = probs[lo:hi]
        j = int(np.argmax(seg))
        return (lo + j) * FRAME_SEC if seg[j] >= min_peak else t

    out = []
    for onset, offset in spans:
        a, b = refine(onset, onset_probs), refine(offset, offset_probs)
        if b - a >= 1e-3:
            out.append((round(a, 4), round(b, 4)))
        else:
            out.append((onset, offset))
    return out
