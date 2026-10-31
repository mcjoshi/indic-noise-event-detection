"""PretrainedSED waveform encoders for CRNN feature fusion."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import torch
import torch.nn as nn


_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "PretrainedSED"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class PretrainedSedEncoder(nn.Module):
    """AudioSet-Strong backbone returning native-rate frame embeddings."""

    _SPECS = {
        "fpasst": ("fpasst_strong_1.pt", 768, 250),
        "m2d": ("M2D_strong_1.pt", 3840, 62),
        "asit": ("ASIT_strong_1.pt", 768, 497),
        "atstf": ("ATST-F_strong_1.pt", 768, 250),
        "wavlm": ("", 768, 500),
        "wavlm_large": ("", 1024, 500),
        "xlsr": ("", 1024, 500),
        "hubert_large": ("", 1024, 500),
        "dasheng": ("", 768, 250),
        "dasheng06b": ("", 1280, 250),
        "whisper_medium": ("", 1024, 1500),
        "whisper_large": ("", 1280, 1500),
    }
    HF_FAMILY = ("wavlm", "wavlm_large", "xlsr", "hubert_large")
    WHISPER = ("whisper_medium", "whisper_large")
    segment_samples = 160_000

    def __init__(self, encoder: str, checkpoint: str):
        super().__init__()
        if encoder not in self._SPECS:
            raise ValueError(f"unknown PretrainedSED encoder: {encoder}")
        self.encoder = encoder
        self.checkpoint = checkpoint
        _, self.embed_dim, self.frames_per_segment = self._SPECS[encoder]
        self.backbone = self._build_backbone()
        if encoder in self.HF_FAMILY:
            self.layer_weights = nn.Parameter(torch.zeros(self.backbone.config.num_hidden_layers + 1))
        if encoder in self.WHISPER:
            # Whisper's encoder takes a fixed 30 s log-mel window; one window covers a whole <= 15 s clip
            from transformers import WhisperFeatureExtractor
            self.segment_samples = 480_000
            self.layer_weights = nn.Parameter(torch.zeros(self.backbone.config.encoder_layers + 1))
            mel = WhisperFeatureExtractor.from_pretrained(checkpoint).mel_filters
            self.register_buffer("mel_filters", torch.tensor(mel, dtype=torch.float32), persistent=False)
            self.register_buffer("mel_window", torch.hann_window(400), persistent=False)
        self.load_checkpoint(checkpoint)

    def _build_backbone(self) -> nn.Module:
        if self.encoder in self.WHISPER:
            from transformers import WhisperModel
            return WhisperModel.from_pretrained(self.checkpoint, torch_dtype=torch.float32).encoder
        if self.encoder in self.HF_FAMILY:
            from transformers import AutoModel
            return AutoModel.from_pretrained(self.checkpoint, output_hidden_states=True)
        if self.encoder in ("dasheng", "dasheng06b"):
            from dasheng import dasheng_06B, dasheng_base
            return dasheng_06B() if self.encoder == "dasheng06b" else dasheng_base()
        if self.encoder == "fpasst":
            from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
            return FPaSSTWrapper()
        if self.encoder == "m2d":
            from models.m2d.M2D_wrapper import M2DWrapper
            return M2DWrapper()
        if self.encoder == "asit":
            from models.asit.ASIT_wrapper import ASiTWrapper
            return ASiTWrapper()
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return ATSTWrapper()

    def load_checkpoint(self, path: str) -> None:
        checkpoint = Path(path)
        self.checkpoint = str(checkpoint)
        if self.encoder in self.HF_FAMILY or self.encoder in self.WHISPER:
            return
        if self.encoder in ("dasheng", "dasheng06b"):
            # empty path keeps the released weights; otherwise a localisation-pretrained state dict
            if checkpoint.is_file():
                self.backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            return
        if checkpoint.name.endswith("_loc.pt"):
            # localisation-pretext output: the backbone's own state dict
            self.backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            return
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            name = self._SPECS[self.encoder][0]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/{name}"
            subprocess.run(["curl", "-fL", "-o", str(checkpoint), url], check=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = {
            "fpasst": "fpasst.",
            "m2d": "m2d.",
            "asit": "asit.",
            "atstf": "",
        }[self.encoder]
        state = {
            prefix + key[len("model."):]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        missing = [key for key in missing if "mel" not in key]
        if missing or unexpected:
            raise RuntimeError(
                f"PretrainedSED {self.encoder} checkpoint mismatch: "
                f"missing {missing}, unexpected {unexpected}"
            )

    def _whisper_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """WhisperFeatureExtractor's log-mel, in torch: 25 ms Hann / 10 ms hop, log10, 8-decade floor, (x + 4) / 4."""
        stft = torch.stft(wav.float(), 400, 160, window=self.mel_window, return_complex=True)
        power = stft[..., :-1].abs() ** 2
        logmel = torch.clamp(self.mel_filters.T @ power, min=1e-10).log10()
        logmel = torch.maximum(logmel, logmel.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (logmel + 4.0) / 4.0

    def _forward_segment(self, wav: torch.Tensor) -> torch.Tensor:
        if self.encoder in self.WHISPER:
            hidden_states = self.backbone(self._whisper_logmel(wav), output_hidden_states=True).hidden_states
            weights = self.layer_weights.softmax(dim=0).to(hidden_states[0].dtype)
            return sum(weight * state for weight, state in zip(weights, hidden_states))
        if self.encoder in self.HF_FAMILY:
            hidden_states = self.backbone(wav, output_hidden_states=True).hidden_states
            weights = self.layer_weights.softmax(dim=0).to(hidden_states[0].dtype)
            return sum(weight * state for weight, state in zip(weights, hidden_states))
        if self.encoder in ("dasheng", "dasheng06b"):
            return self.backbone(wav.float())
        mel = self.backbone.mel_forward(wav.float())
        return self.backbone(mel)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self.encoder in self.HF_FAMILY:
            wav = wav.float()
            wav = (wav - wav.mean(dim=1, keepdim=True)) / wav.std(dim=1, keepdim=True).clamp(min=1e-5)
        chunks = []
        for chunk in wav.split(self.segment_samples, dim=1):
            samples = chunk.shape[1]
            if samples < self.segment_samples:
                chunk = torch.nn.functional.pad(chunk, (0, self.segment_samples - samples))
            embeddings = self._forward_segment(chunk)
            keep = max(1, round(self.frames_per_segment * samples / self.segment_samples))
            chunks.append(embeddings[:, :keep])
        return torch.cat(chunks, dim=1)
