"""Spectral Tokenizer: converts raw biosignals into frequency-band tokens."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralTokenizer(nn.Module):
    """Converts raw multi-channel signals into spectral band tokens.

    For each channel and each physiological frequency band, computes STFT,
    extracts the relevant frequency bins, and projects to a fixed-dim token.

    Input:  (B, C, T) raw signal
    Output: (B, C * n_bands, d_token) spectral tokens
    """

    PHYSIOLOGICAL_BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0),
    }

    def __init__(
        self,
        n_channels: int,
        d_token: int = 128,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        sample_rate: float = 125.0,
        bands: Optional[dict[str, tuple[float, float]]] = None,
        n_temporal_pools: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_token = d_token
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.bands = bands or self.PHYSIOLOGICAL_BANDS
        self.n_bands = len(self.bands)
        self.n_temporal_pools = n_temporal_pools

        # Compute frequency bin indices for each band
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        self.band_masks = {}
        for name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs <= high)
            self.band_masks[name] = mask

        # Number of freq bins per band varies; we'll pool then project
        # Projection: pool across time frames, then linear projection
        n_freq_bins = n_fft // 2 + 1
        n_time_frames = None  # determined at runtime

        # Each band: extract masked freq bins -> avg pool -> project
        self.band_proj = nn.ModuleDict()
        for name, mask in self.band_masks.items():
            n_bins = mask.sum().item()
            if n_bins == 0:
                n_bins = 1  # fallback
            self.band_proj[name] = nn.Linear(n_bins, d_token)

        # Positional embeddings: channel ID + band ID
        self.channel_emb = nn.Embedding(n_channels, d_token)
        self.band_emb = nn.Embedding(self.n_bands, d_token)

        # Temporal position embeddings (when keeping multiple time pools)
        if n_temporal_pools > 1:
            self.temporal_emb = nn.Embedding(n_temporal_pools, d_token)

        # Register band masks as buffers
        for i, (name, mask) in enumerate(self.band_masks.items()):
            self.register_buffer(f"mask_{name}", mask)

        # STFT window
        self.register_buffer("window", torch.hann_window(win_length))

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw signal
        Returns:
            tokens: (B, C * n_bands, d_token)
        """
        B, C, T = x.shape
        device = x.device

        # Reshape to (B*C, T) for batch STFT
        x_flat = x.reshape(B * C, T)

        # torch.stft/complex kernels are not consistently supported for bf16
        # under torch.compile+inductor. Keep spectral ops in fp32 for stability.
        x_flat = x_flat.float()

        # STFT: output (B*C, n_freq, n_frames) complex
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            normalized=True,
        )
        # Magnitude spectrogram: (B*C, n_freq, n_frames)
        mag = spec.abs()

        # Pool time frames
        if self.n_temporal_pools > 1:
            # Keep multiple temporal buckets via adaptive average pooling
            mag_pooled = F.adaptive_avg_pool1d(mag, self.n_temporal_pools)
        else:
            # Average all time frames (original behavior)
            mag_pooled = mag.mean(dim=-1, keepdim=True)  # (B*C, n_freq, 1)

        n_tp = mag_pooled.shape[-1]
        # (B, C, n_freq, n_tp)
        mag_pooled = mag_pooled.reshape(B, C, -1, n_tp)

        # Extract per-band tokens
        tokens_list = []
        for band_idx, (name, _) in enumerate(self.bands.items()):
            mask = getattr(self, f"mask_{name}").to(device)
            # (B, C, n_bins, n_tp) -> (B, C, n_tp, n_bins)
            band_feat = mag_pooled[:, :, mask, :].permute(0, 1, 3, 2)
            # Project: (B, C, n_tp, d_token)
            band_token = self.band_proj[name](band_feat)

            # Add positional embeddings
            ch_ids = torch.arange(C, device=device)
            band_id = torch.tensor(band_idx, device=device)
            band_token = band_token + self.channel_emb(ch_ids)[None, :, None, :]
            band_token = band_token + self.band_emb(band_id)

            if self.n_temporal_pools > 1:
                temp_ids = torch.arange(n_tp, device=device)
                band_token = band_token + self.temporal_emb(temp_ids)[None, None, :, :]

            tokens_list.append(band_token)

        # Stack: (B, C, n_bands, n_tp, d_token) -> (B, C*n_bands*n_tp, d_token)
        tokens = torch.stack(tokens_list, dim=2)  # (B, C, n_bands, n_tp, d_token)
        tokens = tokens.reshape(B, C * self.n_bands * n_tp, self.d_token)

        return tokens

    @property
    def n_tokens(self) -> int:
        return self.n_channels * self.n_bands * self.n_temporal_pools


class TokenDecoder(nn.Module):
    """Decodes spectral tokens back to raw signal for reconstruction loss.

    Input:  (B, C * n_bands, d_token)
    Output: (B, C, T) reconstructed signal
    """

    def __init__(
        self,
        n_channels: int,
        d_token: int = 128,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        output_length: int = 500,
        sample_rate: float = 125.0,
        bands: Optional[dict[str, tuple[float, float]]] = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_token = d_token
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.output_length = output_length

        n_freq = n_fft // 2 + 1
        # torch.stft uses center=True by default, which pads the signal
        # and produces output_length // hop_length + 1 frames
        n_frames = output_length // hop_length + 1

        # Project tokens back to full spectrogram
        n_bands = len(bands or SpectralTokenizer.PHYSIOLOGICAL_BANDS)
        self.deproj = nn.Linear(d_token * n_bands, n_freq * n_frames)
        self.n_freq = n_freq
        self.n_frames = n_frames

        self.register_buffer("window", torch.hann_window(win_length))

    @torch.compiler.disable
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, C * n_bands, d_token)
        Returns:
            signal: (B, C, T) reconstructed
        """
        B = tokens.shape[0]
        n_bands = tokens.shape[1] // self.n_channels

        # Reshape to (B, C, n_bands * d_token)
        tokens = tokens.reshape(B, self.n_channels, n_bands * self.d_token)

        # Project to spectrogram shape: (B, C, n_freq * n_frames)
        spec_flat = self.deproj(tokens)

        # Reshape: (B*C, n_freq, n_frames)
        spec = spec_flat.reshape(B * self.n_channels, self.n_freq, self.n_frames)

        # complex() does not accept bf16 tensors; cast to fp32 first.
        spec = spec.float()

        # Convert magnitude to complex (zero phase for reconstruction)
        spec_complex = torch.complex(spec, torch.zeros_like(spec))

        # iSTFT
        signal = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=self.output_length,
        )

        # (B, C, T)
        signal = signal.reshape(B, self.n_channels, self.output_length)
        return signal
