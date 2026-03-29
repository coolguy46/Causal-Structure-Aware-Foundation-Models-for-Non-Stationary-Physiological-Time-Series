"""Spectral reconstruction loss: MSE in STFT domain."""
import torch
import torch.nn.functional as F


def spectral_reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 64,
) -> torch.Tensor:
    """Compute MSE loss between STFT magnitudes of predicted and target signals.

    Args:
        x_hat: (B, C, T) reconstructed signal
        x:     (B, C, T) original signal
        n_fft: STFT window size
        hop_length: STFT hop length
    Returns:
        Scalar loss
    """
    B, C, T = x.shape
    window = torch.hann_window(n_fft, device=x.device)

    # Flatten channels into batch for efficient STFT
    x_flat = x.reshape(B * C, T)
    x_hat_flat = x_hat.reshape(B * C, T)

    spec_target = torch.stft(
        x_flat, n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True, normalized=True,
    ).abs()

    spec_pred = torch.stft(
        x_hat_flat, n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True, normalized=True,
    ).abs()

    return F.mse_loss(spec_pred, spec_target)
