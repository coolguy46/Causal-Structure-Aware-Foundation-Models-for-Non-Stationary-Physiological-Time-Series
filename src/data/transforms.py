"""Signal transforms: z-score normalization, bandpass filtering, artifact rejection."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


class ZScoreNormalize:
    """Per-channel z-score normalization."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Args: x shape (C, T). Returns normalized x."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (x - mean) / std


class BandpassFilter:
    """Butterworth bandpass filter applied per-channel."""

    def __init__(self, low: float, high: float, fs: float, order: int = 5):
        self.sos = butter(order, [low, high], btype="band", fs=fs, output="sos")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Args: x shape (C, T). Returns filtered x."""
        return sosfiltfilt(self.sos, x, axis=-1).astype(np.float32)


class ArtifactRejection:
    """Simple amplitude-threshold artifact rejection.

    Replaces windows exceeding amplitude threshold with zeros.
    """

    def __init__(self, threshold_uv: float = 500.0):
        self.threshold = threshold_uv

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Args: x shape (C, T). Returns x with artifacts zeroed."""
        max_amp = np.abs(x).max(axis=-1)  # (C,)
        mask = max_amp > self.threshold
        x[mask] = 0.0
        return x
