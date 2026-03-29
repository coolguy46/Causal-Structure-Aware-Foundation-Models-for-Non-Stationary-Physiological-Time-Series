"""Subject-stratified train/val/test splits."""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def subject_stratified_split(
    subject_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_path: str | None = None,
) -> dict[str, list[str]]:
    """Split subjects into train/val/test ensuring no subject leakage.

    Args:
        subject_ids: List of unique subject identifiers.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.
        output_path: If provided, save splits as JSON.

    Returns:
        Dictionary with 'train', 'val', 'test' lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    ids = np.array(subject_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train : n_train + n_val].tolist(),
        "test": ids[n_train + n_val :].tolist(),
    }

    logger.info(
        f"Split {n} subjects -> train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Saved splits to {output_path}")

    return splits
