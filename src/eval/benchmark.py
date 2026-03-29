"""Evaluation: F1, AUROC, ECE metrics on a dataset."""
import logging

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> dict[str, float]:
    """Run evaluation and compute metrics.

    Returns:
        Dictionary with f1_macro, auroc, ece, loss.
    """
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        output = model(signal)
        logits = output["logits"]

        loss = torch.nn.functional.cross_entropy(logits, label)
        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

    if n_batches == 0:
        return {"f1_macro": 0.0, "auroc": 0.0, "ece": 1.0, "loss": float("inf")}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()

    # F1 macro
    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

    # AUROC (one-vs-rest)
    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            auroc = roc_auc_score(labels_np, probs[:, 1])
        else:
            auroc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0

    # ECE
    ece = expected_calibration_error(probs, labels_np)

    metrics = {
        "f1_macro": float(f1),
        "auroc": float(auroc),
        "ece": float(ece),
        "loss": total_loss / n_batches,
    }

    return metrics
