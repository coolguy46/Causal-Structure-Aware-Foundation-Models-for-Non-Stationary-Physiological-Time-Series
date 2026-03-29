"""Task loss and joint objective combining all loss terms."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.1,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy classification loss with label smoothing and optional class weighting."""
    return F.cross_entropy(
        logits, labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def joint_loss(
    recon_loss: torch.Tensor,
    causal_loss: torch.Tensor,
    task_loss: torch.Tensor,
    lambda_recon: float = 1.0,
    lambda_causal: float = 0.1,
    lambda_task: float = 1.0,
    sparsity_loss: torch.Tensor | None = None,
    dag_loss: torch.Tensor | None = None,
    lambda_sparsity: float = 0.01,
    lambda_dag: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint objective combining reconstruction, causal, and task losses.

    L = lambda_recon * L_recon + lambda_causal * L_causal + lambda_task * L_task
        + lambda_sparsity * L_sparsity + lambda_dag * L_dag

    Returns:
        total_loss: scalar
        loss_dict: breakdown for logging
    """
    total = (
        lambda_recon * recon_loss
        + lambda_causal * causal_loss
        + lambda_task * task_loss
    )

    loss_dict = {
        "recon": recon_loss.item(),
        "causal": causal_loss.item(),
        "task": task_loss.item(),
    }

    if sparsity_loss is not None:
        total = total + lambda_sparsity * sparsity_loss
        loss_dict["sparsity"] = sparsity_loss.item()

    if dag_loss is not None:
        total = total + lambda_dag * dag_loss
        loss_dict["dag"] = dag_loss.item()

    loss_dict["total"] = total.item()
    return total, loss_dict
