"""Subject Adapter: lightweight per-subject adaptation module.

Shifts the causal graph prior without retraining the backbone.
Frozen backbone + trainable adapter at inference time.
"""
import torch
import torch.nn as nn


class SubjectAdapter(nn.Module):
    """Lightweight residual MLP adapter for subject-level adaptation.

    ~16k parameters by default — small enough to fit per-subject
    without significant memory overhead.

    Input:  (B, N, d_model) transformer embeddings
    Output: (B, N, d_model) adapted embeddings
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        n_layers: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual

        layers = []
        in_dim = d_model
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.GELU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, d_model)
        Returns:
            (B, N, d_model)
        """
        if self.residual:
            return z + self.net(z)
        return self.net(z)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
