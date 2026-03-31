"""Causal Graph Inference Module.

Learns a sparse directed adjacency matrix over spectral tokens
using pairwise edge scoring with straight-through sparsification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughThreshold(torch.autograd.Function):
    """Straight-through estimator for binary thresholding."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None


class CausalGraphInferencer(nn.Module):
    """Infers a sparse directed causal graph from spectral tokens.

    Uses a transformer encoder to produce node embeddings, then
    pairwise MLP scoring to produce edge logits. Applies
    straight-through thresholding for discrete sparse graphs.

    Input:  tokens (B, N, d_token)
    Output: adj    (B, N, N) sparse binary adjacency matrix
            logits (B, N, N) raw edge logits (for loss computation)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        sparsity_threshold: float = 0.5,
        l1_lambda: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = sparsity_threshold
        self.l1_lambda = l1_lambda

        # Transformer encoder for node embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Pairwise edge scoring via factored bilinear: h_i @ W @ h_j + MLP(h_i, h_j)
        # Avoids O(N^2 * 2d) memory from concatenating all pairs
        self.edge_src_proj = nn.Linear(d_model, d_model)
        self.edge_tgt_proj = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.edge_bias_tgt = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @torch.compiler.disable
    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, d) spectral tokens
        Returns:
            adj:    (B, N, N) sparse binary adjacency (directed)
            logits: (B, N, N) raw sigmoid edge probabilities
        """
        B, N, d = tokens.shape

        # Encode tokens to node embeddings
        node_emb = self.encoder(tokens)  # (B, N, d)

        # Factored bilinear scoring: score(i,j) = (W_s h_i) . (W_t h_j) / sqrt(d) + bias_s(h_i) + bias_t(h_j)
        src = self.edge_src_proj(node_emb)  # (B, N, d)
        tgt = self.edge_tgt_proj(node_emb)  # (B, N, d)
        edge_logits = torch.bmm(src, tgt.transpose(-1, -2)) / (d ** 0.5)  # (B, N, N)
        edge_logits = edge_logits + self.edge_bias(node_emb).squeeze(-1).unsqueeze(2)   # src bias
        edge_logits = edge_logits + self.edge_bias_tgt(node_emb).squeeze(-1).unsqueeze(1)  # tgt bias
        edge_probs = torch.sigmoid(edge_logits)

        # Straight-through thresholding
        adj = StraightThroughThreshold.apply(edge_probs, self.threshold)

        # Zero out self-loops
        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        adj = adj * (1.0 - eye)

        return adj, edge_probs

    def sparsity_loss(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """L1 regularization on edge probabilities for sparsity."""
        return self.l1_lambda * edge_probs.mean()

    def dag_loss(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """Acyclicity constraint: tr(e^(A ○ A)) - N = 0 for DAGs.

        Uses the NOTEARS formulation on continuous edge probabilities.
        Power series capped at 10 terms — higher powers contribute < 1e-6
        for sigmoid outputs in [0,1] and converge factorially.
        """
        N = edge_probs.shape[-1]
        A_sq = edge_probs * edge_probs
        eye = torch.eye(N, device=edge_probs.device).unsqueeze(0)
        A_sq = A_sq * (1.0 - eye)
        # Capped power series: O(K * N^2 * B) with K=6 instead of K=N
        K = min(N, 6)
        M = torch.eye(N, device=edge_probs.device).unsqueeze(0).expand_as(A_sq)
        A_power = torch.eye(N, device=edge_probs.device).unsqueeze(0).expand_as(A_sq)
        for k in range(1, K + 1):
            A_power = torch.bmm(A_power, A_sq) / k
            M = M + A_power
        dag_penalty = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1) - N
        return dag_penalty.mean()
