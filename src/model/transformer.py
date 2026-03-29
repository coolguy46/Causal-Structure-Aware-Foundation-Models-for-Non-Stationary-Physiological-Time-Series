"""Graph-Conditioned Transformer.

Attention is masked by the inferred causal graph — attention logits are
zeroed where the adjacency matrix indicates no causal edge.
Uses F.scaled_dot_product_attention for hardware-accelerated kernels.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GraphAttentionLayer(nn.Module):
    """Single graph-conditioned attention + FFN layer.

    Uses fused QKV projection and F.scaled_dot_product_attention for
    dispatch to memory-efficient / Flash Attention CUDA kernels.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Fused QKV — single matmul instead of three separate projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, d_model) token embeddings
            adj: (B, N, N) adjacency matrix (1 = attend, 0 = mask)
        Returns:
            out: (B, N, d_model)
        """
        B, N, _ = x.shape

        # Self-connections must always be allowed (adj has no self-loops).
        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        adj_with_self = torch.clamp(adj + eye, max=1.0)
        # Additive mask: (B, 1, N, N) — broadcasts across heads, no expand needed
        attn_mask = (1.0 - adj_with_self).unsqueeze(1) * -1e9

        # Pre-norm + fused QKV projection
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # SDPA dispatches to memory-efficient or math kernel (Flash when mask=None)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        # Merge heads back: (B, n_heads, N, head_dim) -> (B, N, d_model)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        return x


class GraphConditionedTransformer(nn.Module):
    """Stack of graph-conditioned attention layers.

    The key theoretical contribution: attention is constrained by
    inferred causality, preventing overfitting to spurious correlations.

    Input:  tokens (B, N, d_model), adj (B, N, N)
    Output: embeddings (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, d_model) spectral tokens
            adj:    (B, N, N) causal adjacency matrix
        Returns:
            out:    (B, N, d_model)
        """
        x = tokens
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, adj, use_reentrant=False)
            else:
                x = layer(x, adj)
        return self.final_norm(x)


class ClassificationHead(nn.Module):
    """Classification head on top of transformer embeddings."""

    def __init__(
        self,
        d_model: int = 128,
        n_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) transformer output
        Returns:
            logits: (B, n_classes)
        """
        # Global average pooling over tokens
        x_pool = x.mean(dim=1)  # (B, d_model)
        return self.head(x_pool)
