"""Baseline models for comparison.

NeurIPS reviewers will expect comparisons against:
1. PatchTST — state-of-the-art time series transformer
2. Static GNN — graph neural net with fixed electrode-distance graph
3. Correlation-based — attention masked by Pearson/coherence connectivity
4. Vanilla Transformer — full (unconstrained) attention (key ablation)
5. Raw Waveform — no spectral tokenizer (tokenizer ablation)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline 1: PatchTST
# ---------------------------------------------------------------------------
class PatchTSTBaseline(nn.Module):
    """Simplified PatchTST: patches raw signal, applies vanilla transformer.

    Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers" (ICLR 2023)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        window_samples: int = 500,
        patch_size: int = 25,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.n_patches = window_samples // patch_size
        self.d_model = d_model

        # Channel-independent patching (CI mode from PatchTST paper)
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        self.channel_embed = nn.Embedding(n_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, T = x.shape
        device = x.device

        # Patch each channel independently: (B, C, n_patches, patch_size)
        x = x.reshape(B, C, self.n_patches, self.patch_size)
        # (B, C, n_patches, d_model)
        x = self.patch_embed(x)

        # Add channel embedding
        ch_ids = torch.arange(C, device=device)
        x = x + self.channel_embed(ch_ids).reshape(1, C, 1, self.d_model)

        # Flatten channels and patches: (B, C * n_patches, d_model)
        x = x.reshape(B, C * self.n_patches, self.d_model)

        # Add positional embedding (repeat for each channel)
        pos = self.pos_embed.repeat(1, C, 1)
        x = x + pos

        x = self.encoder(x)
        x = self.norm(x)

        # Global pool -> classify
        x = x.mean(dim=1)
        logits = self.head(x)

        return {"logits": logits}


# ---------------------------------------------------------------------------
# Baseline 2: Static GNN (electrode distance graph)
# ---------------------------------------------------------------------------
class StaticGNNBaseline(nn.Module):
    """GNN with fixed adjacency based on electrode distances.

    Uses the same spectral tokenizer as our model but replaces learned
    causal graph with a static graph from electrode montage positions.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_bands: int = 5,
        adjacency_type: str = "distance",  # "distance" or "neighbor"
    ):
        super().__init__()
        self.n_tokens = n_channels * n_bands

        # Simple linear tokenizer (no spectral bands for fairness)
        self.proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(self.n_tokens, d_model)

        # Static adjacency (registered as buffer, not learned)
        adj = self._build_static_adjacency(n_channels, n_bands, adjacency_type)
        self.register_buffer("static_adj", adj)

        # Use same transformer architecture as our model
        from src.model.transformer import GraphConditionedTransformer
        self.transformer = GraphConditionedTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def _build_static_adjacency(
        self, n_ch: int, n_bands: int, adj_type: str
    ) -> torch.Tensor:
        """Build a static adjacency matrix based on electrode topology."""
        N = n_ch * n_bands

        if adj_type == "distance":
            # All-to-all within same channel, neighbor channels connected
            adj = torch.zeros(N, N)
            for c in range(n_ch):
                for b1 in range(n_bands):
                    for b2 in range(n_bands):
                        adj[c * n_bands + b1, c * n_bands + b2] = 1.0
                # Connect adjacent channels
                if c + 1 < n_ch:
                    for b in range(n_bands):
                        adj[c * n_bands + b, (c + 1) * n_bands + b] = 1.0
                        adj[(c + 1) * n_bands + b, c * n_bands + b] = 1.0
        else:  # fully connected
            adj = torch.ones(N, N)

        # Remove self-loops
        adj = adj * (1.0 - torch.eye(N))
        return adj.unsqueeze(0)  # (1, N, N)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, T = x.shape
        device = x.device

        # Simple tokenization: split into chunks per channel*band
        n_per = T // (self.n_tokens // C)
        tokens = x.reshape(B, self.n_tokens, -1).mean(dim=-1, keepdim=True)
        tokens = self.proj(tokens)  # (B, N, d_model)

        pos_ids = torch.arange(self.n_tokens, device=device)
        tokens = tokens + self.pos_embed(pos_ids)

        adj = self.static_adj.expand(B, -1, -1).to(device)
        x = self.transformer(tokens, adj)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)

        return {"logits": logits}


# ---------------------------------------------------------------------------
# Baseline 3: Correlation-based connectivity
# ---------------------------------------------------------------------------
class CorrelationGraphBaseline(nn.Module):
    """Uses Pearson correlation between channels as attention mask.

    Computes per-window correlation matrix from raw signals and uses that
    (thresholded) as the attention mask. No learning of the graph — purely
    signal-derived.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_bands: int = 5,
        corr_threshold: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_tokens = n_channels * n_bands
        self.corr_threshold = corr_threshold
        self.d_model = d_model

        self.proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(self.n_tokens, d_model)

        from src.model.transformer import GraphConditionedTransformer
        self.transformer = GraphConditionedTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def _compute_correlation_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Compute thresholded Pearson correlation as adjacency.

        Args:
            x: (B, C, T) raw signal
        Returns:
            adj: (B, N, N) where N = C * n_bands
        """
        B, C, T = x.shape
        device = x.device

        # Channel-level correlation
        x_centered = x - x.mean(dim=-1, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(-1, -2)) / (T - 1)
        std = x.std(dim=-1, keepdim=True)
        std_outer = torch.bmm(std, std.transpose(-1, -2))
        corr = cov / (std_outer + 1e-8)  # (B, C, C)

        # Threshold
        adj_ch = (corr.abs() > self.corr_threshold).float()

        # Expand to token level (each channel has n_bands tokens)
        adj = adj_ch.repeat_interleave(self.n_bands, dim=1)
        adj = adj.repeat_interleave(self.n_bands, dim=2)

        # Remove self-loops
        eye = torch.eye(adj.shape[-1], device=device).unsqueeze(0)
        adj = adj * (1.0 - eye)

        return adj

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, T = x.shape
        device = x.device

        adj = self._compute_correlation_graph(x)

        tokens = x.reshape(B, self.n_tokens, -1).mean(dim=-1, keepdim=True)
        tokens = self.proj(tokens)

        pos_ids = torch.arange(self.n_tokens, device=device)
        tokens = tokens + self.pos_embed(pos_ids)

        out = self.transformer(tokens, adj)
        out = self.norm(out)
        out = out.mean(dim=1)
        logits = self.head(out)

        return {"logits": logits}


# ---------------------------------------------------------------------------
# Baseline 4: Vanilla Transformer (full attention, key ablation)
# ---------------------------------------------------------------------------
class VanillaTransformerBaseline(nn.Module):
    """Standard full-attention transformer. Same tokenizer, no graph masking.

    This is the critical ablation — same architecture minus graph masking.
    The theorem predicts this generalizes worse on distribution shift.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        window_samples: int = 500,
        sample_rate: float = 125.0,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
    ):
        super().__init__()
        from src.model.tokenizer import SpectralTokenizer

        self.tokenizer = SpectralTokenizer(
            n_channels=n_channels, d_token=d_model,
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            sample_rate=sample_rate,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(x)
        out = self.encoder(tokens)
        out = self.norm(out)
        out = out.mean(dim=1)
        logits = self.head(out)
        return {"logits": logits}


# ---------------------------------------------------------------------------
# Baseline 5: Raw Waveform Tokenizer (no spectral decomposition)
# ---------------------------------------------------------------------------
class RawWaveformBaseline(nn.Module):
    """Uses raw waveform patches instead of spectral band tokens.

    Same graph-conditioned transformer, but tokens are raw time-domain
    patches instead of frequency-band features. This is the tokenizer
    ablation — shows whether spectral decomposition matters.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        window_samples: int = 500,
        patch_size: int = 50,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.n_patches = window_samples // patch_size
        self.n_tokens = n_channels * self.n_patches

        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Embedding(self.n_tokens, d_model)

        from src.model.causal_graph import CausalGraphInferencer
        from src.model.transformer import GraphConditionedTransformer

        self.graph_inferencer = CausalGraphInferencer(
            d_model=d_model, n_layers=2, n_heads=4, dropout=dropout,
        )
        self.transformer = GraphConditionedTransformer(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, T = x.shape
        device = x.device

        # Raw waveform patching
        x = x.reshape(B, C, self.n_patches, self.patch_size)
        x = x.reshape(B, self.n_tokens, self.patch_size)
        tokens = self.patch_proj(x)

        pos_ids = torch.arange(self.n_tokens, device=device)
        tokens = tokens + self.pos_embed(pos_ids)

        adj, edge_probs = self.graph_inferencer(tokens)
        out = self.transformer(tokens, adj)
        out = self.norm(out)
        out = out.mean(dim=1)
        logits = self.head(out)

        return {"logits": logits, "edge_probs": edge_probs}
