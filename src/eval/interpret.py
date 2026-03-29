"""Graph visualization and attention interpretability."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)


def visualize_causal_graph(
    adj: np.ndarray,
    channel_names: list[str] | None = None,
    title: str = "Inferred Causal Graph",
    save_path: str | None = None,
    threshold: float = 0.3,
):
    """Visualize an adjacency matrix as a directed graph.

    Args:
        adj: (N, N) adjacency matrix
        channel_names: Optional node labels
        title: Plot title
        save_path: If provided, save figure
        threshold: Edge weight threshold for display
    """
    N = adj.shape[0]
    if channel_names is None:
        channel_names = [f"N{i}" for i in range(N)]

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, label=channel_names[i])

    for i in range(N):
        for j in range(N):
            if i != j and adj[i, j] > threshold:
                G.add_edge(i, j, weight=adj[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Graph layout
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)
    weights = [e[2]["weight"] for e in edges]

    nx.draw_networkx(
        G, pos, ax=axes[0],
        labels={i: channel_names[i] for i in range(N)},
        node_color="lightblue",
        node_size=500,
        edge_color=weights,
        edge_cmap=plt.cm.Reds,
        width=2,
        arrows=True,
        arrowsize=15,
        font_size=8,
    )
    axes[0].set_title(title)

    # Adjacency heatmap
    im = axes[1].imshow(adj, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_xticks(range(N))
    axes[1].set_yticks(range(N))
    if N <= 20:
        axes[1].set_xticklabels(channel_names, rotation=45, ha="right", fontsize=7)
        axes[1].set_yticklabels(channel_names, fontsize=7)
    axes[1].set_title("Adjacency Matrix")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved graph visualization to {save_path}")
    plt.close()


@torch.no_grad()
def extract_graphs_for_subject(
    model,
    loader,
    device: torch.device,
    n_windows: int = 10,
) -> list[np.ndarray]:
    """Extract inferred causal graphs for multiple windows of a subject."""
    model.eval()
    graphs = []

    for i, batch in enumerate(loader):
        if i >= n_windows:
            break
        signal = batch["signal"].to(device)
        output = model(signal, return_graph=True)
        adj = output["adj"].cpu().numpy()
        graphs.append(adj[0])  # first sample in batch

    return graphs


def plot_graph_stability(
    graphs: list[np.ndarray],
    save_path: str | None = None,
):
    """Plot temporal stability of inferred graphs across windows."""
    n = len(graphs)
    if n < 2:
        return

    # Pairwise Frobenius distances
    distances = []
    for i in range(n - 1):
        diff = np.linalg.norm(graphs[i] - graphs[i + 1], "fro")
        distances.append(diff)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(distances)), distances, "o-")
    ax.set_xlabel("Window Pair")
    ax.set_ylabel("Frobenius Distance")
    ax.set_title("Graph Stability Across Windows")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
