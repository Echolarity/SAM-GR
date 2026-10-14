"""Plot MoE / contrastive training curves (matplotlib)."""

from __future__ import annotations

import os

import numpy as np


def plot_moe_training_curves(history, is_collab: bool, mode: str, save_dir: str = ".") -> str:
    """Plot MoE training loss + per-expert gate means.

    ``history`` is a list of dicts with keys: ``total``, ``rec``, ``ortho``,
    ``gates`` (list/array of per-expert mean gate), optional ``ortho_weight``.
    Returns the saved figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return ""

    epochs = list(range(1, len(history) + 1))
    total = [h.get("total", 0.0) for h in history]
    rec = [h.get("rec", 0.0) for h in history]
    ortho = [h.get("ortho", 0.0) for h in history]
    gates = np.array([h.get("gates", []) for h in history])  # [E, num_experts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, total, label="Total", lw=2)
    ax.plot(epochs, rec, label="Rec", lw=1.5)
    ax.plot(epochs, ortho, label="Ortho", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"MoE training loss ({mode}, collab={is_collab})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if gates.size:
        num_experts = gates.shape[1]
        for k in range(num_experts):
            ax.plot(epochs, gates[:, k], label=f"expert {k}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean gate")
        ax.set_title("Expert gate means")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"moe_training_curves_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
