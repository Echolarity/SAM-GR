"""CSGA: Collaborative Similarity Graph Aggregation.

Design (aligned with the paper description):
  * collaborative embeddings (SASRec) define an *implicit* item-item graph via
    Top-K cosine neighbours;
  * for every item, the multimodal (semantic) features of its neighbours are
    aggregated with *learned* attention weights;
  * weights are produced only from collaborative-space signals
    ``[s_i, s_j, cos_ij]`` -> no explicit cross-space mapping is learned;
  * the neighbour scorer is trained so that the aggregated neighbour feature can
    reconstruct the item's own semantic feature (graph-guided semantic denoising).

Final item representation: ``h_i = 0.5 * (m_i + a_i)`` where ``a_i`` is the
learned-weight aggregation of neighbours' multimodal features.  A degeneracy
guard falls back to ``m_i`` if the fused space collapses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_neighbors(x_c: np.ndarray, k: int, device: str, chunk: int = 2048):
    """Return L2-normalised collab matrix, Top-K neighbour ids and cosines."""
    x = torch.from_numpy(np.asarray(x_c, dtype=np.float32)).to(device)
    x = F.normalize(x, dim=-1)
    n = x.shape[0]
    k = max(1, min(int(k), n - 1))
    idx = torch.empty(n, k, dtype=torch.long, device=device)
    cos = torch.empty(n, k, dtype=torch.float32, device=device)
    with torch.no_grad():
        for st in range(0, n, chunk):
            en = min(st + chunk, n)
            sim = x[st:en] @ x.t()
            rows = torch.arange(st, en, device=device)
            sim[torch.arange(en - st, device=device), rows] = -2.0
            v, ix = sim.topk(k, dim=1)
            idx[st:en] = ix
            cos[st:en] = v
    return x, idx, cos


class CSGA(nn.Module):
    """Collaborative-guided neighbour scorer + semantic reconstructor."""

    def __init__(self, collab_dim: int, sem_dim: int, hidden: int = 256, tau: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.scorer = nn.Sequential(
            nn.Linear(collab_dim * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        dec_hidden = max(256, sem_dim // 4)
        self.decoder = nn.Sequential(
            nn.Linear(sem_dim, dec_hidden),
            nn.GELU(),
            nn.Linear(dec_hidden, sem_dim),
        )

    def weights(self, s_i, s_nb, cos_nb):
        b, k, c = s_nb.shape
        feat = torch.cat(
            [s_i.unsqueeze(1).expand(b, k, c), s_nb, cos_nb.unsqueeze(-1)], dim=-1
        )
        logits = self.scorer(feat).squeeze(-1) / self.tau
        return torch.softmax(logits, dim=-1)


def csga_align(x_m: np.ndarray, x_c: np.ndarray, args) -> np.ndarray:
    """Fuse multimodal semantic features ``x_m`` guided by collaborative ``x_c``."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n, dim = x_m.shape
    collab_dim = x_c.shape[1]
    k = int(getattr(args, "csga_topk", 10))
    epochs = int(getattr(args, "csga_epochs", 60))
    lr = float(getattr(args, "csga_lr", 1e-3))
    bs = int(getattr(args, "csga_batch_size", 512))

    collab, idx, cos = _topk_neighbors(x_c, k, device=device)
    sem = torch.from_numpy(np.asarray(x_m, dtype=np.float32)).to(device)
    # remove the dominant shared component before neighbour averaging; otherwise the
    # averaged neighbour feature is dominated by the common direction and h becomes
    # over-correlated across items (same failure mode seen in the MoE v1 collapse).
    sem = sem - sem.mean(dim=0, keepdim=True)

    model = CSGA(collab_dim, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for st in range(0, n, bs):
            b = perm[st : st + bs]
            m_i = sem[b]
            s_i = collab[b]
            nb = idx[b]
            s_nb = collab[nb]
            c_nb = cos[b]
            w = model.weights(s_i, s_nb, c_nb)
            agg = (w.unsqueeze(-1) * sem[nb]).sum(dim=1)
            rec = model.decoder(agg)
            loss = F.smooth_l1_loss(rec, m_i.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        total /= n
        if ep % 5 == 0:
            print(f"[CSGA] epoch {ep}: recon={total:.4f}", flush=True)

    out = np.empty_like(np.asarray(x_m, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        for st in range(0, n, bs):
            en = min(st + bs, n)
            b = torch.arange(st, en, device=device)
            m_i = sem[b]
            s_i = collab[b]
            nb = idx[b]
            s_nb = collab[nb]
            c_nb = cos[b]
            w = model.weights(s_i, s_nb, c_nb)
            agg = (w.unsqueeze(-1) * sem[nb]).sum(dim=1)
            # conservative fusion: keep the item's own semantic signal dominant and
            # add the collaborative aggregation as a bounded correction.
            h = m_i + 0.5 * agg
            out[st:en] = h.cpu().numpy()

    # degeneracy guard
    sample = out[: min(2000, n)]
    sn = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9)
    off = sn @ sn.T
    off = off[~np.eye(len(sn), dtype=bool)]
    mean_cos = float(off.mean())
    print(f"[CSGA] fused sanity: std={sample.std():.4f} mean_inter_item_cos={mean_cos:.4f}", flush=True)
    if mean_cos > 0.99 or sample.std() < 1e-2:
        print(
            f"[CSGA][WARN] fused space degenerated (mean_cos={mean_cos:.4f}); "
            f"falling back to multimodal features.",
            flush=True,
        )
        out = np.asarray(x_m, dtype=np.float32)
    return out


# backward-compatible alias for the standalone csga_collab mode
csga_fuse = csga_align
