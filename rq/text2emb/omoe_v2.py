"""OMoE v2: discriminability-aware orthogonal MoE fusion.

Why v2: the v1 OMoE was almost neutral on top of CSGA because its fused branch
collapsed to a low-rank, near-duplicate space (nn-cos > 0.99 for ~96% of items),
so it could only act as a small correction.  v2 targets that directly:

  * alignment keeps LayerNorm but **drops the L2-normalisation** that crushed rank;
  * **2 shared experts** and longer training (more capacity);
  * adds a **near-duplicate repulsion** loss on fused outputs;
  * adds a **variance-matching** loss (per-dim std vs. the aligned concat);
  * a stronger residual weight lets the fused branch genuinely contribute.

Still keeps: modality-specific + shared experts, orthogonality on expert outputs,
load balancing, mean-centred inputs, residual identity path, and degeneracy guards.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OMoE2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_shared_experts: int = 2,
        gate_hidden: int | None = None,
        ortho_weight: float = 0.05,
        load_weight: float = 0.1,
        dup_weight: float = 0.05,
        var_weight: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_modalities = 3
        self.num_experts = self.num_modalities + num_shared_experts
        self.concat_dim = dim * self.num_modalities
        self.ortho_weight = ortho_weight
        self.load_weight = load_weight
        self.dup_weight = dup_weight
        self.var_weight = var_weight

        def _align() -> nn.Sequential:
            return nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

        self.align_text = _align()
        self.align_image = _align()
        self.align_collab = _align()

        self.expert_text = self._mlp(dim, dim)
        self.expert_image = self._mlp(dim, dim)
        self.expert_collab = self._mlp(dim, dim)
        self.shared_experts = nn.ModuleList(
            [self._mlp(self.concat_dim, dim) for _ in range(num_shared_experts)]
        )

        gh = gate_hidden or max(128, self.concat_dim // 32)
        self.gate = nn.Sequential(
            nn.Linear(self.concat_dim, gh), nn.ReLU(), nn.Linear(gh, self.num_experts)
        )
        self._zero_init_gate()

        self.out_proj = nn.Sequential(nn.Linear(self.concat_dim, dim), nn.LayerNorm(dim))
        dec_hidden = max(256, dim // 4)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dec_hidden), nn.GELU(), nn.Linear(dec_hidden, self.concat_dim)
        )

    @staticmethod
    def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def _zero_init_gate(self):
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _aligned(self, e_t, e_i, e_c):
        # NOTE: no L2-normalisation (v1 collapse contributor)
        return [self.align_text(e_t), self.align_image(e_i), self.align_collab(e_c)]

    def forward(self, e_t, e_i, e_c):
        feats = self._aligned(e_t, e_i, e_c)
        concat = torch.cat(feats, dim=-1)

        experts = [
            self.expert_text(feats[0]),
            self.expert_image(feats[1]),
            self.expert_collab(feats[2]),
        ]
        for shared in self.shared_experts:
            experts.append(shared(concat))

        gates = F.softmax(self.gate(concat), dim=-1)
        moe = torch.zeros_like(experts[0])
        for k in range(self.num_experts):
            moe = moe + gates[:, k].unsqueeze(-1) * experts[k]

        fused = self.out_proj(concat) + moe

        if not self.training:
            return fused, None

        rec = F.smooth_l1_loss(self.decoder(fused), concat.detach())

        ortho = fused.new_zeros(())
        pairs = 0
        for a in range(self.num_experts):
            for b in range(a + 1, self.num_experts):
                ortho = ortho + F.cosine_similarity(experts[a], experts[b], dim=-1).abs().mean()
                pairs += 1
        ortho = ortho / max(1, pairs)

        mean_gate = gates.mean(dim=0)
        load = self.num_experts * (mean_gate**2).sum()

        # near-duplicate repulsion: penalise random pairs with high cosine
        b = fused.shape[0]
        perm = torch.randperm(b, device=fused.device)
        pair_cos = F.cosine_similarity(fused, fused[perm], dim=-1)
        dup = (F.relu(pair_cos - 0.3) ** 2).mean()

        # variance matching: per-dim std of fused vs. the mean aligned feature
        mean_feat = torch.stack(feats, dim=0).mean(dim=0).detach()
        var = F.mse_loss(fused.std(dim=0), mean_feat.std(dim=0))

        total = (
            rec
            + self.ortho_weight * ortho
            + self.load_weight * load
            + self.dup_weight * dup
            + self.var_weight * var
        )

        self._monitor_weights = mean_gate.detach().cpu().numpy()
        self._monitor_loss_rec_real = float(rec.detach().cpu().item())
        self._monitor_loss_ortho_real = float(ortho.detach().cpu().item())
        return fused, total


def omoe_fuse_v2(x_t: np.ndarray, x_i: np.ndarray, x_c: np.ndarray, args) -> np.ndarray:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n, dim = x_t.shape
    epochs = int(getattr(args, "omoe2_epochs", 60))
    lr = float(getattr(args, "omoe2_lr", 1e-3))
    bs = int(getattr(args, "omoe2_batch_size", 512))
    shared = int(getattr(args, "omoe2_shared", 2))

    t = torch.from_numpy(np.asarray(x_t, dtype=np.float32)).to(device)
    i = torch.from_numpy(np.asarray(x_i, dtype=np.float32)).to(device)
    c = torch.from_numpy(np.asarray(x_c, dtype=np.float32)).to(device)
    t = t - t.mean(dim=0, keepdim=True)
    i = i - i.mean(dim=0, keepdim=True)
    c = c - c.mean(dim=0, keepdim=True)

    model = OMoE2(dim, num_shared_experts=shared).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for st in range(0, n, bs):
            b = perm[st : st + bs]
            _, loss = model(t[b], i[b], c[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        total /= n
        if ep % 10 == 0:
            gates = getattr(model, "_monitor_weights", np.zeros(model.num_experts))
            print(
                f"[OMoE-v2] epoch {ep}: total={total:.4f} "
                f"ortho={getattr(model, '_monitor_loss_ortho_real', 0.0):.4f} "
                f"gates={['%.3f' % g for g in gates]}",
                flush=True,
            )

    out = np.empty_like(np.asarray(x_t, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        for st in range(0, n, bs):
            en = min(st + bs, n)
            fused, _ = model(t[st:en], i[st:en], c[st:en])
            out[st:en] = fused.cpu().numpy()

    aligned = np.asarray(x_c, dtype=np.float32)
    target_std = float(getattr(args, "omoe_target_std", 4.0))
    cur = float(out.std())
    if cur > 1e-8:
        out = out * (target_std / cur)
    print(f"[OMoE-v2] fused branch scale: std {cur:.4f} -> {target_std:.4f}", flush=True)

    resid = float(getattr(args, "omoe2_residual", 1.0))
    out = aligned + resid * out
    print(f"[OMoE-v2] CSGA residual weight={resid}; final std={out.std():.4f}", flush=True)

    sample = out[: min(2000, n)]
    sn = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9)
    aff = sn @ sn.T
    np.fill_diagonal(aff, -2.0)
    nn = aff.max(axis=1)
    mean_cos = float(aff[aff > -1.5].mean())
    frac_dup = float(np.mean(nn > 0.99))
    print(
        f"[OMoE-v2] fused sanity: std={sample.std():.4f} mean_cos={mean_cos:.4f} "
        f"nn_dup_frac={frac_dup:.3f}",
        flush=True,
    )
    if mean_cos > 0.99 or sample.std() < 1e-2 or frac_dup > 0.8:
        print("[OMoE-v2][WARN] degenerated; falling back to CSGA feature.", flush=True)
        out = aligned
    return out
