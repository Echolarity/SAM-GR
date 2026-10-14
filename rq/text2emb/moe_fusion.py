"""MoE decoupler and contrastive adapter for multimodal item embeddings.

These modules run on the *main* process after `gather_object` inside
`multimodal_amazon_text2emb.py`.  They fuse the text / image / (optional) collab
embeddings produced by the VLM into a single per-item vector whose dimensionality
equals the single-modality dim (i.e. `input_dim`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEDecoupler(nn.Module):
    """Mixture-of-Experts decoupled fusion.

    Every modality is first aligned to a common `input_dim` space via
    ``Linear -> LayerNorm``.  Then:
      * one *unique* expert per modality (MLP on that modality alone),
      * `num_shared_experts` *shared* experts (MLP on the concatenation),
      * a gating network over the concatenation (last layer zero-initialised).

    Returns ``(fused, total_loss)`` in training and ``(fused, None)`` in eval.
    """

    def __init__(self, input_dim: int, num_shared_experts: int = 1, collab_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_shared_experts = num_shared_experts
        self.collab_dim = collab_dim
        self.num_modalities = 3 if collab_dim is not None else 2
        self.num_experts = self.num_modalities + num_shared_experts
        self.concat_dim = input_dim * self.num_modalities

        # per-modality alignment: Linear -> LayerNorm
        self.align_text = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        self.align_image = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        if collab_dim is not None:
            self.align_collab = nn.Sequential(nn.Linear(collab_dim, input_dim), nn.LayerNorm(input_dim))

        # unique experts (one per modality)
        self.expert_text = self._mlp(input_dim, input_dim)
        self.expert_image = self._mlp(input_dim, input_dim)
        if collab_dim is not None:
            self.expert_collab = self._mlp(input_dim, input_dim)

        # shared experts (input = full concat)
        self.shared_experts = nn.ModuleList(
            [self._mlp(self.concat_dim, input_dim) for _ in range(num_shared_experts)]
        )

        # gating network (last layer zero-initialised)
        self.gate = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.concat_dim // 2, self.num_experts),
        )
        self.temperature = 1.0
        self._zero_init_gate()

    @staticmethod
    def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def _zero_init_gate(self):
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _aligned_features(self, e_t, e_i, e_c=None):
        feats = [self.align_text(e_t), self.align_image(e_i)]
        if e_c is not None:
            feats.append(self.align_collab(e_c))
        return feats

    def forward(self, e_t, e_i, e_c=None):
        feats = self._aligned_features(e_t, e_i, e_c)
        concat = torch.cat(feats, dim=-1)

        expert_outputs = [self.expert_text(feats[0]), self.expert_image(feats[1])]
        if e_c is not None:
            expert_outputs.append(self.expert_collab(feats[2]))
        for shared in self.shared_experts:
            expert_outputs.append(shared(concat))

        logits = self.gate(concat) / self.temperature
        gates = F.softmax(logits, dim=-1)  # [B, num_experts]

        fused = torch.zeros_like(expert_outputs[0])
        for k in range(self.num_experts):
            fused = fused + gates[:, k].unsqueeze(-1) * expert_outputs[k]

        if not self.training:
            return fused, None

        # reconstruction: SmoothL1 against the mean aligned feature
        mean_feat = torch.stack(feats, dim=0).mean(dim=0).detach()
        loss_rec = F.smooth_l1_loss(fused, mean_feat)

        # orthogonal penalty on all modality pairs
        pairs = [(0, 1)]
        if e_c is not None:
            pairs += [(0, 2), (1, 2)]
        cos_sum = 0.0
        for i, j in pairs:
            cos_sum = cos_sum + F.cosine_similarity(feats[i], feats[j], dim=-1).abs().mean()
        ortho_weight = 0.2
        loss_ortho = ortho_weight * (cos_sum / len(pairs))

        # load-balance: penalise gate collapse onto a single expert
        mean_gate = gates.mean(dim=0)  # [num_experts]
        load_bal_weight = 0.5
        loss_load = load_bal_weight * (self.num_experts * (mean_gate**2).sum())

        total_loss = loss_rec + loss_ortho + loss_load

        # monitor attributes
        self._monitor_weights = mean_gate.detach().cpu().numpy()
        self._monitor_loss_rec_real = float(loss_rec.detach().cpu().item())
        self._monitor_loss_ortho_real = float(loss_ortho.detach().cpu().item())
        self._monitor_ortho_weight = ortho_weight

        return fused, total_loss


class ContrastiveAdapter(nn.Module):
    """Dual projection head + learnable temperature, trained with symmetric InfoNCE."""

    def __init__(self, input_dim: int, proj_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim or input_dim
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, self.proj_dim), nn.GELU(), nn.Linear(self.proj_dim, self.proj_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, self.proj_dim), nn.GELU(), nn.Linear(self.proj_dim, self.proj_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _proj(self, x_t, x_i):
        z_t = F.normalize(self.text_proj(x_t), dim=-1)
        z_i = F.normalize(self.image_proj(x_i), dim=-1)
        return z_t, z_i

    def forward(self, x_t, x_i):
        z_t, z_i = self._proj(x_t, x_i)
        temp = self.temperature.clamp(min=1e-4)
        logits = z_t @ z_i.t() / temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss

    @torch.no_grad()
    def get_fused_features(self, x_t, x_i):
        return self.text_proj(x_t) + self.image_proj(x_i)


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T
