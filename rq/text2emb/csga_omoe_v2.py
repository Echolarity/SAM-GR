"""Combined CSGA -> OMoE-v2 pipeline (discriminability-aware)."""

from __future__ import annotations

import numpy as np

from csga import csga_align
from omoe_v2 import omoe_fuse_v2


def csga_omoe_fuse_v2(
    x_t: np.ndarray, x_i: np.ndarray, x_m: np.ndarray, x_c: np.ndarray, args
) -> np.ndarray:
    aligned = csga_align(x_m, x_c, args)
    print("[CSGA-OMoE-v2] CSGA alignment done; running OMoE-v2 fusion...", flush=True)
    fused = omoe_fuse_v2(x_t, x_i, aligned, args)
    print("[CSGA-OMoE-v2] OMoE-v2 fusion done.", flush=True)
    return fused
