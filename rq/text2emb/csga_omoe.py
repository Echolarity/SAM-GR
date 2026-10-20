"""Combined CSGA -> OMoE pipeline.

CSGA first aligns collaborative signals into the semantic domain (by aggregating
multimodal semantic features of collaborative Top-K neighbours), producing a
collaboratively aligned feature.  OMoE then fuses the textual, visual and
collaboratively aligned features with modality-specific/shared experts,
orthogonality and load-balancing regularisation.
"""

from __future__ import annotations

import numpy as np

from csga import csga_align
from omoe import omoe_fuse


def csga_omoe_fuse(
    x_t: np.ndarray, x_i: np.ndarray, x_m: np.ndarray, x_c: np.ndarray, args
) -> np.ndarray:
    aligned = csga_align(x_m, x_c, args)
    print("[CSGA-OMoE] CSGA alignment done; running OMoE fusion...", flush=True)
    fused = omoe_fuse(x_t, x_i, aligned, args)
    print("[CSGA-OMoE] OMoE fusion done.", flush=True)
    return fused
