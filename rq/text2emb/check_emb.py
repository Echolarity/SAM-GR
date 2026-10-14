"""Check a saved item embedding .npy: integrity, semantic spread, distribution."""

import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else None


def main():
    if not PATH:
        print("usage: python rq/text2emb/check_emb.py <emb.npy>")
        sys.exit(1)

    emb = np.load(PATH)
    print(f"shape: {emb.shape}")
    print(f"NaN: {np.isnan(emb).any()}, Inf: {np.isinf(emb).any()}")
    print(f"mean={emb.mean():.4f} std={emb.std():.4f}")

    if emb.shape[0] > 1:
        a = emb[0] / (np.linalg.norm(emb[0]) + 1e-9)
        b = emb[1] / (np.linalg.norm(emb[1]) + 1e-9)
        c = emb[min(100, emb.shape[0] - 1)] / (np.linalg.norm(emb[min(100, emb.shape[0] - 1)]) + 1e-9)
        sim01 = float(a @ b)
        sim02 = float(a @ c)
        print(f"sim(0,1)={sim01:.4f} sim(0,100)={sim02:.4f}")
        print("语义区分度:", "ok" if sim01 > sim02 + 0.1 else "较弱(不影响使用)")


if __name__ == "__main__":
    main()
