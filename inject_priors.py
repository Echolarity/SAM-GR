"""Inject RQ-VAE codebook priors into the LLM embedding of new SID tokens.

Optional experiment: gives ``<a_*> / <b_*> / <c_*> / <d_*>`` tokens a structured
initial embedding derived from the trained RQ-VAE codebook, instead of random init.

Usage:
  python inject_priors.py --model_id Qwen/Qwen2.5-0.5B-Instruct \
      --new_tokens_path data/<CATEGORY>/new_tokens.json \
      --rqvae_ckpt rq_output/<CATEGORY>/best_collision_model.pth \
      --output_path models/Qwen2.5-0.5B-Inject-<CATEGORY>
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import torch

TOKEN_RE = re.compile(r"^<([a-z])_(\d+)>$")
LEVEL_OF = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--new_tokens_path", type=str, required=True)
    p.add_argument("--rqvae_ckpt", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--blend_ratio", type=float, default=0.5)
    p.add_argument("--noise_scale", type=float, default=0.01)
    p.add_argument("--download_dir", type=str, default="./models/hub")
    return p.parse_args()


def resolve_model_dir(model_id: str, download_dir: str) -> str:
    """Return a local dir for the model (use local path or modelscope download)."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from modelscope import snapshot_download

        ms_id = model_id
        if ms_id.startswith("Qwen/"):
            ms_id = "qwen/" + ms_id[len("Qwen/") :]
        return snapshot_download(ms_id, cache_dir=download_dir)
    except Exception as e:
        raise RuntimeError(f"无法下载模型 {model_id}: {e}. 请提供本地路径 --model_id.") from e


def extract_codebooks(ckpt_path: str) -> dict[int, np.ndarray]:
    """Return {level: codebook[n_e, e_dim]} from RQ-VAE checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = cktpt_state_dict(ckpt)
    codebooks = {}
    for key, val in state_dict.items():
        m = re.match(r"rq\.vq_layers\.(\d+)\.embedding\.weight", key)
        if m:
            codebooks[int(m.group(1))] = val.detach().cpu().numpy()
    if not codebooks:
        raise ValueError(f"未在 checkpoint 中找到 vq_layers.*.embedding.weight: {list(state_dict.keys())[:20]}")
    return codebooks


def cktpt_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def tile_to_dim(code: np.ndarray, dim: int) -> np.ndarray:
    code = code.astype(np.float32).reshape(-1)
    repeats = int(np.ceil(dim / len(code)))
    tiled = np.tile(code, repeats)[:dim].astype(np.float32)
    return tiled


def standardize(v: np.ndarray, mean: float, std: float) -> np.ndarray:
    v_mean, v_std = float(v.mean()), float(v.std())
    if v_std < 1e-8:
        v_std = 1e-8
    return (v - v_mean) / v_std * std + mean


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model_id, args.download_dir)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    hidden = config.hidden_size
    print(f"hidden_size = {hidden}")

    with open(args.new_tokens_path, encoding="utf-8") as f:
        new_tokens = __import__("json").load(f)

    codebooks = extract_codebooks(args.rqvae_ckpt)
    print(f"codebook levels: {sorted(codebooks.keys())}, shapes: {[codebooks[k].shape for k in sorted(codebooks)]}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    emb = model.get_input_embeddings().weight  # [vocab, hidden]

    llm_mean = float(emb.data.mean().item())
    llm_std = float(emb.data.std().item())

    rng = np.random.default_rng(0)
    injected = 0
    for token in new_tokens:
        m = TOKEN_RE.match(token)
        if not m:
            continue
        level = LEVEL_OF[m.group(1)]
        code_idx = int(m.group(2))
        if level not in codebooks or code_idx >= codebooks[level].shape[0]:
            print(f"[skip] {token}: level {level} code {code_idx} 越界")
            continue

        code = codebooks[level][code_idx]
        prior = tile_to_dim(code, hidden)
        prior = standardize(prior, llm_mean, llm_std)
        prior = prior + rng.normal(0.0, llm_std * args.noise_scale, size=prior.shape).astype(np.float32)

        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id is None or tok_id < 0 or tok_id == tokenizer.unk_token_id:
            print(f"[skip] {token}: 无法映射 token id")
            continue

        random_init = emb.data[tok_id].detach().cpu().numpy().astype(np.float32)
        final = args.blend_ratio * prior + (1.0 - args.blend_ratio) * random_init
        final = standardize(final, llm_mean, llm_std)  # 恢复方差

        assert hidden == emb.shape[1], f"hidden mismatch: {hidden} != {emb.shape[1]}"
        emb.data[tok_id] = torch.from_numpy(final).to(emb.data.dtype)
        injected += 1

    print(f"注入完成: {injected}/{len(new_tokens)} 个 token")

    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"保存到 {args.output_path}")


if __name__ == "__main__":
    main()
