"""Multimodal item text/image embedding extractor for MIMIGenRec.

Replaces the text-only ``amazon_text2emb.py`` stage.  Uses Qwen3-VL-8B via
``AutoModelForImageTextToText`` and supports 11 feature modes:

  multimodal, text_only, image_only, concat, image2text, image2text_supp,
  moe_decouple, contrastive_adapter, multimodal_collab, concat_collab, moe_collab

Sharded across processes with ``accelerate``; MoE / contrastive fusion runs on
the main process after ``gather_object``.

Output (sorted by item_id):
  ``{root}/{dataset}.emb-{plm_name}-{mode}.npy``
and, for image2text* modes:
  ``{root}/{dataset}.item-{mode}.json``
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from moe_fusion import ContrastiveAdapter, MoEDecoupler
from moe_plotter import plot_moe_training_curves

MODES = {
    "multimodal",
    "text_only",
    "image_only",
    "concat",
    "image2text",
    "image2text_supp",
    "moe_decouple",
    "contrastive_adapter",
    "multimodal_collab",
    "concat_collab",
    "moe_collab",
}

IMAGE2TEXT_PROMPT = (
    "Describe the product shown in the image in detail, including its appearance, "
    "color, material, and purpose."
)
IMAGE2TEXT_SUPP_PROMPT = (
    "Describe only the additional visual details of the product that are not already "
    "obvious from its text metadata."
)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def clean_text(text):
    if not text:
        return ""
    return str(text).replace("\n", " ").strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def release_model(model):
    """Free the VLM before gather (order matters: free -> wait -> gather)."""
    if model is None:
        return
    try:
        model.cpu()
    except Exception:
        pass
    try:
        for p in model.parameters():
            p.data = torch.empty(0)
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #


def load_item_data(args):
    path = os.path.join(args.root, f"{args.dataset}.item.json")
    item2feature = load_json(path)
    item_data_list = []
    missing_img = 0
    for item_idx, data in item2feature.items():
        text_parts = []
        for key in ("title", "description"):
            v = clean_text(data.get(key, ""))
            if v:
                text_parts.append(v)
        final_text = " ".join(text_parts) if text_parts else "unknown item"
        try:
            item_id = int(item_idx)
        except (ValueError, TypeError):
            item_id = item_idx

        image_paths = data.get("image_paths", [])
        valid_img = None
        if image_paths and os.path.exists(image_paths[0]):
            valid_img = image_paths[0]
        else:
            missing_img += 1

        item_data_list.append((item_id, final_text, valid_img, data))
    return item_data_list, missing_img


# --------------------------------------------------------------------------- #
# 消息构造 / 前向
# --------------------------------------------------------------------------- #


def _text_message(text):
    return [{"role": "user", "content": [{"type": "text", "text": text[:1500]}]}]


def _image_message():
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the product shown in the image."}]}]


def _multimodal_message(text):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text[:1500]}]}]


def _forward_pool(processor, model, messages, images):
    prompts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
    inputs = processor(
        text=prompts,
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.to(torch.float32).cpu().numpy()


def _load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((384, 384), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 分片编码
# --------------------------------------------------------------------------- #


def compute_embedding(args, item_data_list, processor, model, accelerator, kind):
    """kind in {text, image, multimodal}."""
    all_ids = [x[0] for x in item_data_list]
    all_texts = [x[1] for x in item_data_list]
    all_imgs = [x[2] for x in item_data_list]

    num_proc = accelerator.num_processes
    rank = accelerator.process_index
    chunk = int(np.ceil(len(all_ids) / num_proc))
    start, end = rank * chunk, min((rank + 1) * chunk, len(all_ids))

    local_ids = all_ids[start:end]
    local_texts = all_texts[start:end]
    local_imgs = all_imgs[start:end]

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = args.batch_size
    results = []
    pbar = tqdm(total=len(local_ids), desc=f"[{kind}] proc {rank}", disable=not accelerator.is_local_main_process)

    for i in range(0, len(local_ids), batch_size):
        ids = local_ids[i : i + batch_size]
        texts = local_texts[i : i + batch_size]
        imgs = local_imgs[i : i + batch_size]

        # group by image presence
        img_ids, img_msgs, img_pils = [], [], []
        txt_ids, txt_msgs = [], []

        for item_id, text, img_path in zip(ids, texts, imgs):
            if img_path and kind != "text":
                pil = _load_image(img_path)
                if pil is not None:
                    img_ids.append(item_id)
                    img_pils.append(pil)
                    img_msgs.append(_multimodal_message(text) if kind == "multimodal" else _image_message())
                    continue
            txt_ids.append(item_id)
            txt_msgs.append(_text_message(text))

        if img_msgs:
            embs = _forward_pool(processor, model, img_msgs, img_pils)
            for item_id, emb in zip(img_ids, embs):
                results.append((item_id, emb))

        if txt_msgs:
            embs = _forward_pool(processor, model, txt_msgs, None)
            for item_id, emb in zip(txt_ids, embs):
                results.append((item_id, emb))

        pbar.update(len(ids))

    pbar.close()
    return results


def compute_image2text_embedding(args, item_data_list, processor, model, accelerator, kind):
    """Generate visual text via VLM, combine with original text, then re-embed."""
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_proc = accelerator.num_processes
    rank = accelerator.process_index
    chunk = int(np.ceil(len(item_data_list) / num_proc))
    start, end = rank * chunk, min((rank + 1) * chunk, len(item_data_list))
    local = item_data_list[start:end]

    generated = {}  # item_id -> combined text
    for item_id, text, img_path, _ in tqdm(local, desc=f"[{kind}] gen proc {rank}", disable=not accelerator.is_local_main_process):
        if img_path:
            pil = _load_image(img_path)
            gen_text = ""
            if pil is not None:
                prompt = IMAGE2TEXT_PROMPT if kind == "image2text" else IMAGE2TEXT_SUPP_PROMPT
                gen_text = _generate(processor, model, pil, prompt, args.max_gen_tokens).strip()
            if kind == "image2text":
                combined = f"[Original Meta]: {text} [Image Visual]: {gen_text}" if (args.combine_orig_text and gen_text) else (gen_text or text)
            else:  # image2text_supp: 原文在前, 死保原文
                combined = f"{text} [Visual Supplement]: {gen_text}" if gen_text else text
        else:
            combined = text  # 无图 item 用原文
        generated[item_id] = combined

    # re-embed combined text (text-only pass)
    ids = list(generated.keys())
    texts = [generated[i] for i in ids]
    results = []
    batch_size = args.batch_size
    for i in range(0, len(ids), batch_size):
        b_ids = ids[i : i + batch_size]
        b_texts = texts[i : i + batch_size]
        embs = _forward_pool(processor, model, [_text_message(t) for t in b_texts], None)
        for item_id, emb in zip(b_ids, embs):
            results.append((item_id, emb))

    return results, generated


def _generate(processor, model, pil, prompt, max_new_tokens=150):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs.input_ids.shape[1]
    return processor.tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# 融合 (gather 之后, 主进程)
# --------------------------------------------------------------------------- #


def _to_matrix(results, n_items):
    """Convert [(id, emb)] -> [n_items, dim] sorted by id."""
    results = sorted(results, key=lambda x: x[0])
    dim = results[0][1].shape[-1]
    mat = np.zeros((n_items, dim), dtype=np.float32)
    for item_id, emb in results:
        mat[int(item_id)] = emb
    return mat


def train_moe(x_t, x_i, x_c, args, collab_dim, mode):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MoEDecoupler(x_t.shape[1], num_shared_experts=1, collab_dim=collab_dim).to(device)

    t = torch.from_numpy(x_t).float().to(device)
    i = torch.from_numpy(x_i).float().to(device)
    c = torch.from_numpy(x_c).float().to(device) if x_c is not None else None

    n = t.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.moe_epochs)

    history = []
    best_loss, best_state = float("inf"), None
    batch_size = 256

    for epoch in range(args.moe_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            optimizer.zero_grad()
            _, loss = model(t[idx], i[idx], c[idx] if c is not None else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        epoch_loss /= n

        rec = float(getattr(model, "_monitor_loss_rec_real", 0.0))
        ortho = float(getattr(model, "_monitor_loss_ortho_real", 0.0))
        gates = list(getattr(model, "_monitor_weights", np.zeros(model.num_experts)))
        history.append({"total": epoch_loss, "rec": rec, "ortho": ortho, "gates": gates})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0:
            print(
                f"[MoE] epoch {epoch}: total={epoch_loss:.4f} rec={rec:.4f} ortho={ortho:.4f} "
                f"gates={['%.3f' % g for g in gates]}"
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fused, _ = model(t, i, c if c is not None else None)
    plot_moe_training_curves(history, is_collab=(x_c is not None), mode=mode, save_dir=args.root)
    return fused.cpu().numpy()


def train_contrastive(x_t, x_i, args, mode):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adapter = ContrastiveAdapter(x_t.shape[1]).to(device)
    t = torch.from_numpy(x_t).float().to(device)
    i = torch.from_numpy(x_i).float().to(device)
    n = t.shape[0]
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    batch_size = 512
    adapter.train()
    for epoch in range(args.moe_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            optimizer.zero_grad()
            loss = adapter(t[idx], i[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if epoch % 5 == 0:
            print(f"[Contrastive] epoch {epoch}: loss={epoch_loss / n:.4f}")
    adapter.eval()
    with torch.no_grad():
        fused = adapter.get_fused_features(t, i)
    return fused.cpu().numpy()


# --------------------------------------------------------------------------- #
# 保存
# --------------------------------------------------------------------------- #


def save_embeddings(args, matrix):
    file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-{args.mode}.npy")
    np.save(file_path, matrix)
    print(f"Saved embeddings ({matrix.shape}) -> {file_path}")


def export_image2text_item_json(args, item_data_list, generated):
    """Export item json with title empty and description set to generated text."""
    path = os.path.join(args.root, f"{args.dataset}.item-{args.mode}.json")
    item_json = load_json(os.path.join(args.root, f"{args.dataset}.item.json"))
    out = {}
    for item_id, text, img_path, data in item_data_list:
        entry = dict(data)
        entry["title"] = ""
        entry["description"] = generated.get(item_id, text)
        out[str(item_id)] = entry
    # 保持 item.json 原始键顺序不变(仅替换 title/description)
    for k in item_json:
        if k not in out:
            out[k] = item_json[k]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved image2text item json -> {path}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Industrial_and_Scientific")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--plm_checkpoint", type=str, required=True)
    p.add_argument("--plm_name", type=str, default="Qwen3VL")
    p.add_argument("--mode", type=str, default="multimodal", choices=sorted(MODES))
    p.add_argument("--collab_emb_path", type=str, default="")
    p.add_argument("--text_weight", type=float, default=-1.0)
    p.add_argument("--combine_orig_text", action="store_true")
    p.add_argument("--moe_epochs", type=int, default=30)
    p.add_argument("--max_sent_len", type=int, default=1024)
    p.add_argument("--max_gen_tokens", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=60))])

    if accelerator.is_main_process:
        print(f"Mode: {args.mode}, processes: {accelerator.num_processes}")

    item_data_list, missing_img = load_item_data(args)
    n_items = len(item_data_list)
    if accelerator.is_main_process:
        print(f"Items: {n_items}, missing image: {missing_img}")

    # collab embedding
    collab_embeddings = None
    if "_collab" in args.mode:
        if not args.collab_emb_path or not os.path.exists(args.collab_emb_path):
            raise FileNotFoundError(f"collab mode 需要 --collab_emb_path: {args.collab_emb_path}")
        collab_embeddings = np.load(args.collab_emb_path)
        if accelerator.is_main_process:
            print(f"Collab embeddings: {collab_embeddings.shape}")

    # 加载 VLM
    with accelerator.main_process_first():
        processor = AutoProcessor.from_pretrained(args.plm_checkpoint, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.plm_checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": accelerator.local_process_index},
        )
    model.eval()

    mode = args.mode
    is_collab = "_collab" in mode

    # ---- 单模态 / 原生多模态 ----
    if mode in ("text_only", "image_only", "multimodal", "multimodal_collab"):
        kind = {"text_only": "text", "image_only": "image", "multimodal": "multimodal", "multimodal_collab": "multimodal"}[mode]
        results = compute_embedding(args, item_data_list, processor, model, accelerator, kind)
        release_model(model)
        accelerator.wait_for_everyone()
        gathered = gather_object(results)
        if accelerator.is_main_process:
            mat = _to_matrix(gathered, n_items)
            if mode == "multimodal_collab":
                mat = np.concatenate([mat, collab_embeddings], axis=-1)
            save_embeddings(args, mat)

    # ---- image2text ----
    elif mode in ("image2text", "image2text_supp"):
        results, generated = compute_image2text_embedding(args, item_data_list, processor, model, accelerator, mode)
        release_model(model)
        accelerator.wait_for_everyone()
        gathered = gather_object(results)
        gathered_gen = gather_object(generated)
        if accelerator.is_main_process:
            mat = _to_matrix(gathered, n_items)
            save_embeddings(args, mat)
            merged_gen = {}
            for g in gathered_gen:
                merged_gen.update(g)
            export_image2text_item_json(args, item_data_list, merged_gen)

    # ---- concat / MoE / contrastive ----
    else:
        need_image = mode in ("concat", "concat_collab", "moe_decouple", "moe_collab", "contrastive_adapter")
        text_results = compute_embedding(args, item_data_list, processor, model, accelerator, "text")
        image_results = compute_embedding(args, item_data_list, processor, model, accelerator, "image") if need_image else []
        release_model(model)
        accelerator.wait_for_everyone()
        gathered_text = gather_object(text_results)
        gathered_image = gather_object(image_results)
        if accelerator.is_main_process:
            x_t = _to_matrix(gathered_text, n_items)
            x_i = _to_matrix(gathered_image, n_items) if need_image else None

            if mode == "concat":
                mat = np.concatenate([x_t, x_i], axis=-1)
            elif mode == "concat_collab":
                mat = np.concatenate([x_t, x_i, collab_embeddings], axis=-1)
            elif mode == "moe_decouple":
                mat = train_moe(x_t, x_i, None, args, None, mode)
            elif mode == "moe_collab":
                mat = train_moe(x_t, x_i, collab_embeddings, args, collab_embeddings.shape[1], mode)
            elif mode == "contrastive_adapter":
                mat = train_contrastive(x_t, x_i, args, mode)
            save_embeddings(args, mat)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Done.")


if __name__ == "__main__":
    main()
