<p align="center">
  <img src="assets/MIMIGenRec_logo.png" alt="MIMIGenRec logo"/>
</p>

<p align="center"><strong>SAM-GR: Multimodal Generative Recommendation with Collaborative-Semantic Aggregation and Mixture-of-Experts Fusion</strong></p>

<p align="center"><strong>Built on <a href="https://github.com/cocoshe/MIMIGenRec">MIMIGenRec</a></strong></p>

---

## About this repository

This repository **inherits [MIMIGenRec](https://github.com/cocoshe/MIMIGenRec)** (Modular, Integrated, Mutable, Interchangeable GenRec), a flexible training framework for **generative recommendation**, and extends it with new multimodal item-representation learning and end-to-end data pipelines. All credits for the base framework go to the MIMIGenRec authors.

### Inherited from MIMIGenRec

- **SFT stage**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)-based supervised fine-tuning via YAML configs (0.5B / 1.5B / 3B, LoRA / full, DeepSpeed ZeRO).
- **RL stage**: [TRL](https://github.com/huggingface/trl) `GRPOTrainer` with custom ranking rewards (NDCG / HR) and constrained beam-search rollout over SIDs via a **flexible Trie**.
- **SID construction**: [RQ-VAE](https://github.com/AkaliKong/MiniOneRec) semantic IDs, `new_tokens.json` / `id2sid.json` formats, SFT/RL dataset schemas and registries.

### What is new in this repository

- **Multimodal item embeddings**: item representations produced by **Qwen3-VL** from both product text and product images, replacing the text-only embeddings of the original pipeline.
- **Collaborative-semantic alignment (CSGA)**: *Collaborative Similarity Graph Aggregation* — collaborative signals (SASRec item embeddings) define an implicit item-item Top-K graph; each item aggregates the multimodal semantic features of its collaborative neighbors with learned attention weights, trained to reconstruct its own semantic feature (graph-guided denoising).
- **Mixture-of-Experts fusion**:
  - **OMoE** — orthogonal Mixture-of-Experts with modality-specific + shared experts, orthogonality regularization and load balancing.
  - **OMoE v2** — discriminability-aware variant (2 shared experts, near-duplicate repulsion and variance-matching losses) that avoids rank collapse.
  - **CSGA→OMoE pipelines** — CSGA alignment followed by OMoE / OMoE-v2 fusion of text, image and collaboratively-aligned features.
- **14 fusion modes** in a single script (`rq/text2emb/multimodal_amazon_text2emb.py`): `multimodal`, `text_only`, `image_only`, `concat`, `image2text`, `image2text_supp`, `moe_decouple`, `contrastive_adapter`, and the collaborative variants `multimodal_collab`, `concat_collab`, `moe_collab`, `csga_collab`, `csga_moe_collab`, `csga_moe_collab_v2`.
- **4-layer RQ-VAE** SIDs (4 semantic tokens per item) with controllable low-frequency item merging (`SID_MERGE_RATIO`).
- **One-click pipeline** (`main.sh`): 7 stages from raw data to SFT + evaluation, with stage-level resume (`START_STAGE`), multi-GPU support and background execution via `nohup_run.sh`.
- **More datasets**: `Industrial_and_Scientific`, `Sports_and_Outdoors`, `Toys_and_Games`, and `Books` (including download / filtering / image-download tooling).
- **Updated baselines**: GRU4Rec, Caser and SASRec with collaborative embedding export for the `*_collab` fusion modes.

## Pipeline overview

```
raw reviews + meta + images (data/)
        │
  ① possess_data ──────────────► item.json + *.inter + image list
        │
  ② SASRec (collab modes only) ► sasrec_item_emb.npy
        │
  ③ multimodal text2emb (Qwen3-VL + optional CSGA/OMoE fusion)
        │                        ► {category}.emb-{model}-{mode}.npy
  ④ RQ-VAE (4-layer codebook)  ► best_collision_model.pth
        │
  ⑤ generate_indices           ► {category}.index.json
        │
  ⑥ preprocess (sid_num=4)     ► sft/ + rl/ + new_tokens.json
        │
  ⑦ SFT (LlamaFactory) + evaluate
```

## Environment & installation

```bash
git clone git@github.com:Echolarity/SAM-GR.git
cd SAM-GR
pip install -e .
```

This installs LLaMA-Factory in editable mode with all dependencies (PyTorch, transformers, TRL, accelerate, peft, datasets, etc.). The `llamafactory-cli` command is available after install.

Optionally:

- Set `HF_ENDPOINT` (e.g. `https://hf-mirror.com`) if you use a mirror.
- Set `WANDB_API_KEY` / `WANDB_PROJECT` for experiment logging.

> **Tested versions**. If the pipeline fails due to version incompatibilities, align your environment with:

| Package       | Version     |
|---------------|-------------|
| Python        | 3.12        |
| torch         | 2.8.0+cu128 |
| transformers  | 4.57.1      |
| trl           | 0.24.0      |
| accelerate    | 1.11.0      |
| peft          | 0.17.1      |
| datasets      | 4.0.0       |

## Data preparation

### 1. Raw data layout

Put raw Amazon review files (`{Category}.jsonl.gz`), metadata (`meta_{Category}.jsonl.gz`) and product images under `data/`:

```
data/
├── {Category}.jsonl.gz          # reviews
├── meta_{Category}.jsonl.gz     # item metadata
├── {Category}_images/           # one image per item: {asin}_MAIN.jpg
└── ...
```

Helper scripts are provided for common categories:

```bash
# Sports_and_Outdoors + Toys_and_Games: integrity check, 5-core filtering, image list + fast download
bash nohup_run.sh run cats_images_dl bash download_cats.sh

# Books: subset building, meta/review merging, filtering
python data/books_process.py --help
python data/make_books_subset.py --help
```

### 2. Multimodal embedding model

Download a VLM checkpoint (default `Qwen3-VL-8B-Instruct`) and point `TEXT2EMB_MODEL` to it:

```bash
export TEXT2EMB_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

## Usage: one-click pipeline

`main.sh` orchestrates the 7 stages above. The most important switches are environment variables:

| Variable         | Default                    | Description |
|------------------|----------------------------|-------------|
| `CATEGORY`       | `Industrial_and_Scientific`| Dataset category |
| `MODAL_CHOICE`   | `text_only`                | Fusion mode (see table below) |
| `CUDA_VISIBLE`   | `0,1,2,3`                  | GPUs for text2emb / SFT |
| `MONO_CUDA`      | `0`                        | GPU for RQ-VAE / index generation |
| `TEXT2EMB_MODEL` | local Qwen3-VL-8B path     | VLM checkpoint for item embeddings |
| `Optimization`   | `dsz2`                     | DeepSpeed stage used by the SFT YAML (`dsz0/dsz2/dsz3`) |
| `START_STAGE`    | `1`                        | Resume from stage N (1–7) |
| `ONLY_TASK1`     | `0`                        | `1` = generate only the task1 (history→SID) ablation data |
| `SID_MERGE_RATIO`| `0`                        | Merge ratio of low-frequency items into hot collaborative neighbors (collab modes) |
| `USER_K` / `ITEM_K` | `5` / `5`              | 5-core filtering thresholds |
| `SAMPLE_SIZE`    | `-1` (all)                 | Limit number of users |
| `MOE_EPOCHS`     | `30`                       | Fusion training epochs (MoE / CSGA / contrastive modes) |
| `SASREC_*`       | —                          | SASRec collab-embedding training hyper-parameters |

**Fusion modes** (`MODAL_CHOICE`):

| Mode | Description |
|------|-------------|
| `text_only` | Text-only item embeddings (original MIMIGenRec-style) |
| `image_only` | Image-only embeddings |
| `multimodal` | Concatenated text+image embeddings from the VLM |
| `concat` | Concat text/image embeddings |
| `image2text` / `image2text_supp` | VLM caption of the product image (optionally combined with original text) |
| `moe_decouple` | MoE decoupled fusion of text/image |
| `contrastive_adapter` | Contrastive-adapter fusion |
| `multimodal_collab` / `concat_collab` / `moe_collab` | Above fusions with SASRec collaborative embeddings |
| `csga_collab` | CSGA: collaborative-graph-guided aggregation of multimodal neighbors |
| `csga_moe_collab` | CSGA → OMoE fusion |
| `csga_moe_collab_v2` | CSGA → OMoE-v2 fusion |

### Run

```bash
# Full pipeline, e.g. CSGA -> OMoE-v2 on Industrial_and_Scientific
CATEGORY=Industrial_and_Scientific MODAL_CHOICE=csga_moe_collab_v2 bash main.sh

# In the background with logs (nohup_logs/)
bash nohup_run.sh run pipeline bash main.sh

# Resume from stage 5 (skips data/embedding/RQ-VAE stages, validates artifacts)
START_STAGE=5 CATEGORY=Industrial_and_Scientific MODAL_CHOICE=csga_moe_collab_v2 bash main.sh

# Baselines (text-only, task1-only ablation)
CATEGORY=Industrial_and_Scientific MODAL_CHOICE=text_only ONLY_TASK1=1 bash main.sh
```

### Manual stage-by-stage execution

The stages can also be run individually (see `main.sh` for exact commands):

```bash
# 1. Preprocess raw data -> item.json, *.inter, image list
python data/possess_data.py --category Industrial_and_Scientific --output_path ./data/Amazon --raw_dir ./data --img_dir ./data/Industrial_and_Scientific_images

# 2. (collab modes) Train SASRec and export collaborative item embeddings
bash scripts/rec_zoo/train_sasrec.sh --data_dir data/Amazon/Industrial_and_Scientific

# 3. Multimodal item embeddings (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 rq/text2emb/multimodal_amazon_text2emb.py \
    --dataset Industrial_and_Scientific --root data/Amazon/Industrial_and_Scientific \
    --plm_checkpoint /path/to/Qwen3-VL-8B-Instruct --plm_name Qwen3VL \
    --mode csga_moe_collab_v2 --collab_emb_path sasrec_models/Industrial_and_Scientific/sasrec_item_emb.npy

# 4-5. RQ-VAE + SID indices
python rq/rqvae.py --data_path data/Amazon/Industrial_and_Scientific/Industrial_and_Scientific.emb-Qwen3VL-csga_moe_collab_v2.npy --ckpt_dir rq_output/Industrial_and_Scientific --num_emb_list 256 256 256 256
python rq/generate_indices.py --dataset Industrial_and_Scientific --ckpt_path rq_output/Industrial_and_Scientific/best_collision_model.pth --output_dir data/Amazon/Industrial_and_Scientific

# 6. SFT/RL datasets (4 SIDs per item)
python preprocess_data_sft_rl.py --data_dir ./data/Amazon --category Industrial_and_Scientific --output_dir data/Industrial_and_Scientific --sid_num 4
```

## SFT, RL and evaluation

```bash
# SFT (LlamaFactory; edit sft.sh for GPUs / model size / DeepSpeed stage)
bash sft.sh

# RL (TRL GRPOTrainer; set MODEL_PATH / DATA_DIR / INDEX_PATH / OUTPUT_DIR in trl_trainer.sh)
bash trl_trainer.sh

# Evaluation (set exp_name / test_data_path / output_dir in evaluate.sh)
bash evaluate.sh
```

Ready-made SFT configs for the new pipelines are in `examples/train_full/{Category}/new_industry_rec_full_sft_0.5b_*.yaml` (one per fusion mode / DeepSpeed stage / ablation).

## Custom dataset (SFT/RL only)

If you only need SFT/RL with your own data, first build SIDs for your items, then provide under `data/{Category}/`:

```
data/{Category}/
├── new_tokens.json    # SID vocabulary (JSON array, e.g. ["<a_100>", "<b_230>", "<c_0>"])
├── id2sid.json        # item_id -> [sid1, sid2, sid3]
├── sft/{train,valid,test}.json
└── rl/{train,valid,test}.json
```

Register the datasets in `data/dataset_info.json`, point `add_tokens_list` in your SFT YAML to `new_tokens.json`, then run `llamafactory-cli train your_config.yaml`. See the [MIMIGenRec README](https://github.com/cocoshe/MIMIGenRec) for the exact JSON schemas.

## Baselines

```bash
bash scripts/rec_zoo/train_gru.sh   --data_dir data/{Category}
bash scripts/rec_zoo/train_caser.sh --data_dir data/{Category}
bash scripts/rec_zoo/train_sasrec.sh --data_dir data/{Category}
```

SASRec additionally exports `sasrec_item_emb.npy`, which is required by all `*_collab` fusion modes.

## Acknowledgments

- [MIMIGenRec](https://github.com/cocoshe/MIMIGenRec) — the base framework this repository is built upon
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT training framework
- [TRL](https://github.com/huggingface/trl) — reinforcement learning training
- [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) — first fully open-source generative recommendation framework

## License

Apache-2.0, same as the inherited MIMIGenRec codebase. See [LICENSE](LICENSE).
