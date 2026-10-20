import argparse
import collections
import json
import os

import numpy as np
import torch
from datasets import EmbDataset
from models.rqvae import RQVAE
from torch.utils.data import DataLoader
from tqdm import tqdm


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups


def merge_collaborative_neighbors(all_indices, collab_emb_path, train_inter, merge_ratio):
    """Intentionally merge a fraction of items into a more popular collaborative neighbour.

    This creates *moderate* SID sharing so that popular items' codes are shared by
    semantically/collaboratively close items, which lowers the generation difficulty
    (fewer distinct targets) at a controlled cost of SID-level ambiguity.

    Least-popular items are merged first, and only into a *more popular* canonical
    neighbour, so hot items keep their own codes.
    """
    n = len(all_indices)
    if merge_ratio <= 0 or not collab_emb_path or not os.path.exists(collab_emb_path):
        return all_indices
    emb = np.load(collab_emb_path).astype(np.float32)
    if emb.shape[0] != n:
        print(f"[merge] skip: collab rows {emb.shape[0]} != items {n}")
        return all_indices
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    pop = np.zeros(n, dtype=np.float64)
    if train_inter and os.path.exists(train_inter):
        with open(train_inter, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 3:
                    pop[int(p[2])] += 1

    top1 = np.empty(n, dtype=np.int64)
    chunk = 2048
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        sim = emb[st:en] @ emb.T
        sim[np.arange(en - st), np.arange(st, en)] = -2.0
        top1[st:en] = sim.argmax(axis=1)

    budget = int(merge_ratio * n)
    order = np.argsort(pop)  # least popular first
    canonical = np.ones(n, dtype=bool)
    merged = 0
    for i in order:
        if merged >= budget:
            break
        i = int(i)
        j = int(top1[i])
        if j == i or not canonical[j] or not canonical[i]:
            continue
        if pop[j] < pop[i]:
            continue
        all_indices[i] = list(all_indices[j])
        canonical[i] = False
        merged += 1
    print(f"[merge] merged {merged}/{n} items ({merged / n:.4f}) into more popular collaborative neighbours")
    return all_indices


def parse_args():
    p = argparse.ArgumentParser(description="Generate RQ-VAE indices and save to id2sid-style JSON.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. Industrial_and_Scientific)")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to best_collision_model.pth (or similar)")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory; default: data/<dataset>/")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--collab_emb_path", type=str, default="")
    p.add_argument("--train_inter", type=str, default="")
    p.add_argument("--merge_ratio", type=float, default=0.0,
                   help="fraction of (least popular) items whose SID is merged into a more popular collaborative neighbour")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    ckpt_path = args.ckpt_path
    output_dir = args.output_dir or os.path.join("data", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset}.index.json")
    device = torch.device(args.device)

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(ckpt_args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=ckpt_args.num_workers,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
    )

    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    all_indices = []
    all_indices_str = []
    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print("Collision groups:", len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    # moderate collaborative SID merging (after collision resolution)
    if args.merge_ratio > 0:
        all_indices = merge_collaborative_neighbors(
            all_indices, args.collab_emb_path, args.train_inter, args.merge_ratio
        )
        all_indices_str = np.array([str(list(x)) for x in all_indices])

    print("All indices number:", len(all_indices))
    counts = get_indices_count(all_indices_str)
    print("Max number of conflicts:", max(counts.values()) if counts else 0)
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision rate:", (tot_item - tot_indice) / tot_item if tot_item else 0)

    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
    with open(output_file, "w") as fp:
        json.dump(all_indices_dict, fp)
    print("Saved to", output_file)


if __name__ == "__main__":
    main()
