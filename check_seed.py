"""Verify determinism of evaluate.py by running it twice on the same small input.

Usage:
  python check_seed.py --model_name_or_path <ckpt> --test_data_path <json> \
      --index_path <id2sid.json> --num_beams 20
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def run_eval(args, out_path, seed):
    cmd = [
        sys.executable, "-u", "evaluate.py",
        "--model_name_or_path", args.model_name_or_path,
        "--test_data_path", args.test_data_path,
        "--result_json_path", out_path,
        "--index_path", args.index_path,
        "--batch_size", "4",
        "--max_new_tokens", "64",
        "--num_beams", str(args.num_beams),
        "--temperature", "1.0",
        "--do_sample", "False",
        "--length_penalty", "0.0",
        "--seed", str(seed),
        "--compute_metrics_flag", "False",
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--test_data_path", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--num_beams", type=int, default=20)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        r1 = os.path.join(tmp, "r1.json")
        r2 = os.path.join(tmp, "r2.json")
        run_eval(args, r1, 42)
        run_eval(args, r2, 42)
        d1 = json.load(open(r1))
        d2 = json.load(open(r2))
        same = d1 == d2
        print(f"deterministic = {same}")
        if not same:
            for i, (a, b) in enumerate(zip(d1, d2)):
                if a != b:
                    print(f"first diff at sample {i}")
                    break
            sys.exit(1)


if __name__ == "__main__":
    main()
