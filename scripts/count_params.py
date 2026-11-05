"""
Usage: python scripts/count_params.py [configs/65m ...]
Builds every config in each size folder and reports parameter counts.
"""

import glob
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from archbench.config import load_config, resolve_steps  # noqa: E402
from archbench.model import LM  # noqa: E402

VOCAB = 50304


def count(path: str):
    cfg = load_config([path])
    cfg.model.vocab_size = VOCAB
    with torch.device("meta"):
        model = LM(cfg.model)
    total = model.num_params()
    tokens_per_step = cfg.train.batch_size * cfg.data.seq_len
    steps = resolve_steps(cfg.train, tokens_per_step, total)
    return total, model.num_params(non_embedding=True), cfg, steps, steps * tokens_per_step


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    dirs = sys.argv[1:] or sorted(
        (d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and os.path.basename(d) != "arch"),
        key=lambda d: float(os.path.basename(d).rstrip("mb")) * (1e3 if os.path.basename(d).endswith("b") else 1),
    )
    for d in dirs:
        rows = []
        for path in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            if os.path.basename(path).startswith("_"):
                continue
            rows.append((os.path.splitext(os.path.basename(path))[0], *count(path)))
        mean = sum(r[1] for r in rows) / len(rows)
        print(f"\n{os.path.basename(d)}  (mean total {mean / 1e6:.1f}M)")
        print(f"  {'config':14s} {'total':>8s} {'non-emb':>8s} {'dev':>6s} {'tokens':>7s} {'steps':>6s} {'batch':>6s}  shape")
        for name, total, non_emb, cfg, steps, tokens in rows:
            m = cfg.model
            heads = m.layer.get("num_heads", "-")
            print(
                f"  {name:14s} {total / 1e6:7.1f}M {non_emb / 1e6:7.1f}M {100 * (total / mean - 1):+5.1f}% "
                f"{tokens / 1e9:6.2f}B {steps:6d} {cfg.train.batch_size:6d}  d={m.d_model} L={m.n_layer} H={heads} mlp={m.mlp}"
            )


if __name__ == "__main__":
    main()
