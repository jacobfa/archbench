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

from archbench.config import load_config  # noqa: E402
from archbench.model import LM  # noqa: E402

VOCAB = 50304


def count(path: str) -> tuple[int, int, dict]:
    cfg = load_config([path])
    cfg.model.vocab_size = VOCAB
    with torch.device("meta"):
        model = LM(cfg.model)
    return model.num_params(), model.num_params(non_embedding=True), cfg.model


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    dirs = sys.argv[1:] or sorted(
        (d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and os.path.basename(d) != "arch"),
        key=lambda d: float(os.path.basename(d).rstrip("mb")) * (1e3 if os.path.basename(d).endswith("b") else 1),
    )
    for d in dirs:
        rows = []
        for path in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            total, non_emb, m = count(path)
            rows.append((os.path.splitext(os.path.basename(path))[0], total, non_emb, m))
        mean = sum(r[1] for r in rows) / len(rows)
        print(f"\n{os.path.basename(d)}  (mean total {mean / 1e6:.1f}M)")
        print(f"  {'config':14s} {'total':>9s} {'non-emb':>9s} {'dev':>7s}  shape")
        for name, total, non_emb, m in rows:
            heads = m.layer.get("num_heads", "-")
            print(f"  {name:14s} {total / 1e6:8.1f}M {non_emb / 1e6:8.1f}M {100 * (total / mean - 1):+6.1f}%  d={m.d_model} L={m.n_layer} H={heads} mlp={m.mlp}")


if __name__ == "__main__":
    main()
