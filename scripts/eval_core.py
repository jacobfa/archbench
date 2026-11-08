"""
Usage: python scripts/eval_core.py <ckpt_dir>/<run_name> [tag=best] [max_per_task=-1]
Evaluates a saved checkpoint on the CORE benchmark (downloaded to train.core_dir on first use).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archbench.config import Config  # noqa: E402
from archbench.core_eval import evaluate_core  # noqa: E402
from archbench.model import LM  # noqa: E402


def main():
    args = sys.argv[1:]
    run_dir = args[0]
    tag = args[1] if len(args) > 1 else "best"
    max_per_task = int(args[2]) if len(args) > 2 else -1
    state = torch.load(os.path.join(run_dir, tag, "mp_rank_00_model_states.pt"), map_location="cpu", weights_only=False)
    cfg = Config.from_dict(state["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LM(cfg.model).to(device)
    model.load_state_dict(state["module"])
    model.eval()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(state.get("tokenizer", "gpt2"))
    print(f"{run_dir}/{tag}  step {state['step']}  arch {cfg.model.arch}  params {model.num_params():,}")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    with autocast:
        metric, _, _ = evaluate_core(lambda x: model(x)[0], tok, device, cfg.train.core_dir, max_per_task, cfg.data.seq_len, verbose=True)
    print(f"{'CORE':35s} {metric:.4f}")


if __name__ == "__main__":
    main()
