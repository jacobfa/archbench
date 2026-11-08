"""
Usage:
  ./run.sh <run_name> configs/65m/pgdn.yaml [configs/optim_soap.yaml] [key=value ...]
  deepspeed --num_gpus 8 train.py configs/130m/gdn.yaml wandb.name=gdn-130m train.steps=2000
  python train.py configs/65m/pgdn.yaml train.resume=<run_name>
  python train.py configs/65m/gdn.yaml configs/no_tweaks.yaml   # baseline without the nanochat-style tweaks
  python scripts/eval_core.py <ckpt_dir>/<run_name> [best|last]  # CORE on a saved checkpoint
Sizes: configs/{40m,65m,130m,380m,650m,1b}/<arch>.yaml; check with scripts/count_params.py.
Every model of a size trains on the same token budget (train.tokens in configs/<size>/_size.yaml);
steps = train.tokens / tokens_per_step. Without train.tokens, steps = tokens_per_param * params / tokens_per_step.
Set train.steps to override either.
Logs go to train.log_dir/<run_name>, checkpoints to train.ckpt_dir/<run_name>.
"""

import sys

from archbench.config import load_config
from archbench.trainer import train

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--local_rank")]
    train(load_config(args))
