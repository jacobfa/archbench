"""
Usage:
  ./run.sh <run_name> configs/65m/fftnet.yaml [configs/optim_soap.yaml] [key=value ...]
  deepspeed --num_gpus 8 train.py configs/130m/gdn.yaml wandb.name=gdn-130m train.steps=2000
  python train.py configs/65m/fftnet.yaml train.resume=<run_name>
Sizes: configs/{40m,65m,130m,380m}/<arch>.yaml; check with scripts/count_params.py.
Logs go to train.log_dir/<run_name>, checkpoints to train.ckpt_dir/<run_name>.
"""

import sys

from archbench.config import load_config
from archbench.trainer import train

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--local_rank")]
    train(load_config(args))
