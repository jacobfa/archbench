import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    arch: str = "nsgdn"
    d_model: int = 512
    n_layer: int = 8
    vocab_size: Optional[int] = None
    mlp: bool = True
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    layer: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    path: str = "/dev/shm/fftnet/data/fineweb_edu"
    seq_len: int = 1024


@dataclass
class OptimConfig:
    name: str = "muon_adamw"
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 250
    grad_clip: float = 1.0
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    muon_ns_steps: int = 5
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 8192


@dataclass
class DeepSpeedConfig:
    zero_stage: int = 0
    overlap_comm: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    steps: Optional[int] = None  # None -> tokens / tokens_per_step, else tokens_per_param * params / tokens_per_step
    tokens: Optional[float] = None  # fixed token budget shared by every model of a size (configs/<size>/_size.yaml)
    tokens_per_param: float = 20.0
    batch_size: int = 256
    grad_accum: int = 2
    eval_every: int = 250
    eval_steps: int = 32
    log_every: int = 10
    ckpt_every: int = 1000
    log_dir: str = "logs"
    ckpt_dir: str = "/dev/shm/fftnet/checkpoints"
    compile: bool = True
    dtype: str = "bfloat16"
    seed: int = 0
    resume: Optional[str] = None
    peak_tflops: Optional[float] = None


@dataclass
class WandbConfig:
    project: str = "archbench"
    entity: Optional[str] = None
    name: Optional[str] = None
    mode: str = "online"
    tags: list = field(default_factory=list)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_steps(train: TrainConfig, tokens_per_step: int, n_params: int) -> int:
    if train.steps is not None:
        return train.steps
    tokens = train.tokens if train.tokens is not None else train.tokens_per_param * n_params
    return max(1, round(tokens / tokens_per_step))


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set(d: dict, key: str, value):
    *path, last = key.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = value


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    bases = d.pop("_base_", None) or []
    merged = {}
    for base in [bases] if isinstance(bases, str) else bases:
        merged = _merge(merged, _load_yaml(os.path.join(os.path.dirname(path), base)))
    return _merge(merged, d)


def _build(cls, d: dict):
    names = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in d.items():
        t = names[k].type
        kwargs[k] = _build(t, v) if is_dataclass(t) and isinstance(v, dict) else v
    return cls(**kwargs)


def load_config(args: list[str]) -> Config:
    d = asdict(Config())
    for a in args:
        if a.endswith((".yaml", ".yml")):
            d = _merge(d, _load_yaml(a))
        elif "=" in a:
            key, _, val = a.partition("=")
            _set(d, key, yaml.safe_load(val))
        else:
            raise ValueError(f"expected a yaml path or key=value override, got {a!r}")
    return _build(Config, d)
