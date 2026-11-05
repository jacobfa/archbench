import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import build_mixer
from .layers.fla import FLAMixer


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: float):
        super().__init__()
        hidden = (int(2 * ratio * d_model / 3) + 63) // 64 * 64
        self.up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mixer = build_mixer(cfg.arch, cfg.d_model, layer_idx, cfg.n_layer, **cfg.layer)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps) if cfg.mlp else None
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio) if cfg.mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.vocab_size is None:
            raise ValueError("model.vocab_size must be set")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        external = {id(p) for m in self.modules() if isinstance(m, FLAMixer) for p in m.parameters()}
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and id(m.weight) not in external:
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer) if name.endswith(("o_proj", "down")) else 0.02
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def param_breakdown(self) -> dict[str, int]:
        groups = {"embed": self.embed.weight.numel(), "mixer": 0, "mlp": 0, "norm": 0, "lm_head": 0}
        for name, p in self.named_parameters():
            part = name.split(".")
            if part[0] == "blocks":
                groups[part[2] if part[2] in ("mixer", "mlp") else "norm"] += p.numel()
            elif part[0] == "norm":
                groups["norm"] += p.numel()
            elif part[0] == "lm_head":
                groups["lm_head"] += p.numel()
        return groups

    def layer_params(self) -> dict[str, dict[str, tuple[torch.Size, int]]]:
        block = self.blocks[0]
        parts = {"mixer": block.mixer, "mlp": block.mlp} if block.mlp is not None else {"mixer": block.mixer}
        return {k: {n: (p.shape, p.numel()) for n, p in m.named_parameters()} for k, m in parts.items()}

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
