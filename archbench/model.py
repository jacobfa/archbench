import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import build_mixer
from .layers.attention import Attention
from .layers.fla import FLAMixer

from .fused_ce import fused_ce_available, fused_linear_cross_entropy

OUT_PROJ = ("o_proj", "out_proj", "down")
SMEAR_DIM = 24


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: float, act: str = "swiglu"):
        super().__init__()
        if act not in ("swiglu", "relu2"):
            raise ValueError(f"unknown mlp_act {act!r}; choose swiglu or relu2")
        self.act = act
        gated = act == "swiglu"
        hidden = (int((2 / 3 if gated else 1) * ratio * d_model) + 63) // 64 * 64
        self.up = nn.Linear(d_model, (2 if gated else 1) * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "relu2":
            return self.down(F.relu(self.up(x)).square())
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mixer = build_mixer(cfg.arch, cfg.d_model, layer_idx, cfg.n_layer, **cfg.layer)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps) if cfg.mlp else None
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.mlp_act) if cfg.mlp else None

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
        L = cfg.n_layer
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(L))
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.resid_lambdas = self.x0_lambdas = self.smear_gate = self.smear_lambda = self.backout_lambda = None
        if cfg.resid_scalars:
            self.resid_lambdas = nn.Parameter(torch.tensor([1.15 - 0.10 * i / max(L - 1, 1) for i in range(L)]))
            self.x0_lambdas = nn.Parameter(torch.tensor([0.20 - 0.15 * i / max(L - 1, 1) for i in range(L)]))
        if cfg.smear:
            self.smear_gate = nn.Linear(SMEAR_DIM, 1, bias=False)
            self.smear_lambda = nn.Parameter(torch.zeros(1))
        if cfg.backout:
            self.backout_lambda = nn.Parameter(torch.full((1,), 0.2))
        self._init_weights()

    def _init_weights(self):
        external = {id(p) for m in self.modules() if isinstance(m, FLAMixer) for p in m.parameters()}
        for name, m in self.named_modules():
            if not isinstance(m, (nn.Linear, nn.Embedding)) or m is self.smear_gate:
                continue
            out = name.endswith(OUT_PROJ)
            if out and self.cfg.zero_init:
                nn.init.zeros_(m.weight)
            elif id(m.weight) not in external:
                nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer) if out else 0.02)
        if self.smear_gate is not None:
            nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

    def scalar_groups(self) -> list[tuple[list[nn.Parameter], float]]:
        groups = []
        if self.resid_lambdas is not None:
            groups += [([self.resid_lambdas], 0.01), ([self.x0_lambdas], 1.0)]
        rest = [p for p in (self.smear_lambda, self.backout_lambda) if p is not None]
        rest += [self.smear_gate.weight] if self.smear_gate is not None else []
        return groups + ([(rest, 0.4)] if rest else [])

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def flops_per_token(self, seq_len: int) -> int:
        matmul = sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))
        n_attn = sum(isinstance(b.mixer, Attention) for b in self.blocks)
        return 6 * matmul + 12 * self.cfg.d_model * seq_len * n_attn

    def param_breakdown(self) -> dict[str, int]:
        groups = {"embed": self.embed.weight.numel(), "mixer": 0, "mlp": 0, "norm": 0, "lm_head": 0, "scalars": 0}
        for name, p in self.named_parameters():
            part = name.split(".")
            if part[0] == "blocks":
                groups[part[2] if part[2] in ("mixer", "mlp") else "norm"] += p.numel()
            elif part[0] in ("norm", "lm_head"):
                groups[part[0]] += p.numel()
            elif part[0] != "embed":
                groups["scalars"] += p.numel()
        return groups

    def layer_params(self) -> dict[str, dict[str, tuple[torch.Size, int]]]:
        block = self.blocks[0]
        parts = {"mixer": block.mixer, "mlp": block.mlp} if block.mlp is not None else {"mixer": block.mixer}
        return {k: {n: (p.shape, p.numel()) for n, p in m.named_parameters()} for k, m in parts.items()}

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        c = self.cfg
        x = self.embed(idx)
        if c.embed_norm:
            x = F.rms_norm(x, (x.size(-1),))
        if self.smear_gate is not None:
            gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :SMEAR_DIM]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        x0, x_mid = x, None
        for i, block in enumerate(self.blocks):
            if self.resid_lambdas is not None:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x)
            if i == c.n_layer // 2:
                x_mid = x
        if self.backout_lambda is not None:
            x = x - self.backout_lambda * x_mid
        x = self.norm(x)
        if targets is not None and fused_ce_available() and x.is_cuda and not c.logit_softcap:
            dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
            return None, fused_linear_cross_entropy(x.to(dt), targets, self.lm_head.weight.to(dt))
        logits = self.lm_head(x)
        if c.logit_softcap:
            logits = c.logit_softcap * torch.tanh(logits.float() / c.logit_softcap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
