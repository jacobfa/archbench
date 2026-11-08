"""CORE metric (DCLM, https://arxiv.org/abs/2406.11794), ported from nanochat."""

import csv
import json
import os
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


def _bos(tok) -> int:
    return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id


def _download(dest: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eval_bundle.zip")
        urllib.request.urlretrieve(BUNDLE_URL, path)
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(os.path.join(tmp, "eval_bundle"), dest)


def _prompts(kind: str, item: dict, delim: str, shots: list[dict]) -> list[str]:
    if kind == "multiple_choice":
        prefix = "".join(f"{e['query']}{delim}{e['choices'][e['gold']]}\n\n" for e in shots)
        return [f"{prefix}{item['query']}{delim}{c}" for c in item["choices"]]
    if kind == "schema":
        prefix = "".join(f"{e['context_options'][e['gold']]}{delim}{e['continuation']}\n\n" for e in shots)
        return [f"{prefix}{c}{delim}{item['continuation']}" for c in item["context_options"]]
    if kind == "language_modeling":
        prefix = "".join(f"{e['context'].strip()}{delim}{e['continuation']}\n\n" for e in shots)
        ctx = f"{prefix}{item['context'].strip()}{delim}"
        return [ctx.strip(), ctx + item["continuation"]]
    raise ValueError(f"unsupported task type {kind!r}")


def _common_len(seqs: list[list[int]], suffix: bool = False) -> int:
    n = min(map(len, seqs))
    for i in range(n):
        j = -1 - i if suffix else i
        if any(s[j] != seqs[0][j] for s in seqs):
            return i
    return n


@torch.no_grad()
def _example(fn, tok, data: list[dict], idx: int, task: dict, max_len: int | None, device) -> bool:
    item, kind, bos = data[idx], task["icl_task_type"], _bos(tok)
    n_shot = task["num_fewshot"][0]
    shots = [data[i] for i in random.Random(1234 + idx).sample([i for i in range(len(data)) if i != idx], n_shot)] if n_shot else []
    prompts = _prompts(kind, item, task.get("continuation_delimiter", " "), shots)
    tokens = [[bos] + t for t in tok(prompts, add_special_tokens=False)["input_ids"]]
    if kind == "multiple_choice":
        spans = [(_common_len(tokens), len(t)) for t in tokens]
    elif kind == "schema":
        n = _common_len(tokens, suffix=True)
        spans = [(len(t) - n, len(t)) for t in tokens]
    else:
        tokens, spans = tokens[1:], [(_common_len(tokens), len(tokens[1]))]
    if max_len:
        crop = [max(0, len(t) - max_len) for t in tokens]
        tokens = [t[c:] for t, c in zip(tokens, crop)]
        spans = [(max(1, s - c), e - c) for (s, e), c in zip(spans, crop)]
    ids = torch.full((len(tokens), max(map(len, tokens))), bos, dtype=torch.long)
    for i, t in enumerate(tokens):
        ids[i, : len(t)] = torch.tensor(t)
    ids = ids.to(device)
    logits = fn(ids)[..., : len(tok)].float()
    if kind == "language_modeling":
        s, e = spans[0]
        return bool((logits[0, s - 1 : e - 1].argmax(-1) == ids[0, s:e]).all())
    losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    means = [losses[i, s - 1 : e - 1].mean().item() for i, (s, e) in enumerate(spans)]
    return means.index(min(means)) == item["gold"]


def evaluate_core(fn, tok, device, bundle_dir: str, max_per_task: int = -1, max_len: int | None = None, log=None):
    """fn maps (B, T) token ids to (B, T, V) logits. Returns (core_metric, accuracy per task, centered accuracy per task).
    If log is given, each task's result is passed to it as a string when it completes (rank 0 only)."""
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    bundle_dir = os.path.expanduser(bundle_dir)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and not os.path.exists(bundle_dir):
        _download(bundle_dir)
    if world > 1:
        dist.barrier()
    with open(os.path.join(bundle_dir, "core.yaml")) as f:
        tasks = yaml.safe_load(f)["icl_tasks"]
    with open(os.path.join(bundle_dir, "eval_meta_data.csv")) as f:
        baselines = {r["Eval Task"]: 0.01 * float(r["Random baseline"]) for r in csv.DictReader(f)}
    acc, centered = {}, {}
    for task in tasks:
        with open(os.path.join(bundle_dir, "eval_data", task["dataset_uri"]), encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        t0 = time.perf_counter()
        correct = torch.zeros(len(data), device=device)
        for i in range(rank, len(data), world):
            correct[i] = float(_example(fn, tok, data, i, task, max_len, device))
        if world > 1:
            dist.all_reduce(correct)
        label, b = task["label"], baselines[task["label"]]
        acc[label] = correct.mean().item()
        centered[label] = (acc[label] - b) / (1 - b)
        if log is not None and rank == 0:
            log(
                f"  core {label:28s} {task['num_fewshot'][0]}-shot {task['icl_task_type']:28s} "
                f"acc {acc[label]:.4f} | centered {centered[label]:+.4f} | took {time.perf_counter() - t0:.1f}s"
            )
    return sum(centered.values()) / len(centered), acc, centered


@torch.no_grad()
def sample(fn, tok, device, prompts: list[str] = SAMPLE_PROMPTS, max_tokens: int = 16) -> list[str]:
    out = []
    for p in prompts:
        ids = torch.tensor([[_bos(tok)] + tok(p, add_special_tokens=False)["input_ids"]], device=device)
        for _ in range(max_tokens):
            ids = torch.cat([ids, fn(ids)[:, -1:, : len(tok)].argmax(-1)], dim=1)
        out.append(tok.decode(ids[0, 1:]))
    return out
