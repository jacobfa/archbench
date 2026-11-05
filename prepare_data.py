import argparse
import json
import os
import threading
from multiprocessing import Pool

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

_tok = _eos = _dtype = _key = None
_files: dict = {}


def _init(tokenizer, dtype: str, text_key: str):
    global _tok, _eos, _dtype, _key
    os.environ["HF_HUB_OFFLINE"] = "1"
    _tok, _eos, _dtype, _key = tokenizer, tokenizer.eos_token_id, np.dtype(dtype), text_key


def _encode(task) -> bytes:
    path, rg = task
    pf = _files.get(path) or _files.setdefault(path, pq.ParquetFile(path))
    docs = pf.read_row_group(rg, columns=[_key]).column(0).to_pylist()
    ids = _tok(docs, add_special_tokens=False)["input_ids"]
    return np.concatenate([np.asarray(d + [_eos], dtype=_dtype) for d in ids]).tobytes()


def _tasks(repo: str, files: list[str], lookahead: int):
    results, done = [None] * len(files), [threading.Event() for _ in files]

    def fetch(i):
        try:
            results[i] = hf_hub_download(repo, files[i], repo_type="dataset")
        except BaseException as e:
            results[i] = e
        done[i].set()

    def start(i):
        threading.Thread(target=fetch, args=(i,), daemon=True).start()

    for i in range(min(lookahead, len(files))):
        start(i)
    for i in range(len(files)):
        done[i].wait()
        if isinstance(results[i], BaseException):
            raise results[i]
        if i + lookahead < len(files):
            start(i + lookahead)
        for rg in range(pq.ParquetFile(results[i]).metadata.num_row_groups):
            yield results[i], rg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--prefix", default="sample/10BT/")
    p.add_argument("--text_key", default="text")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out", default="/dev/shm/fftnet/data/fineweb_edu")
    p.add_argument("--max_tokens", type=float, default=1e9)
    p.add_argument("--val_tokens", type=float, default=1e7)
    p.add_argument("--workers", type=int, default=min(64, os.cpu_count() // 2))
    p.add_argument("--lookahead", type=int, default=2)
    args = p.parse_args()

    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tok)
    dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    itemsize = np.dtype(dtype).itemsize
    max_tokens, val_tokens = int(args.max_tokens), int(args.val_tokens)

    files = sorted(f for f in HfApi().list_repo_files(args.dataset, repo_type="dataset") if f.startswith(args.prefix) and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {args.prefix!r} in {args.dataset}")

    os.makedirs(args.out, exist_ok=True)
    out = {s: open(os.path.join(args.out, f"{s}.bin"), "wb") for s in ("val", "train")}
    counts = {"val": 0, "train": 0}
    pbar = tqdm(total=max_tokens, unit="tok", unit_scale=True, dynamic_ncols=True)

    with Pool(args.workers, initializer=_init, initargs=(tok, np.dtype(dtype).name, args.text_key)) as pool:
        for buf in pool.imap(_encode, _tasks(args.dataset, files, args.lookahead), chunksize=1):
            n = len(buf) // itemsize
            take = min(n, val_tokens - counts["val"])
            if take > 0:
                out["val"].write(buf[: take * itemsize])
                counts["val"] += take
            if take < n:
                out["train"].write(buf[take * itemsize :])
                counts["train"] += n - take
            pbar.update(n)
            if counts["val"] + counts["train"] >= max_tokens:
                pool.terminate()
                break
    pbar.close()
    for f in out.values():
        f.close()

    meta = {
        "dataset": args.dataset,
        "name": args.prefix,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
