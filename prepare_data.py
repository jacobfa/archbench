import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text_key", default="text")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out", default="data/fineweb_edu")
    p.add_argument("--max_tokens", type=float, default=1e9)
    p.add_argument("--val_tokens", type=float, default=1e7)
    p.add_argument("--batch_docs", type=int, default=1024)
    args = p.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    vocab_size = len(tok)
    dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    max_tokens, val_tokens = int(args.max_tokens), int(args.val_tokens)

    ds = load_dataset(args.dataset, args.name or None, split=args.split, streaming=True)
    os.makedirs(args.out, exist_ok=True)
    files = {s: open(os.path.join(args.out, f"{s}.bin"), "wb") for s in ("val", "train")}
    counts = {"val": 0, "train": 0}
    pbar = tqdm(total=max_tokens, unit="tok", unit_scale=True, dynamic_ncols=True)

    def write(arr: np.ndarray):
        take = min(len(arr), val_tokens - counts["val"])
        if take > 0:
            files["val"].write(arr[:take].tobytes())
            counts["val"] += take
        if take < len(arr):
            files["train"].write(arr[take:].tobytes())
            counts["train"] += len(arr) - take
        pbar.update(len(arr))

    def encode(docs: list[str]) -> np.ndarray:
        ids = tok(docs, add_special_tokens=False)["input_ids"]
        return np.concatenate([np.asarray(d + [eos], dtype=dtype) for d in ids])

    batch = []
    for ex in ds:
        batch.append(ex[args.text_key])
        if len(batch) == args.batch_docs:
            write(encode(batch))
            batch = []
            if counts["val"] + counts["train"] >= max_tokens:
                break
    if batch and counts["val"] + counts["train"] < max_tokens:
        write(encode(batch))
    pbar.close()
    for f in files.values():
        f.close()

    meta = {
        "dataset": args.dataset,
        "name": args.name,
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
