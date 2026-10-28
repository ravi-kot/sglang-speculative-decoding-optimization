"""FR-Spec style hot-token map for --speculative-token-map, from text independent of our prompts.

  python data/build_token_map.py --model <snapshot> --k 32768 65536

Corpus (local, deterministic, disjoint from the prompt sources SGLang srt/, Gutenberg, MT-Bench):
CPython stdlib *.py (code + docstrings) and package METADATA files (English prose) from the
Python install, sglang excluded. All special/added tokens (EOS, <|im_end|>, <think>, ...) are
always kept. Output: data/token_maps/hot_<k>.pt (sorted int list) + meta json with coverage.
"""

import argparse
import collections
import hashlib
import json
import sysconfig
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent / "token_maps"


def corpus():
    std = Path(sysconfig.get_paths()["stdlib"])
    site = Path(sysconfig.get_paths()["purelib"])
    files = sorted(p for p in std.rglob("*.py") if "test" not in p.parts and "site-packages" not in p.parts)
    files += sorted(p for p in site.glob("*.dist-info/METADATA") if "sglang" not in p.parent.name.lower())
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--mix", type=float, default=1.0,
                    help="fraction of k taken by corpus frequency; the rest by BPE merge rank (low id ~ frequent "
                         "in the tokenizer's own training data). <1 writes hot_<k>_mix<f>.pt")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    cnt, h = collections.Counter(), hashlib.sha256()
    for p in corpus():
        text = p.read_text(errors="ignore")
        h.update(text.encode())
        cnt.update(tok(text, add_special_tokens=False)["input_ids"])
    special = sorted(set(tok.all_special_ids) | set(tok.added_tokens_decoder))
    OUT.mkdir(exist_ok=True)
    total = sum(cnt.values())
    for k in a.k:
        keep = set(special)
        for t, _ in cnt.most_common():
            if len(keep) >= int(k * a.mix):
                break
            keep.add(t)
        filler = 0  # rest (or corpus has fewer distinct tokens than k): fill by BPE merge rank (low id ~ frequent)
        for t in range(len(tok)):
            if len(keep) >= k:
                break
            if t not in keep:
                keep.add(t)
                filler += 1
        ids = sorted(keep)
        name = f"hot_{k}" if a.mix >= 1 else f"hot_{k}_mix{a.mix:g}"
        torch.save(ids, OUT / f"{name}.pt")
        meta = {"k": len(ids), "mix": a.mix, "corpus_files": len(corpus()), "corpus_tokens": total, "corpus_sha256": h.hexdigest(),
                "corpus_coverage": round(sum(cnt[t] for t in ids) / total, 4), "special_kept": len(special), "filled_by_id_rank": filler,
                "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(meta)


if __name__ == "__main__":
    main()
