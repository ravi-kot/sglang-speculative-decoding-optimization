"""Build fixed, exact-length token-id prompt sets (deterministic; identical inputs on every GPU).

  python data/build_prompts.py --model <hf snapshot dir>            # build all sets
  python data/build_prompts.py --model <other snapshot> --verify    # check template/tokenizer match

Sources (raw files pinned by sha256 in SOURCES; data/raw/ is gitignored):
  gutenberg     Pride and Prejudice (harness validation only)
  mtbench       MT-Bench first turns (FastChat b494d0c6), natural length (short-context control)
  agentic_code  repository files rendered as agent tool outputs + a coding task (primary workload)

Each prompt = chat-template prefix ids + context ids truncated to fit + suffix ids, so the
length is exact. Thinking is disabled in the template. Output: data/prompts/<set>.json.gz
"""

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "prompts"
SOURCES = {
    "mtbench_question.jsonl": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    "pg1342.txt": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
}
LENGTHS = (1024, 4096, 8192, 16384)
N_PER_SET = 10  # bench.py uses the last `warmup` prompts of a set for warmup only
MARK = "⁣CTX⁣"
CODE_TASKS = [
    "Identify the most likely bug or unhandled edge case in the files above. Explain the failure "
    "scenario, then write the corrected version of the affected function.",
    "Trace the main call path through the files above. Explain each step, then write pytest unit "
    "tests for the function you consider most critical.",
    "Refactor the longest function in the last file above for readability without changing its "
    "behavior. Output the full refactored function and a short justification.",
    "Propose a minimal patch that adds input validation and clear error messages to the public "
    "functions in the files above. Show the patch as a unified diff.",
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def ids_sha(seqs):
    return sha(json.dumps(seqs, separators=(",", ":")).encode())


def template_pieces(tok, instruction_before, instruction_after):
    msgs = [{"role": "user", "content": f"{instruction_before}{MARK}{instruction_after}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    pre, post = text.split(MARK)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    return enc(pre), enc(post)


def exact(tok, ctx_ids, before, after, length):
    pre, post = template_pieces(tok, before, after)
    k = length - len(pre) - len(post)
    assert k > 0 and len(ctx_ids) >= k, (length, len(ctx_ids))
    return pre + ctx_ids[:k] + post


def code_corpus():
    import sglang

    root = Path(sglang.__file__).resolve().parent
    files = sorted(p for p in (root / "srt").rglob("*.py") if 4_000 < p.stat().st_size < 60_000)
    return root, files, sglang.__version__


def build(tok):
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    sets = {}

    # mtbench: first turn, natural length
    rows = [json.loads(l) for l in (RAW / "mtbench_question.jsonl").read_text().splitlines()]
    sets["mtbench"] = [enc(tok.apply_chat_template([{"role": "user", "content": r["turns"][0]}],
                                                   tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)) for r in rows]

    book = enc((RAW / "pg1342.txt").read_text(encoding="utf-8"))
    root, files, sgl_ver = code_corpus()
    for L in LENGTHS:
        rng = random.Random(L)
        out = []
        for i in range(N_PER_SET if L <= 4096 else 0):  # gutenberg: harness only
            start = rng.randrange(0, len(book) - L)
            out.append(exact(tok, book[start:start + L], "Here is a passage from a novel:\n\n",
                             "\n\nSummarize the passage in detail, scene by scene.", L))
        if out:
            sets[f"gutenberg_{L}"] = out
        out = []
        for i in range(N_PER_SET):
            picked = rng.sample(files, len(files))  # distinct file order per prompt: no shared prefix
            ctx, n = [], 0
            for p in picked:
                block = enc(f"\n[tool:read_file] {p.relative_to(root.parent)}\n```python\n{p.read_text()}\n```\n")
                ctx += block
                n += len(block)
                if n > L:
                    break
            out.append(exact(tok, ctx, "You are a coding agent working in the SGLang repository. "
                             "The tool outputs below show files you opened.\n",
                             "\n\nTask: " + CODE_TASKS[i % len(CODE_TASKS)], L))
        sets[f"agentic_code_{L}"] = out
    return sets, sgl_ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local HF snapshot dir (tokenizer + chat template)")
    ap.add_argument("--verify", action="store_true", help="only check template/tokenizer equivalence")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok_sha = sha((Path(a.model) / "tokenizer.json").read_bytes())
    pieces = template_pieces(tok, "A", "B")
    if a.verify:  # prompts built with another snapshot are valid iff these match
        meta = json.loads((OUT / "meta.json").read_text())
        ok = meta["tokenizer_sha"] == tok_sha and meta["template_pieces"] == [list(p) for p in pieces]
        print("VERIFY", "OK" if ok else "MISMATCH: rebuild prompts with this model")
        raise SystemExit(0 if ok else 1)
    for f, h in SOURCES.items():
        assert sha((RAW / f).read_bytes()) == h, f"raw file changed: {f}"
    sets, sgl_ver = build(tok)
    OUT.mkdir(exist_ok=True)
    meta = {"tokenizer_sha": tok_sha, "template_pieces": [list(p) for p in pieces],
            "built_with": str(a.model), "code_corpus": f"sglang {sgl_ver} srt/*.py", "sets": {}}
    for name, seqs in sets.items():
        with gzip.GzipFile(OUT / f"{name}.json.gz", "wb", mtime=0) as fh:  # mtime=0: reproducible bytes
            fh.write(json.dumps(seqs, separators=(",", ":")).encode())
        lens = [len(s) for s in seqs]
        meta["sets"][name] = {"n": len(seqs), "len_min": min(lens), "len_max": max(lens),
                              "sha256": ids_sha(seqs)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    for k, v in meta["sets"].items():
        print(f"{k:<20} n={v['n']:<3} len={v['len_min']}..{v['len_max']} {v['sha256'][:12]}")


if __name__ == "__main__":
    main()
