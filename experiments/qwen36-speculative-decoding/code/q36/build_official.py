"""Phase 5: replay turns from the official MLPerf Agentic Inference dataset (SHA-256 1beb24c8...14ae), the population
Coral Bricks benchmarked: all assistant turns of the 113 coding traces (rows without _source=Workato), <= 240K tokens
= 15,777 turns (15,981 - 204). Each turn = full trajectory up to that assistant message, rendered with the Qwen3.6
chat template (tools, earlier reasoning and tool calls preserved), add_generation_prompt, thinking off and on.
Token counts per turn: each conversation's deepest turn is tokenized exactly, other turns are scaled by rendered
characters; sampled turns are then tokenized exactly. Prints bin counts to compare with Coral's
(2,614 / 2,272 / 5,852 / 5,039). Writes turns_dev64.jsonl (16/bin) and turns_head128.jsonl (32/bin, disjoint).
Usage: python build_official.py --data agentic.jsonl --tok <model> --out <dir>"""
import argparse, ast, collections, hashlib, json, os, random
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--sessions-out", help="write --n-sessions official traces in order, every turn kept, and stop")
ap.add_argument("--n-sessions", type=int, default=16)
ap.add_argument("--session-turns", type=int, default=64)
ap.add_argument("--hold-out", help="write a clean held-out turn set here (see --only-sessions) and stop")
ap.add_argument("--only-sessions", nargs="*")
ap.add_argument("--exclude-sessions", nargs="*")
ap.add_argument("--exclude-turns", nargs="*")
ap.add_argument("--per-bin", type=int, default=32)
ap.add_argument("--per-trace", type=int, default=2)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.tok))
BINS = [("lt32k", 0, 32768), ("32_64k", 32768, 65536), ("64_128k", 65536, 131072), ("gt128k", 131072, 240000)]
rng = random.Random(a.seed)


def lit(x):
    if isinstance(x, str):
        for f in (json.loads, ast.literal_eval):
            try:
                return f(x)
            except Exception:
                pass
    return x


conv = collections.OrderedDict()
for line in open(a.data):
    r = json.loads(line)
    if "conversation_id" in r and r.get("_source") != "Workato":
        conv.setdefault(r["conversation_id"], []).append(r)
print("coding conversations", len(conv), "assistant turns", sum(m["role"] == "assistant" for ms in conv.values() for m in ms))


def to_msgs(rows):
    msgs, tools = [], None
    for m in rows:
        if m.get("system"):
            msgs.append({"role": "system", "content": m["system"]})
        if m.get("tools") and tools is None:
            tools = lit(m["tools"])
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif m["role"] == "assistant":
            d = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning_content"):
                d["reasoning_content"] = m["reasoning_content"]
            tc = lit(m.get("tool_calls")) if m.get("tool_calls") else None
            if tc:
                for t in tc:
                    f = t.get("function", {})
                    f["arguments"] = lit(f.get("arguments", {})) if isinstance(f.get("arguments"), str) else f.get("arguments", {})
                d["tool_calls"] = tc
            msgs.append(d)
        elif m["role"] == "tool":
            for t in lit(m.get("tool_results")) or []:
                msgs.append({"role": "tool", "tool_call_id": t.get("tool_call_id"), "content": t.get("content") if isinstance(t.get("content"), str) else json.dumps(t.get("content"))})
    return msgs, tools


def render(rows, thinking):
    msgs, tools = to_msgs(rows)
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=thinking, preserve_thinking=True)


cands = collections.defaultdict(list); total = collections.Counter(); excluded = 0
for cid, ms in conv.items():
    ms = sorted(ms, key=lambda m: int(m["turn"]))
    idx = [i for i, m in enumerate(ms) if m["role"] == "assistant" and i > 0]
    if not idx:
        continue
    deep = render(ms[: idx[-1]], False)
    ratio = len(tok(deep).input_ids) / max(len(deep), 1)
    cum = 0; pos = 0
    sizes = [len(json.dumps(m)) for m in ms]
    tot_sz = sum(sizes[: idx[-1]])
    for k in idx:  # characters of raw rows up to k scaled to the deepest render (cheap estimate)
        est = int(sum(sizes[:k]) / max(tot_sz, 1) * len(deep) * ratio)
        b = next((n for n, lo, hi in BINS if lo <= est < hi), None)
        if b is None:
            excluded += 1; continue
        total[b] += 1; cands[b].append((cid, k))
print("estimated bin counts (Coral: lt32k 2614, 32_64k 2272, 64_128k 5852, gt128k 5039; excluded 204):", dict(total), "excluded", excluded)
if a.hold_out:
    # Clean held-out set for the trained drafter: only traces listed in --only-sessions (never trained on), minus traces in
    # --exclude-sessions (checkpoint selection), minus turns already in --exclude-turns; --per-bin per bin, at most
    # --per-trace turns per trace per bin (limits trace clustering).
    S = lambda fs: {json.loads(l)[f] for p in fs or [] for l in open(p) for f in ["session"]}
    only, xs = S(a.only_sessions), S(a.exclude_sessions)
    xt = {json.loads(l)["turn_id"] for p in a.exclude_turns or [] for l in open(p)}
    with open(a.hold_out, "w") as fh:
        for name, lo, hi in BINS:
            pool = sorted(c for c in cands[name] if c[0] in only and c[0] not in xs); rng.shuffle(pool)
            n, per = 0, collections.Counter()
            for cid, k in pool:
                tid = hashlib.sha256(f"{cid}:{k}".encode()).hexdigest()[:12]
                if n >= a.per_bin or tid in xt or per[cid] >= a.per_trace:
                    continue
                rows = sorted(conv[cid], key=lambda m: int(m["turn"]))[:k]
                off = render(rows, False); nt = len(tok(off).input_ids)
                if not (lo <= nt < hi):
                    continue
                fh.write(json.dumps({"turn_id": tid, "session": cid, "k": k, "bin": name, "prompt_tokens": nt, "text_off": off, "text_on": render(rows, True)}) + "\n")
                n += 1; per[cid] += 1
            print(name, "held-out sampled", n, "from traces", len(per), flush=True)
    raise SystemExit
if a.sessions_out:
    # Concurrency workload: consecutive assistant turns of one trace in chronological order, short turns included, start
    # bins drawn with the MLCommons weights. Each user replays one trace; history comes from the trace.
    W = {"lt32k": 2614, "32_64k": 2272, "64_128k": 5852, "gt128k": 5039}
    quota = {b: max(1, round(a.n_sessions * w / sum(W.values()))) for b, w in W.items()}
    used, out = set(), []
    for name, lo, hi in BINS:
        pool = sorted(cands[name]); rng.shuffle(pool); n = 0
        for cid, k in pool:
            if n >= quota[name]:
                break
            if cid in used:
                continue
            ms = sorted(conv[cid], key=lambda m: int(m["turn"]))
            idx = [i for i, m in enumerate(ms) if m["role"] == "assistant" and i >= k][: a.session_turns]
            if len(idx) < a.session_turns:
                continue
            rows = [{"k": i, "text": render(ms[:i], False)} for i in idx]
            first, last = (len(tok(rows[j]["text"]).input_ids) for j in (0, -1))
            if not (lo <= first < hi and last < 240000):
                continue
            rows[0]["prompt_tokens"], rows[-1]["prompt_tokens"] = first, last
            used.add(cid); n += 1
            out.append({"session": cid, "bin": name, "first_tokens": first, "last_tokens": last, "turns": rows})
        print(name, "sessions", n, flush=True)
    rng.shuffle(out)
    with open(a.sessions_out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + chr(10))
    raise SystemExit
os.makedirs(a.out, exist_ok=True)
dev, head = open(f"{a.out}/turns_dev64.jsonl", "w"), open(f"{a.out}/turns_head128.jsonl", "w")
man = {"counts": dict(total), "excluded": excluded}
for name, lo, hi in BINS:
    pool = cands[name]; rng.shuffle(pool)
    n = 0
    for cid, k in pool:
        if n >= 48:
            break
        rows = sorted(conv[cid], key=lambda m: int(m["turn"]))[:k]
        off = render(rows, False); nt = len(tok(off).input_ids)
        if not (lo <= nt < hi):  # estimate landed in the wrong bin: skip
            continue
        rec = {"turn_id": hashlib.sha256(f"{cid}:{k}".encode()).hexdigest()[:12], "session": cid, "k": k, "bin": name, "prompt_tokens": nt, "text_off": off, "text_on": render(rows, True)}
        (dev if n < 16 else head).write(json.dumps(rec) + "\n"); n += 1
    man[name] = n; print(name, "sampled", n, flush=True)
json.dump(man, open(f"{a.out}/turns_manifest.json", "w"), indent=1)
