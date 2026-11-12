"""Drafter fine-tuning data, turns: every assistant turn (in order) of the official MLPerf Agentic coding traces that contain NO evaluation
turn (dev64, ind128): 24 traces, ~2,316 turns. Rendered exactly like build_official.py (Qwen3.6 chat template, tools, earlier reasoning
and tool calls preserved, add_generation_prompt), thinking off and on. Turns over ~238K tokens (estimated per trace) are skipped.
Workato option: --source Workato --max-traces N (independent agentic traces with reasoning and tool calls).
Usage: python build_train.py --data agentic.jsonl --tok <model> --exclude turns_dev64.jsonl turns_ind128.jsonl --out train_turns.jsonl"""
import argparse, ast, collections, hashlib, json, os
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--exclude", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", nargs="*", default=None, help="restrict to these trace ids (validation sets from held-out traces)")
ap.add_argument("--source", default="coding", choices=["coding", "Workato", "swe"], help="Workato: the 500 non-coding agentic traces (in no eval set)")
ap.add_argument("--swe", help="--source swe: swe_fetch.py output (OpenHands trajectories)")
ap.add_argument("--turns-per-trace", type=int, default=8, help="--source swe: assistant turns kept per trajectory, evenly spaced")
ap.add_argument("--max-traces", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.tok))


def lit(x):
    if isinstance(x, str):
        for f in (json.loads, ast.literal_eval):
            try:
                return f(x)
            except Exception:
                pass
    return x


def clean(x):   # parquet unions struct keys across rows: drop the None-valued ones
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if v is not None and v != [] and clean(v) != {}}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


if a.source == "swe":   # independent in-domain SWE-agent trajectories; repos overlapping any --exclude trace name are dropped again here
    names = {json.loads(l)["session"].lower() for f in a.exclude for l in open(f)}
    n = skipped = dropped = 0
    with open(a.out, "w") as fh:
        for line in open(a.swe):
            r = json.loads(line); rn = r["repo"].split("/")[-1].lower()
            if any(t == rn or t.startswith(rn + "-") for t in names):
                dropped += 1; continue
            tools = [{"type": "function", "function": clean(t["function"])} for t in (r.get("tools") or [])]
            msgs = []
            for m in r["messages"]:
                d = {"role": m["role"], "content": m.get("content") or ""}
                if m["role"] == "assistant" and m.get("tool_calls"):
                    d["tool_calls"] = [{"id": c.get("id"), "type": "function", "function": {"name": c["function"]["name"],
                                        "arguments": lit(c["function"].get("arguments") or "{}")}} for c in m["tool_calls"]]
                if m["role"] == "tool":
                    d["tool_call_id"] = m.get("tool_call_id"); d["name"] = m.get("name")
                msgs.append(d)
            idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
            if not idx:
                continue
            k_ = max(1, a.turns_per_trace)
            pick = sorted({idx[round(j * (len(idx) - 1) / max(1, k_ - 1))] for j in range(min(k_, len(idx)))})
            rend = lambda k, th: tok.apply_chat_template(msgs[:k], tools=tools or None, tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=th, preserve_thinking=True)
            deep = rend(pick[-1], False); ratio = len(tok(deep).input_ids) / max(len(deep), 1)
            for k in pick:
                off = rend(k, False)
                if len(off) * ratio > 238000:
                    skipped += 1; continue
                fh.write(json.dumps({"turn_id": hashlib.sha256(f"{r['session']}:{k}".encode()).hexdigest()[:12], "session": r["session"], "k": k,
                                     "repo": r["repo"], "est_tokens": int(len(off) * ratio), "text_off": off, "text_on": rend(k, True)}) + chr(10))
                n += 1
    print(f"swe trajectories dropped for overlap {dropped}, turns written {n}, skipped (too long) {skipped}")
    raise SystemExit

conv = collections.OrderedDict()
for line in open(a.data):
    r = json.loads(line)
    if "conversation_id" in r and (r.get("_source") == "Workato") == (a.source == "Workato"):
        conv.setdefault(r["conversation_id"], []).append(r)
held = set()
for f in a.exclude:
    held |= {json.loads(l)["session"] for l in open(f)}


def to_msgs(rows):   # identical to build_official.py
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


import random
keep = list(conv)
if a.max_traces:
    random.Random(a.seed).shuffle(keep); keep = set(keep[: a.max_traces])
n = skipped = 0
with open(a.out, "w") as fh:
    for cid, ms in conv.items():
        if cid not in keep:
            continue
        if (a.only is None and cid in held) or (a.only is not None and cid not in a.only):
            continue
        ms = sorted(ms, key=lambda m: int(m["turn"]))
        idx = [i for i, m in enumerate(ms) if m["role"] == "assistant" and i > 0]
        if not idx:
            continue
        deep = render(ms[: idx[-1]], False)
        ratio = len(tok(deep).input_ids) / max(len(deep), 1)
        for k in idx:
            off = render(ms[:k], False)
            if len(off) * ratio > 238000:
                skipped += 1; continue
            fh.write(json.dumps({"turn_id": hashlib.sha256(f"{cid}:{k}".encode()).hexdigest()[:12], "session": cid, "k": k,
                                 "est_tokens": int(len(off) * ratio), "text_off": off, "text_on": render(ms[:k], True)}) + "\n")
            n += 1
print(f"traces held out {len(held & set(conv))}, training traces {len([c for c in conv if c not in held])}, turns written {n}, skipped (too long) {skipped}")
