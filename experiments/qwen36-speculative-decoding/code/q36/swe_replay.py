"""Phase 5 batch-1 replay (Coral Bricks protocol): one request at a time, full history, greedy, max_tokens 1024,
natural stop, thinking off|on. Per turn: prompt tokens, output tokens, TTFT, decode tok/s = (n-1)/(t_last-t_first)
(their metric: median decode speed), mean accept length, step time = accept/decode_tok_s. Summary: median decode tok/s
(all turns), per-bin medians, the MLCommons-bin-reweighted median, bootstrap 95% CI, mean accept, step time.
Resumable: turns already in --out are skipped.
Usage: python swe_replay.py --turns turns_dev64.jsonl --arm mtp4 --thinking off --out rep_mtp4_off.jsonl"""
import argparse, json, os, random, statistics as st, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--turns", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--thinking", choices=["off", "on"], default="off")
ap.add_argument("--out", required=True)
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
URL = "http://127.0.0.1:30000/generate"
W = {"lt32k": 2614, "32_64k": 2272, "64_128k": 5852, "gt128k": 5039}
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["turn_id"] for l in open(a.out) if l.strip() and '"turn_id"' in l}
turns = [json.loads(l) for l in open(a.turns)]
if a.limit:
    turns = turns[: a.limit]


def one(t):
    body = {"text": t["text_" + a.thinking], "stream": True, "sampling_params": {"temperature": 0, "max_new_tokens": a.max_new}}
    t0 = time.time(); first = last = None; n = 0; meta = {}; txt = ""
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:]); meta = d["meta_info"]; now = time.time()
            k = meta.get("completion_tokens", 0)
            if first is None and k > 0:
                first = now
            if k > n:
                last, n = now, k
            txt = d.get("text", txt)
    dec = (n - 1) / (last - first) if n > 1 and last > first else None
    acc = meta.get("spec_accept_length")
    return {"turn_id": t["turn_id"], "bin": t["bin"], "arm": a.arm, "thinking": a.thinking, "prompt_tokens": meta.get("prompt_tokens"),
            "out_tokens": n, "ttft_s": round((first or time.time()) - t0, 3), "decode_tok_s": dec and round(dec, 2), "accept": acc,
            "step_ms": round(1000 * acc / dec, 3) if acc and dec else None, "finish": (meta.get("finish_reason") or {}).get("type"), "text_head": txt[:600], "text": txt, "ids": meta.get("output_ids")}


with open(a.out, "a") as fh:
    for i, t in enumerate(turns):
        if t["turn_id"] in done:
            continue
        try:
            r = one(t)
        except Exception as e:  # recorded, not hidden
            r = {"turn_id": t["turn_id"], "bin": t["bin"], "arm": a.arm, "thinking": a.thinking, "error": str(e)[:200]}
        fh.write(json.dumps(r) + "\n"); fh.flush()
        print(i, r.get("bin"), r.get("prompt_tokens"), r.get("decode_tok_s"), r.get("accept"), r.get("error", ""), flush=True)

rows = [json.loads(l) for l in open(a.out) if l.strip() and '"turn_id"' in l]
ok_all = [r for r in rows if r.get("decode_tok_s")]
ok = [r for r in ok_all if (r.get("out_tokens") or 0) >= 32]  # decode speed over < 32 tokens is noise-dominated (reported separately)


def wmedian(rs):
    """Median of per-turn decode tok/s with each turn weighted by its bin's MLCommons share / bin sample count."""
    cnt = {b: sum(1 for r in rs if r["bin"] == b) for b in W}
    pts = sorted((r["decode_tok_s"], W[r["bin"]] / cnt[r["bin"]]) for r in rs if cnt.get(r["bin"]))
    tot = sum(w for _, w in pts); acc = 0.0
    for v, w in pts:
        acc += w
        if acc >= tot / 2:
            return v
    return None


boot = []
rnd = random.Random(0)
for _ in range(1000):
    s = [rnd.choice(ok) for _ in ok]
    m = wmedian(s)
    if m:
        boot.append(m)
boot.sort()
summ = {"phase": "summary", "all_turns_reweighted_median": ok_all and wmedian(ok_all), "all_turns_median": ok_all and st.median(r["decode_tok_s"] for r in ok_all), "arm": a.arm, "thinking": a.thinking, "turns": len(rows), "ok": len(ok), "short_excluded": len(ok_all) - len(ok), "errors": len(rows) - len(ok_all), "median_out_tokens": ok and st.median(r["out_tokens"] for r in ok),
        "median_decode_tok_s": ok and st.median(r["decode_tok_s"] for r in ok), "mlc_reweighted_median": ok and wmedian(ok),
        "ci95": boot and [boot[25], boot[975]], "mean_accept": ok and round(st.mean(r["accept"] or 0 for r in ok), 3),
        "median_step_ms": ok and st.median(r["step_ms"] for r in ok if r.get("step_ms")) if any(r.get("step_ms") for r in ok) else None,
        "per_bin_median": {b: st.median([r["decode_tok_s"] for r in ok if r["bin"] == b]) for b in W if any(r["bin"] == b for r in ok)}}
json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
print("SUMMARY", json.dumps(summ))
