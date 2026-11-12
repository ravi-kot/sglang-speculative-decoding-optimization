"""Output divergence vs a target-only greedy reference (no speculation), one thinking mode.
Per arm: identical outputs, first divergent token (texts re-tokenized; round-trip checked), malformed tool calls.
Margins: for up to --margins diverged turns (spread over bins), ask the running reference server for the target's
top-10 log-probs at the first divergent position (prompt + common prefix, one prefill) and report
logp(ref token) - logp(arm token). Small margins = near-ties flipped by rounding in a different verify shape (M=1 decode
vs M=8..16 verify); the prefill path is itself a third shape, so its top-1 is reported as well.
Usage: diverge.py --turns T --mode off --ref rep_ref_off.jsonl --arms rep_a_off.jsonl ... --out div_off.json"""
import argparse, collections, json, re, statistics as st, urllib.request
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--turns", required=True); ap.add_argument("--mode", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--tok", default="/home/ubuntu/lab/models/Qwen3.6-35B-A3B-FP8")
ap.add_argument("--margins", type=int, default=48); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tok)
ld = lambda f: {d["turn_id"]: d for d in map(json.loads, open(f)) if "turn_id" in d and "text" in d}
turns = {t["turn_id"]: t for t in map(json.loads, open(a.turns))}
ref = ld(a.ref)
enc = lambda s: tok(s, add_special_tokens=False).input_ids


def bad_calls(t):  # same rule as s4_report.py
    calls = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", t, flags=re.S)
    return len(calls), sum(1 for c in calls if not re.match(r"\s*<function=[^>\s]+>", c) or c.count("<function=") != 1)


rids = {k: enc(v["text"]) for k, v in ref.items()}
rt_bad = sum(tok.decode(rids[k]) != ref[k]["text"] for k in ref)
res = {"mode": a.mode, "ref": a.ref, "ref_turns": len(ref), "ref_roundtrip_mismatch": rt_bad,
       "ref_tool_calls": [sum(x) for x in zip(*(bad_calls(v["text"]) for v in ref.values()))], "arms": {}}
div = collections.defaultdict(dict)   # turn -> {arm: (d, ref_tok, arm_tok)}
for f in a.arms:
    x = ld(f); name = f.split("rep_")[-1].replace(".jsonl", "")
    com = [t for t in x if t in ref]; same = 0; ds = []; fr = []
    for t in com:
        if x[t]["text"] == ref[t]["text"]:
            same += 1; continue
        ri, ai = rids[t], enc(x[t]["text"])
        d = next((i for i, (p, q) in enumerate(zip(ri, ai)) if p != q), min(len(ri), len(ai)))
        ds.append(d); fr.append(d / max(len(ri), 1))
        if d < min(len(ri), len(ai)):
            div[t][name] = (d, ri[d], ai[d])
    calls = [sum(z) for z in zip(*(bad_calls(x[t]["text"]) for t in com))] or [0, 0]
    res["arms"][name] = {"turns": len(com), "identical": same, "first_div_median": ds and st.median(ds),
                         "first_div_frac_median": fr and round(st.median(fr), 3), "tool_calls": calls[0], "malformed": calls[1]}
    print(f"DIV {name}: identical {same}/{len(com)}; first divergent token median {ds and st.median(ds)} "
          f"({fr and round(st.median(fr), 3)} of output); tool calls {calls[0]}, malformed {calls[1]} (ref {res['ref_tool_calls']})", flush=True)

# Margins at the first divergence: spread the budget evenly over context bins.
by_bin = collections.defaultdict(list)
for t in div:
    by_bin[turns[t]["bin"]].append(t)
pick = [t for i in range(a.margins) for b in sorted(by_bin) for t in by_bin[b][i:i + 1]][: a.margins]
rows = []
for t in pick:
    p = enc(turns[t]["text_" + a.mode])
    for d in sorted({v[0] for v in div[t].values()}):
        body = {"input_ids": p + rids[t][:d], "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True, "top_logprobs_num": 10, "logprob_start_len": -1}
        m = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/generate", json.dumps(body).encode(),
                                                                     {"Content-Type": "application/json"}), timeout=3600).read())["meta_info"]
        top = {int(e[1]): float(e[0]) for e in m["output_top_logprobs"][0]}
        top1 = max(top, key=top.get)
        for arm, (dd, rtok, atok) in div[t].items():
            if dd != d:
                continue
            mg = top[rtok] - top[atok] if rtok in top and atok in top else None
            rows.append({"turn": t, "bin": turns[t]["bin"], "arm": arm, "d": d, "margin": mg, "ref_lp": top.get(rtok),
                         "prefill_top1_is_ref": top1 == rtok, "prefill_top1_is_arm": top1 == atok})
res["margins"] = rows
for arm in res["arms"]:
    r = [x for x in rows if x["arm"] == arm]
    mg = [abs(x["margin"]) for x in r if x["margin"] is not None]
    if r:
        s = {"n": len(r), "in_top10": len(mg), "abs_margin_median": mg and round(st.median(mg), 4),
             "lt_0.05": sum(m < 0.05 for m in mg), "lt_0.2": sum(m < 0.2 for m in mg),
             "prefill_top1_ref": sum(x["prefill_top1_is_ref"] for x in r), "prefill_top1_arm": sum(x["prefill_top1_is_arm"] for x in r)}
        res["arms"][arm]["margin"] = s
        print(f"MARGIN {arm}: {s}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
