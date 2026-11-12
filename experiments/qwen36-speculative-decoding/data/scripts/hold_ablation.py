"""Blog data: every turns_hold128 server experiment, paired against its same-boot control. Writes hold128_experiments.csv.
Speed = MLCommons-bin-reweighted median decode tok/s (turns >= 32 output tokens); CIs = trace-clustered bootstrap (2,000)."""
import collections, json, math, os, random, statistics as st, sys

D = sys.argv[1]
W = {"lt32k": 2614, "32_64k": 2272, "64_128k": 5852, "gt128k": 5039}
turns = {t["turn_id"]: t for t in map(json.loads, open(os.path.join(D, "turns_hold128.jsonl")))}
rng = random.Random(0)
# (session, arm, control, description, note)
EXP = [
    ("s13", "a16c", "stk", "Block 16, Run 2 drafter (no AVL)", "vs stock MTP-4, same boot"),
    ("s13", "a16v", "stk", "Block 16 + AVL, Run 2", "vs stock MTP-4, same boot"),
    ("s13", "a16v", "a16c", "Block 16 + AVL vs block 16", "AVL ablation"),
    ("s13", "a16v", "b8on", "Block 16 + AVL vs block 8 BEST3 (Coral drafter)", "thinking on only"),
    ("s13", "a16c", "b8on", "Block 16 vs block 8 BEST3", "thinking on only"),
    ("s14", "avl_m4", "avlc", "AVL min L 4 vs default (min L 2)", "AVL tuning, off only"),
    ("s14", "avl_l5", "avlc", "AVL lambda 0.5 vs default 0.75", "AVL tuning, off only"),
    ("s14", "avlr5", "avlc", "Run 5c drafter vs Run 2 (both block 16 + AVL)", "training lever, off only"),
    ("s15", "a16a", "a16v", "L-skip (block 16 + AVL)", "fixed hook, tail run"),
    ("s15", "a24v", "a16v", "Block 24 + Run 6 + AVL (no L-skip, no fixes)", ""),
    ("s15", "a24l", "a16v", "Block 24 + Run 6 + AVL + L-skip", "thinking on partial (94 turns)"),
    ("s16", "a16a", "a16v", "L-skip (block 16 + AVL)", "thinking-on tail contaminated by a concurrent diverge.py: use off and medians only"),
    ("s16", "a24l", "a16v", "Block 24 + Run 6 + AVL + L-skip + s5 M<=24 + Bole BV8", "best stack"),
    ("s17", "a24l", "a16v", "Block 24 + Run 6 + AVL + L-skip + s5 M<=24 + Bole BV8", "best stack, reproduction"),
]


def ld(s, arm, mode):
    f = os.path.join(D, "raw", s, f"rep_{arm}_{mode}_hold.jsonl")
    if not os.path.exists(f):
        return {}
    return {d["turn_id"]: d for d in map(json.loads, open(f)) if "turn_id" in d}


ok = lambda r: r.get("decode_tok_s") and (r.get("out_tokens") or 0) >= 32 and r.get("accept") not in (None, "None")


def wmedian(vb):
    cnt = collections.Counter(b for _, b in vb)
    pts = sorted((v, W[b] / cnt[b]) for v, b in vb); tot = sum(w for _, w in pts); acc = 0.0
    for v, w in pts:
        acc += w
        if acc >= tot / 2:
            return v


def boot(ts, fn, n=2000):
    g = collections.defaultdict(list)
    for t in ts:
        g[turns[t]["session"]].append(t)
    cl = list(g.values()); vals = []
    for _ in range(n):
        v = fn([t for c in (rng.choice(cl) for _ in cl) for t in c])
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


rows = []
for s, a, c, desc, note in EXP:
    for m in ("off", "on"):
        A, B = ld(s, a, m), ld(s, c, m)
        ts = [t for t in A if t in B and ok(A[t]) and ok(B[t])]
        if len(ts) < 20:
            continue
        sp = lambda X, q: wmedian([(float(X[t]["decode_tok_s"]), X[t]["bin"]) for t in q])
        ratio = lambda q: sp(A, q) / sp(B, q)
        lo, hi = boot(ts, ratio)
        r = [float(A[t]["decode_tok_s"]) / float(B[t]["decode_tok_s"]) for t in ts]
        acc = lambda X: st.mean(float(X[t]["accept"]) for t in ts)
        stp = lambda X: st.median(float(X[t]["step_ms"]) for t in ts)
        rows.append([s, a, c, m, desc, len(ts), f"{sp(A, ts):.1f}", f"{sp(B, ts):.1f}", f"{ratio(ts):.4f}", f"{lo:.4f}", f"{hi:.4f}",
                     f"{st.median(r):.4f}", f"{math.exp(st.mean(map(math.log, r))):.4f}", sum(x > 1 for x in r),
                     f"{acc(A):.3f}", f"{acc(B):.3f}", f"{stp(A):.3f}", f"{stp(B):.3f}", note])
hdr = ["session", "arm", "control", "mode", "description", "n_paired", "arm_tok_s_rw", "control_tok_s_rw", "ratio_rw", "ratio_ci_lo", "ratio_ci_hi",
       "per_turn_median_ratio", "per_turn_geo_ratio", "turns_faster", "arm_accept_per_round", "control_accept_per_round", "arm_step_ms", "control_step_ms", "note"]
with open(os.path.join(D, "hold128_experiments.csv"), "w", encoding="utf-8") as fh:
    fh.write(",".join(hdr) + "\n")
    for r in rows:
        fh.write(",".join(f'"{x}"' if isinstance(x, str) and ("," in x or " " in x) else str(x) for x in r) + "\n")
for r in rows:
    print(r[0], r[1], "vs", r[2], r[3], "n", r[5], r[6], "vs", r[7], "ratio", r[8], f"({r[9]}-{r[10]})", "med", r[11], "geo", r[12])
