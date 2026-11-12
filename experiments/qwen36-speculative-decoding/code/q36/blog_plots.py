"""Blog figures and tables for the block-size sweep (q36_ft2, RTX PRO 6000, BS1 greedy, max_tokens 1024).
Inputs in <dir>: rep_{stk,sw8,sw10,sw12,sw14,sw16,ref}_{off,on}_hold.jsonl, turns_hold128.jsonl, abl.txt, crit_n8.txt,
crit_n16.txt, div_{off,on}.json. Speed = MLCommons-bin-reweighted median decode tok/s (as swe_replay / blog_stats), turns with
< 32 output tokens excluded. CIs: trace-clustered bootstrap (resample traces, 2,000 replicates), because held-out turns share
traces. Paired ratios use the same turns in both arms.
Usage: python blog_plots.py <dir>   -> <dir>/figs/*.png|svg and <dir>/figs/tables.md"""
import collections, json, math, os, random, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]; F = os.path.join(D, "figs"); os.makedirs(F, exist_ok=True)
W = {"lt32k": 2614, "32_64k": 2272, "64_128k": 5852, "gt128k": 5039}
BINS = list(W); BL = {"lt32k": "<32K", "32_64k": "32–64K", "64_128k": "64–128K", "gt128k": ">128K"}
BLOCKS = [8, 10, 12, 14, 16]; MODES = ["off", "on"]; ML = {"off": "Thinking off", "on": "Thinking on"}
# Reference palette (dataviz skill, light mode): categorical slots 1-3, ordinal blue ramp, ink tokens.
C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "bg": "#fcfcfb"}
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({"figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "savefig.facecolor": C["bg"], "font.size": 10,
                     "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlecolor": C["ink"], "legend.frameon": False, "font.family": "DejaVu Sans"})
BENCH = len(sys.argv) > 2 and sys.argv[2] == "bench"   # 512-turn independent benchmark: rep_{bstk,bb8,bb16,bbest}_{off,on}_bench.jsonl
SUF = "_bench" if BENCH else "_hold"
turns = {t["turn_id"]: t for t in map(json.loads, open(os.path.join(D, "turns_bench.jsonl" if BENCH else "turns_hold128.jsonl")))}
rng = random.Random(0)


def ld(arm, mode):
    f = os.path.join(D, f"rep_{arm}_{mode}{SUF}.jsonl")
    if not os.path.exists(f):
        return {}
    return {d["turn_id"]: d for d in map(json.loads, open(f)) if "turn_id" in d}


def ok(r):
    return r.get("decode_tok_s") and (r.get("out_tokens") or 0) >= 32


def wmedian(vb):
    cnt = collections.Counter(b for _, b in vb)
    pts = sorted((v, W[b] / cnt[b]) for v, b in vb)
    tot = sum(w for _, w in pts); acc = 0.0
    for v, w in pts:
        acc += w
        if acc >= tot / 2:
            return v


def clusters(ts):
    g = collections.defaultdict(list)
    for t in ts:
        g[turns[t]["session"]].append(t)
    return list(g.values())


def boot(ts, fn, n=2000):
    cl = clusters(ts); vals = []
    for _ in range(n):
        s = [t for c in (rng.choice(cl) for _ in cl) for t in c]
        v = fn(s)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


ARMS = ["bstk", "bb8", "bb16", "bbest"] if BENCH else ["stk", "ref"] + [f"sw{b}" for b in BLOCKS]
R = {(a, m): ld(a, m) for a in ARMS for m in MODES}
S = {}   # (arm, mode) -> stats
for (a, m), x in R.items():
    ts = [t for t in x if ok(x[t])]
    if not ts:
        continue
    f = lambda s, x=x: wmedian([(x[t]["decode_tok_s"], x[t]["bin"]) for t in s])
    acc = [x[t]["accept"] for t in ts if x[t].get("accept")]
    S[a, m] = {"n": len(ts), "tok_s": f(ts), "ci": boot(ts, f), "accept": acc and st.mean(acc),
               "step": {b: st.median([x[t]["step_ms"] for t in ts if x[t]["bin"] == b and x[t].get("step_ms")] or [float("nan")]) for b in BINS},
               "step_all": st.median([x[t]["step_ms"] for t in ts if x[t].get("step_ms")] or [float("nan")])}


def paired(a, b, m):   # ratio of reweighted medians, a over b, same turns, clustered CI; plus per-turn geometric mean
    A, B = R[a, m], R[b, m]; ts = [t for t in A if t in B and ok(A[t]) and ok(B[t])]
    if not ts:
        return None
    f = lambda s: wmedian([(A[t]["decode_tok_s"], A[t]["bin"]) for t in s]) / wmedian([(B[t]["decode_tok_s"], B[t]["bin"]) for t in s])
    g = math.exp(st.mean(math.log(A[t]["decode_tok_s"] / B[t]["decode_tok_s"]) for t in ts))
    return {"n": len(ts), "ratio": f(ts), "ci": boot(ts, f), "geo": g}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


if BENCH:   # data only: tables (md), bench.csv / bench.json, two figures
    LAB = {"bstk": "Stock SGLang MTP-4", "bb8": "Block 8 (BEST3, Coral drafter)", "bb16": "Block 16 + AVL (Run 2)",
           "bbest": "Block 24 + AVL + L-skip + exact fixes (Run 6)"}
    out, rows = {}, []
    for a in ARMS:
        for m in MODES:
            if (a, m) not in S:
                continue
            s = S[a, m]; x = R[a, m]; ts = [t for t in x if ok(x[t])]
            perbin = {b: st.median([x[t]["decode_tok_s"] for t in ts if x[t]["bin"] == b] or [float("nan")]) for b in BINS}
            prev = next((p for p in reversed(ARMS[: ARMS.index(a)]) if (p, m) in S), None)
            ps = paired(a, "bstk", m) if a != "bstk" and ("bstk", m) in S else None
            pp = paired(a, prev, m) if prev else None
            d = {"arm": a, "label": LAB[a], "mode": m, "n": s["n"], "tok_s": s["tok_s"], "ci_lo": s["ci"][0], "ci_hi": s["ci"][1],
                 "accept_per_round": s["accept"], "step_ms_median": s["step_all"]}
            d.update({f"tok_s_{b}": perbin[b] for b in BINS}); d.update({f"step_ms_{b}": s["step"][b] for b in BINS})
            for tag, p in (("vs_stock", ps), ("vs_prev", pp)):
                d.update({tag: p and p["ratio"], tag + "_lo": p and p["ci"][0], tag + "_hi": p and p["ci"][1], tag + "_geo": p and p["geo"], tag + "_n": p and p["n"]})
            d["prev_arm"] = prev
            rows.append(d); out[f"{a}_{m}"] = d
    json.dump(out, open(os.path.join(F, "bench.json"), "w"), indent=1)
    NL = chr(10)
    with open(os.path.join(F, "bench.csv"), "w") as fh:
        ks = list(rows[0]); fh.write(",".join(ks) + NL)
        for d in rows:
            fh.write(",".join("" if d[k] is None else (f"{d[k]:.4f}" if isinstance(d[k], float) else str(d[k])) for k in ks) + NL)

    def rat(d, tag, digits):
        if d[tag] is None:
            return "–"
        return f"{d[tag]:.{digits}f} ({d[tag + '_lo']:.{digits}f}–{d[tag + '_hi']:.{digits}f})"
    with open(os.path.join(F, "bench_tables.md"), "w", encoding="utf-8") as fh:
        hdr = ["Arm", "Mode", "n", "tok/s (95% CI)", "vs stock (95% CI)", "geo vs stock", "vs previous arm (95% CI)", "accept/round", "step ms"] + [f"tok/s {BL[b]}" for b in BINS]
        fh.write("| " + " | ".join(hdr) + " |" + NL + "|" + "---|" * len(hdr) + NL)
        for d in rows:
            cells = [d["label"], d["mode"], str(d["n"]), f"{d['tok_s']:.0f} ({d['ci_lo']:.0f}–{d['ci_hi']:.0f})", rat(d, "vs_stock", 2),
                     "–" if d["vs_stock_geo"] is None else f"{d['vs_stock_geo']:.2f}", rat(d, "vs_prev", 3),
                     f"{d['accept_per_round'] or 0:.2f}", f"{d['step_ms_median']:.2f}"] + [f"{d[f'tok_s_{b}']:.0f}" for b in BINS]
            fh.write("| " + " | ".join(cells) + " |" + NL)
    # Figure: tok/s per arm, small multiples per mode, 95% CI.
    names = {"bstk": "Stock" + NL + "MTP-4", "bb8": "Block 8", "bb16": "Block 16" + NL + "+ AVL", "bbest": "Block 24" + NL + "+ all"}
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, m in zip(axs, MODES):
        ar = [a for a in ARMS if (a, m) in S]; y = [S[a, m]["tok_s"] for a in ar]
        lo = [y[i] - S[a, m]["ci"][0] for i, a in enumerate(ar)]; hi = [S[a, m]["ci"][1] - y[i] for i, a in enumerate(ar)]
        cols = [C["gray"] if a == "bstk" else RAMP[min(ARMS.index(a), len(RAMP) - 1)] for a in ar]
        ax.bar(range(len(ar)), y, color=cols, width=0.6, edgecolor=C["bg"], linewidth=2)
        ax.errorbar(range(len(ar)), y, yerr=[lo, hi], fmt="none", ecolor=C["ink2"], capsize=3, lw=1)
        for i, a in enumerate(ar):
            vs = out[f"{a}_{m}"]["vs_stock"]
            ax.text(i, S[a, m]["ci"][1] + 15, f"{y[i]:.0f}" + (f"{NL}{vs:.2f}x" if vs else ""), ha="center", va="bottom", fontsize=8.5, color=C["ink"])
        ax.set_xticks(range(len(ar))); ax.set_xticklabels([names[a] for a in ar], fontsize=8.5); ax.set_title(ML[m])
    axs[0].set_ylabel("Decode tok/s (reweighted median)"); axs[0].set_ylim(0, max(S[k]["ci"][1] for k in S) * 1.25)
    fig.suptitle("512 independent held-out turns, BS1 greedy, RTX PRO 6000 (95% CI, trace-clustered)", fontsize=10, color=C["ink2"], y=0.995)
    save(fig, "bench_throughput")
    # Figure: paired per-turn speedup of the best stack over stock by context bin.
    if all(("bbest", m) in S and ("bstk", m) in S for m in MODES):
        fig, ax = plt.subplots(figsize=(5.6, 3.6)); wdt = 0.36
        for j, (m, c) in enumerate(zip(MODES, (C["s1"], C["s2"]))):
            A, B = R["bbest", m], R["bstk", m]; v = []
            for b in BINS:
                ts = [t for t in A if t in B and ok(A[t]) and ok(B[t]) and A[t]["bin"] == b]
                v.append(st.median(A[t]["decode_tok_s"] / B[t]["decode_tok_s"] for t in ts) if ts else float("nan"))
            xs = [i + (j - 0.5) * wdt for i in range(len(BINS))]
            ax.bar(xs, v, width=wdt, color=c, label=ML[m], edgecolor=C["bg"], linewidth=2)
            for xx, vv in zip(xs, v):
                ax.text(xx, vv + 0.05, f"{vv:.2f}", ha="center", fontsize=8, color=C["ink"])
        ax.axhline(1, color=C["gray"], lw=1, ls="--"); ax.set_xticks(range(len(BINS))); ax.set_xticklabels([BL[b] for b in BINS])
        ax.set_ylim(0, 3.9); ax.set_ylabel("Per-turn speedup vs stock (median)"); ax.set_xlabel("Context length"); ax.legend(loc="upper left", fontsize=8.5, ncol=2)
        ax.set_title("Best stack vs stock SGLang MTP-4 by context")
        save(fig, "bench_speedup_by_context")
    sys.stdout.reconfigure(encoding="utf-8"); print(open(os.path.join(F, "bench_tables.md"), encoding="utf-8").read()); sys.exit(0)


# 1. Throughput by block size, small multiples per mode, stock MTP-4 as a reference line.
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, m in zip(axs, MODES):
    bs = [b for b in BLOCKS if (f"sw{b}", m) in S]
    y = [S[f"sw{b}", m]["tok_s"] for b in bs]
    lo = [y[i] - S[f"sw{b}", m]["ci"][0] for i, b in enumerate(bs)]; hi = [S[f"sw{b}", m]["ci"][1] - y[i] for i, b in enumerate(bs)]
    ax.errorbar(bs, y, yerr=[lo, hi], color=C["s1"], lw=2, marker="o", ms=7, capsize=3, mec=C["bg"], mew=1.5)
    for b, v in zip(bs, y):
        ax.annotate(f"{v:.0f}", (b, v), textcoords="offset points", xytext=(7, 5), ha="left", color=C["ink"], fontsize=9)
    if ("stk", m) in S:
        v = S["stk", m]["tok_s"]; ax.axhline(v, color=C["gray"], lw=1.5, ls="--")
        ax.text(BLOCKS[-1] + 0.3, v, f"stock MTP-4\n{v:.0f}", va="center", ha="left", color=C["ink2"], fontsize=8.5)
    ax.set_title(ML[m]); ax.set_xticks(BLOCKS); ax.set_xlabel("Block size (draft tokens verified per round)")
    ax.set_xlim(7, 18.2)
axs[0].set_ylabel("Decode tok/s (reweighted median)"); axs[0].set_ylim(bottom=0)
fig.suptitle("Decode throughput vs block size, 128 held-out turns (95% CI, trace-clustered)", fontsize=10, color=C["ink2"], y=0.995)
save(fig, "throughput_by_block")

# 2. Accepted tokens per round (server), both modes.
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for m, c in zip(MODES, (C["s1"], C["s2"])):
    bs = [b for b in BLOCKS if (f"sw{b}", m) in S]
    y = [S[f"sw{b}", m]["accept"] for b in bs]
    ax.plot(bs, y, color=c, lw=2, marker="o", ms=7, mec=C["bg"], mew=1.5, label=ML[m])
    if bs:
        ax.annotate(ML[m], (bs[-1], y[-1]), textcoords="offset points", xytext=(8, 0), va="center", color=C["ink"], fontsize=9)
ax.plot(BLOCKS, BLOCKS, color=C["grid"], lw=1.2, ls=":", zorder=0); ax.text(14.6, 15.2, "ceiling (= block)", color=C["ink2"], fontsize=8)
ax.set_xticks(BLOCKS); ax.set_xlim(7, 19); ax.set_ylim(0, 17)
ax.set_xlabel("Block size"); ax.set_ylabel("Accepted tokens per round"); ax.legend(loc="upper left")
ax.set_title("Tokens accepted per verification round")
save(fig, "accept_by_block")

# 3. Drafter ablation (offline chain eval on held-out validation traces, server-like rounds).
CELLS = {("Coral", 8): (6.58, 5.41), ("Coral", 16): (8.39, 6.39), ("Run 1", 16): (9.206, 6.291),   # f2_base, run-1 best step 1500
         ("Run 2", 16): (9.53, 6.66), ("Run 2", 12): (8.254, 6.262), ("Run 2", 10): (7.535, 5.902)}  # run-2 best step 1000, f2_blocks_t
name = {"coral": "Coral", "run1": "Run 1", "run2": "Run 2"}; cur = None
for l in open(os.path.join(D, "abl.txt")):
    mm = re.match(r"ABL (\w+) block (\d+)", l)
    if mm:
        cur = (name[mm.group(1)], int(mm.group(2))); continue
    mm = re.search(r"'chain_off': ([0-9.]+), 'chain_on': ([0-9.]+)", l)
    if mm and cur:
        CELLS[cur] = (float(mm.group(1)), float(mm.group(2))); cur = None
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, i, m in zip(axs, (0, 1), MODES):
    for d, c in zip(("Coral", "Run 1", "Run 2"), (C["s1"], C["s2"], C["s3"])):
        bs = [b for b in BLOCKS if (d, b) in CELLS]
        ax.plot(bs, [CELLS[d, b][i] for b in bs], color=c, lw=2, marker="o", ms=7, mec=C["bg"], mew=1.5, label=d)
        if bs:
            ax.annotate(d, (bs[-1], CELLS[d, bs[-1]][i]), textcoords="offset points", xytext=(8, 0), va="center", fontsize=8.5, color=C["ink"])
    ax.set_title(ML[m]); ax.set_xticks(BLOCKS); ax.set_xlim(7, 18.5); ax.set_xlabel("Block size")
axs[0].set_ylabel("Tokens per round (offline chain eval)"); axs[0].legend(loc="upper left")
fig.suptitle("Drafter ablation: Coral's released drafter vs our fine-tunes (Run 1: no teacher KL; Run 2: teacher KL)", fontsize=10, color=C["ink2"], y=0.995)
save(fig, "drafter_ablation")

# 4. Verification step latency by block and context bin (both modes pooled per bin: median of the two modes' medians).
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for b, c in zip(BINS, RAMP):
    bs = [k for k in BLOCKS if all((f"sw{k}", m) in S for m in MODES)]
    y = [st.mean(S[f"sw{k}", m]["step"][b] for m in MODES) for k in bs]
    ax.plot(bs, y, color=c, lw=2, marker="o", ms=7, mec=C["bg"], mew=1.5, label=BL[b])
    if bs:
        ax.annotate(BL[b], (bs[-1], y[-1]), textcoords="offset points", xytext=(8, 0), va="center", fontsize=8.5, color=C["ink"])
ax.set_xticks(BLOCKS); ax.set_xlim(7, 18.5); ax.set_xlabel("Block size"); ax.set_ylabel("Step time per round (ms)")
ax.legend(title="Context", loc="upper left", fontsize=8.5, title_fontsize=8.5); ax.set_title("Draft + verify step latency by context length")
save(fig, "step_by_block_context")


# 5. Where the extra block-16 time goes (Nsight Systems, 40 steps per regime).
def crit(f):
    out, k = {}, None
    for l in open(os.path.join(D, f)):
        mm = re.match(r"== \w+ regime (\d+)", l)
        if mm:
            k = int(mm.group(1)); out[k] = {}; continue
        mm = re.match(r"steps \d+, attributed ([0-9.]+) ms/step", l)
        if mm:
            out[k]["_total"] = float(mm.group(1)); continue
        mm = re.match(r"\s+([0-9.]+) ms\s+[0-9.]+%\s+(\S+)", l)
        if mm and k:
            out[k][mm.group(2)] = float(mm.group(1))
    return out


MOE = ("fused_moe_kernel", "moe_align_block_size_kernel", "act_and_mul_kernel", "_moe_sum_reduce_kernel", "count_and_sort_expert_tokens_ker", "_router_triton_kernel")
ATT = ("BatchPrefillWithPagedKVCacheKern", "PersistentVariableLengthMergeSta")
CTX = {1: "18K", 2: "101K", 3: "165K"}   # profiled regimes (bs1_profile.py), context tokens
c8, c16 = crit("crit_n8.txt"), crit("crit_n16.txt")
rows = []
for k in sorted(set(c8) & set(c16)):
    part = lambda c, ks: sum(c[k].get(n, 0) for n in ks)
    dm, da = part(c16, MOE) - part(c8, MOE), part(c16, ATT) - part(c8, ATT)
    dt = c16[k]["_total"] - c8[k]["_total"]
    rows.append((CTX.get(k, str(k)), c8[k]["_total"], c16[k]["_total"], dm, da, dt - dm - da))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
x = range(len(rows)); bottom = [0.0] * len(rows)
for j, (lab, c) in enumerate((("MoE (expert weights)", C["s1"]), ("Target attention", C["s2"]), ("Everything else", C["gray"]))):
    v = [r[3 + j] for r in rows]
    ax.bar(x, v, bottom=bottom, color=c, width=0.55, label=lab, edgecolor=C["bg"], linewidth=2)
    bottom = [p + q for p, q in zip(bottom, v)]
for i, r in enumerate(rows):
    ax.text(i, bottom[i] + 0.08, f"+{r[2] - r[1]:.2f} ms\n({r[1]:.2f} → {r[2]:.2f})", ha="center", va="bottom", fontsize=8.5, color=C["ink"])
ax.set_xticks(list(x)); ax.set_xticklabels([f"{r[0]} context" for r in rows]); ax.set_ylim(0, max(bottom) * 1.45)
ax.set_ylabel("Extra ms per step, block 16 vs 8"); ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Where the block-16 verify overhead goes (Nsight)")
save(fig, "overhead_by_context")

# 6. Output agreement with target-only greedy decoding.
DIV = {m: json.load(open(os.path.join(D, f"div_{m}.json"))) for m in MODES if os.path.exists(os.path.join(D, f"div_{m}.json"))}
order = ["stk_{}_hold", "sw8_{}_hold", "sw10_{}_hold", "sw12_{}_hold", "sw14_{}_hold", "sw16_{}_hold"]
lab = ["Stock MTP-4", "Block 8", "Block 10", "Block 12", "Block 14", "Block 16"]
if DIV:
    fig, ax = plt.subplots(figsize=(7, 3.6)); wdt = 0.36
    for j, (m, c) in enumerate(zip(MODES, (C["s1"], C["s2"]))):
        if m not in DIV:
            continue
        arms = DIV[m]["arms"]; v = [100 * arms[o.format(m)]["identical"] / arms[o.format(m)]["turns"] if o.format(m) in arms else float("nan") for o in order]
        xs = [i + (j - 0.5) * wdt for i in range(len(order))]
        ax.bar(xs, v, width=wdt, color=c, label=ML[m], edgecolor=C["bg"], linewidth=2)
        for xx, vv in zip(xs, v):
            if vv == vv:
                ax.text(xx, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=8, color=C["ink"])
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab); ax.set_ylim(0, 112)
    ax.set_ylabel("Outputs identical to target-only (%)"); ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Agreement with target-only greedy decoding (128 held-out turns)")
    save(fig, "divergence")

# Tables.
with open(os.path.join(F, "tables.md"), "w", encoding="utf-8") as fh:
    fh.write("| Arm | Mode | n | tok/s (95% CI) | vs block 8 (95% CI) | per-turn geo | vs stock | accept/round | step ms | step <32K / >128K |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for a in ["stk"] + [f"sw{b}" for b in BLOCKS] + ["ref"]:
        for m in MODES:
            if (a, m) not in S:
                continue
            s = S[a, m]; p8 = paired(a, "sw8", m) if a != "sw8" and ("sw8", m) in S else None; ps = paired(a, "stk", m) if a != "stk" and ("stk", m) in S else None
            fh.write(f"| {a} | {m} | {s['n']} | {s['tok_s']:.0f} ({s['ci'][0]:.0f}–{s['ci'][1]:.0f}) | "
                     f"{p8 and f'{p8['ratio']:.3f} ({p8['ci'][0]:.3f}–{p8['ci'][1]:.3f})' or '–'} | {p8 and f'{p8['geo']:.3f}' or '–'} | "
                     f"{ps and f'{ps['ratio']:.2f}x' or '–'} | {s['accept'] or 0:.2f} | {s['step_all']:.2f} | {s['step']['lt32k']:.2f} / {s['step']['gt128k']:.2f} |\n")
    fh.write("\nDrafter ablation (offline chain tokens/round, off / on):\n\n| Drafter | " + " | ".join(f"B{b}" for b in BLOCKS) + " |\n|---|" + "---|" * len(BLOCKS) + "\n")
    for d in ("Coral", "Run 1", "Run 2"):
        fh.write(f"| {d} | " + " | ".join(f"{CELLS[d, b][0]:.2f} / {CELLS[d, b][1]:.2f}" if (d, b) in CELLS else "–" for b in BLOCKS) + " |\n")
    fh.write("\nNsight step breakdown (ms/step):\n\n| Context | B8 step | B16 step | Δ MoE | Δ attention | Δ other |\n|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:+.2f} | {r[4]:+.2f} | {r[5]:+.2f} |\n")
    for m in DIV:
        fh.write(f"\nDivergence vs target-only, thinking {m} (ref round-trip mismatches {DIV[m]['ref_roundtrip_mismatch']}, ref tool calls / malformed {DIV[m]['ref_tool_calls']}):\n\n")
        fh.write("| Arm | identical | first div. token (median) | tool calls | malformed | margin n | |Δlogp| median | <0.05 | <0.2 | prefill top-1 = ref / arm |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for k, v in DIV[m]["arms"].items():
            g = v.get("margin", {})
            fh.write(f"| {k} | {v['identical']}/{v['turns']} | {v['first_div_median']} | {v['tool_calls']} | {v['malformed']} | {g.get('n', '–')} | "
                     f"{g.get('abs_margin_median', '–')} | {g.get('lt_0.05', '–')} | {g.get('lt_0.2', '–')} | {g.get('prefill_top1_ref', '–')} / {g.get('prefill_top1_arm', '–')} |\n")
sys.stdout.reconfigure(encoding="utf-8"); print(open(os.path.join(F, "tables.md"), encoding="utf-8").read())
