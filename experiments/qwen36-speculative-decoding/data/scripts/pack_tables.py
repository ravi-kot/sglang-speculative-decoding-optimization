"""Blog data: drafter_offline.csv, avl_moe_cost.csv, nsys_kernels.csv, correctness.csv, gates.csv from blog_data/raw/*."""
import ast, glob, json, os, re, sys

D = sys.argv[1]; R = os.path.join(D, "raw")
w = lambda name, hdr, rows: open(os.path.join(D, name), "w", encoding="utf-8").write(
    ",".join(hdr) + "\n" + "".join(",".join(f'"{x}"' if isinstance(x, str) and ("," in x or " " in x) else str(x) for x in r) + "\n" for r in rows))

# 1. Drafter offline chain eval (held-out validation traces): best checkpoint of each training run + eval-only cells.
RUNS = {"drafter_b16": ("Run 1", "block 16 from Coral, lr 1e-4, no teacher KL"), "drafter_b16_t": ("Run 2", "block 16, teacher KL 2.0 to Coral, lr 3e-5"),
        "drafter_r3a": ("Run 3a", "Run 2 + hard post-rejection anchors"), "drafter_r3b": ("Run 3b", "Run 2 + 120 Workato traces (4x)"),
        "drafter_r4": ("Run 4", "Run 2 + 140 SWE-rebench trajectories (4x)"), "drafter_r5c": ("Run 5c", "Run 4 + confidence-weighted loss 2.0"),
        "drafter_b24": ("Run 6", "block 24 from Run 2, teacher KL to Run 2 at block 16"), "drafter_b24b": ("Run 6b", "Run 6 continued (hung, step 200)"),
        "drafter_b24c": ("Run 6c", "Run 6 + 0.4 epoch, new data order"), "probe_overfit": ("Overfit probe", "Run 2 + 372 steps on 4 traces")}
rows = []
for f in sorted(glob.glob(os.path.join(R, "*", "f2_train_*.txt"))):
    k = os.path.basename(f)[9:-4]
    if k not in RUNS:
        continue
    best = None
    for l in open(f, encoding="utf-8", errors="replace"):
        m = re.search(r"^VAL step (\d+): block (\d+) ([0-9.]+) .*?\(by mode (\{.*?\})\)", l)
        if m:
            d = ast.literal_eval(m.group(4)); s = (d.get("chain_off", 0) + d.get("chain_on", 0)) / 2
            if best is None or s > best[0]:
                best = (s, int(m.group(1)), int(m.group(2)), d)
    if best:
        d = best[3]
        rows.append([RUNS[k][0], RUNS[k][1], best[2], best[1], d.get("chain_off"), d.get("chain_on"), d.get("seg_on_think"), d.get("seg_on_tool"), d.get("seg_off_tool"), os.path.relpath(f, D)])
for f in glob.glob(os.path.join(R, "*", "b24e.txt")):
    for l in open(f):
        m = re.search(r"^VAL before: block (\d+) .*'chain_off': ([0-9.]+), .*'seg_off_tool': ([0-9.]+), .*'chain_on': ([0-9.]+), 'seg_on_think': ([0-9.]+), 'seg_on_tool': ([0-9.]+)", l)
        if m:
            rows.append(["Run 2 (eval only)", "positions >= 16 untrained", int(m.group(1)), 0, float(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(3)), os.path.relpath(f, D)])
for f in glob.glob(os.path.join(R, "*", "abl.txt")):
    cur = None
    for l in open(f):
        m = re.match(r"ABL (\w+) block (\d+)", l)
        if m:
            cur = (m.group(1), int(m.group(2))); continue
        m = re.search(r"\(by mode (\{.*?\})\)", l)
        if m and cur:
            d = ast.literal_eval(m.group(1)); rows.append([{"coral": "Coral", "run1": "Run 1", "run2": "Run 2"}[cur[0]] + " (eval only)", "block sweep", cur[1], 0, d.get("chain_off"), d.get("chain_on"), d.get("seg_on_think"), d.get("seg_on_tool"), d.get("seg_off_tool"), os.path.relpath(f, D)]); cur = None
rows += [["Coral (eval only)", "released drafter", 8, 0, 6.58, 5.41, None, None, None, "scoreboard16.md (f2_base)"],
         ["Coral (eval only)", "released drafter", 16, 0, 8.39, 6.39, None, None, None, "scoreboard16.md (f2_base)"],
         ["Run 2 (eval only)", "block sweep", 10, 0, 7.54, 5.90, None, None, None, "scoreboard16.md (f2_blocks_t)"],
         ["Run 2 (eval only)", "block sweep", 12, 0, 8.25, 6.26, None, None, None, "scoreboard16.md (f2_blocks_t)"]]
w("drafter_offline.csv", ["drafter", "recipe", "block", "best_step", "chain_off", "chain_on", "seg_on_think", "seg_on_tool", "seg_off_tool", "source"], rows)

# 2. AVL MoE cost tables (graph-timed, real routing, rows >= L on row 0's experts).
M16 = {1: 1.253, 2: 1.684, 4: 2.308, 6: 2.867, 8: 3.387, 10: 3.727, 12: 4.062, 14: 4.371, 16: 4.798}
rows = [[16, L, v, "", "moe_real.py --run 16 --remap (session 13)"] for L, v in M16.items()]
for f in glob.glob(os.path.join(R, "*", "moe24.txt")):
    for l in open(f):
        m = re.search(r"GRAPH M=(\d+) L=(\d+): ([0-9.]+) us .*distinct ([0-9.]+).*x40 layers ([0-9.]+) ms", l)
        if m:
            rows.append([int(m.group(1)), int(m.group(2)), float(m.group(5)), float(m.group(4)), "moe_real.py --run 24 --remap (session 15)"])
w("avl_moe_cost.csv", ["graph_rows_M", "verify_length_L", "moe_ms_per_round_40_layers", "distinct_experts_per_layer", "source"], rows)

# 3. Nsight critical path per config / context regime (ms per step, top kernels).
CFG = {"n8": "block 8 BEST3", "n16": "block 16", "n16v": "block 16 + AVL", "n16l": "block 16 + AVL + L-skip", "n24l": "block 24 + AVL + L-skip",
       "n24f": "block 24 + AVL + L-skip + s5 M<=24 + Bole BV8 (best)"}
CTX = {1: "~18K", 2: "~101K", 3: "~165K"}
rows = []
for f in sorted(glob.glob(os.path.join(R, "*", "crit_*.txt"))):
    k = os.path.basename(f)[5:-4]; reg = None
    for l in open(f):
        m = re.match(r"== \S+ regime (\d)", l)
        if m:
            reg = int(m.group(1)); continue
        m = re.match(r"steps \d+, attributed ([0-9.]+) ms/step, gaps ([0-9.]+)", l)
        if m:
            rows += [[k, CFG.get(k, k), reg, CTX.get(reg), "_step_total", float(m.group(1))], [k, CFG.get(k, k), reg, CTX.get(reg), "_gaps", float(m.group(2))]]; continue
        m = re.match(r"\s+([0-9.]+) ms\s+[0-9.]+%\s+(.*)", l)
        if m and reg:
            rows.append([k, CFG.get(k, k), reg, CTX.get(reg), m.group(2).strip(), float(m.group(1))])
w("nsys_kernels.csv", ["profile", "config", "regime", "context_approx", "kernel", "ms_per_step"], rows)

# 4. Correctness vs target-only greedy (64 held-out turns per mode).
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(R, "*", "div*.txt"))):
    s = f.split(os.sep)[-2]
    for l in open(f):
        m = re.match(r"DIV (\S+): identical (\d+)/(\d+); first divergent token median ([0-9.]+) \(([0-9.]+) of output\); tool calls (\d+), malformed (\d+)", l)
        if m and (s, m.group(1)) not in seen:
            seen.add((s, m.group(1))); rows.append([s, m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6)), int(m.group(7)), os.path.relpath(f, D)])
w("correctness.csv", ["session", "arm_mode", "identical", "turns", "first_divergent_token_median", "first_divergent_fraction", "tool_calls", "malformed", "source"], rows)

# 5. Isolated kernel gates.
rows = []
for f in glob.glob(os.path.join(R, "*", "lskip_gate*.txt")):
    for l in open(f):
        m = re.match(r"LSKIP L=(\d+) B=(\d+) tiles (\d+)/(\d+) \(tokens < (\d+)\): ([0-9.]+) us vs full ([0-9.]+) us \| max\|diff\| rows<\d+ ([0-9.e+-]+)", l)
        if m:
            rows.append(["L-skip (FlashInfer plan variant)", f"B{m.group(2)} keep {m.group(3)}/{m.group(4)} tiles", int(m.group(1)), float(m.group(7)), float(m.group(6)), m.group(8), f.split(os.sep)[-2]])
for f in glob.glob(os.path.join(R, "*", "a128_gate16.txt")):
    for l in open(f):
        m = re.match(r"A128 L=(\d+) Lr=(\d+) cfg \((.*?)\): ours ([0-9.]+) us \| fa2 B(\d+) ([0-9.]+) us \| err ours ([0-9.e+-]+)", l)
        if m and m.group(3).startswith("32, 8, 1, 128"):
            rows.append(["Own Triton 128-row attention (rejected)", f"B{m.group(5)} rows<{m.group(2)}, cfg {m.group(3)}", int(m.group(1)), float(m.group(6)), float(m.group(4)), m.group(7), f.split(os.sep)[-2]])
for f in glob.glob(os.path.join(R, "*", "s5m24.txt")):
    for l in open(f):
        m = re.match(r"TIME (\S+) M=24 cold: stock ([0-9.]+) us, tile64 ([0-9.]+) us", l)
        if m:
            rows.append([f"s5 narrow dense GEMM {m.group(1)} (bitwise equal M=1..24)", "M=24 cold", "", float(m.group(2)), float(m.group(3)), "0 (bitwise)", f.split(os.sep)[-2]])
rows += [["Bole verify, solve V tile 8 + 2 warps (bitwise equal)", "T=24, 1 layer, graph", "", 27.68, 21.62, "0 (bitwise)", "s16 (console)"],
         ["Bole verify, default", "T=16 vs T=24 default", "", 12.10, 27.68, "-", "s16 (console)"]]
w("gates.csv", ["kernel", "case", "context_tokens", "baseline_us", "candidate_us", "max_abs_diff", "session"], rows)
print(open(os.path.join(D, "drafter_offline.csv")).read()); print(len(rows), "gate rows")
