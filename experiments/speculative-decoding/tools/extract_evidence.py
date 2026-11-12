"""Copy the evidence the article cites from the experiment repo into data/ (compact, no raw traces).

  python tools/extract_evidence.py <experiment-repo> <blog-dir>

Run under WSL with the SGLang venv (needs the raw run directories and, for the NCU export, the
`ncu` CLI). Nothing here profiles or benchmarks: it re-reads existing traces, summaries, the
ledger, and one existing Nsight Compute report. Absolute local paths are stripped.
"""

import csv
import gzip
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO, BLOG = Path(sys.argv[1]), Path(sys.argv[2])
RUNS, OUT = REPO / "experiments" / "runs", BLOG / "data"
NCU = os.environ.get("NCU")  # path to the ncu CLI, optional
sys.path.insert(0, str(REPO / "profiling"))
import analyze  # noqa: E402  (the repo's own critical-path analyzer)


def clean(text):
    text = re.sub(r"/home/[^/\s]+/Workspace/work/sglang-decode-critical-path/?", "", text)
    text = re.sub(r"/home/[^/\s]+/Workspace/work/sglang-fp8head", "<sglang-overlay>", text)
    return re.sub(r"/(home|mnt)/[^\s'\"]+", "<local-path>", text)


def write(rel, text):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(clean(text), encoding="utf-8")
    print("wrote", p.relative_to(BLOG))


# 1. Ledger (every experiment, including failures) and the analyzer outputs the article cites.
write("ledger.jsonl", (REPO / "experiments" / "ledger.jsonl").read_text())
ANALYSES = {
    "20261024-204832-local": ["S2"],          # [L3] stock S2, code 1K
    "20261025-180250-local": ["S2o", "S28"],  # [L13] overlay off vs FP8 draft head, code 1K
    "20261025-185811-local": ["S2", "S28"],   # [L19] stock at 4K context
    "20261025-190409-local": ["S2K", "S28K"], # [L20] split-KV verify at 4K
    "20261026-054627-local": ["S5K4RH32QMD"], # [L55] final profile, code 1K
}
for run, variants in ANALYSES.items():
    for v in variants:
        f = RUNS / run / f"analysis_{v}.txt"
        if f.exists():
            write(f"profiler/analysis_{run}_{v}.txt", f.read_text())

# 2. Isolated draft-head benchmarks ([L6]-[L8]); host fingerprint dropped.
for run in ("20261024-215917-head", "20261024-220613-head", "20261024-220702-head"):
    s = json.loads((RUNS / run / "summary.json").read_text())
    s.pop("fingerprint", None)
    write(f"profiler/head_bench_{run}.json", json.dumps(s, indent=1))


# 3. One representative speculative round per trace, for the timeline figure.
def timeline(run, variant):
    f = next((RUNS / run / "traces").rglob(f"{variant}/*.trace.json.gz"))
    acts, rounds, *_ = analyze.analyze(analyze.load_torch(str(f)))
    starts = [r["start"] for r in rounds]
    durs = [b - a for a, b in zip(starts, starts[1:])]
    mid = sorted(range(len(durs)), key=lambda i: durs[i])[len(durs) // 2]
    t0, t1 = starts[mid], starts[mid + 1]
    segs = [{"kind": a["kind"], "start": round(a["ts"] - t0, 2), "end": round(min(a["end"], t1) - t0, 2),
             "name": a["name"][:60]} for a in acts if t0 <= a["ts"] < t1]
    return {"run": run, "variant": variant, "round_us": round(t1 - t0, 1),
            "round_p50_us": round(statistics.median(durs), 1), "segments": segs}


for run, v, name in (("20261024-204832-local", "S2", "stock_s2_code1k"),
                     ("20261026-054627-local", "S5K4RH32QMD", "final_code1k")):
    write(f"profiler/timeline_{name}.json", json.dumps(timeline(run, v)))


# 4. Kernel launch configurations recorded by CUPTI (grid, block, registers, shared memory).
def launch_configs(run, names):
    rows = {}
    for f in (RUNS / run / "traces").rglob("*.trace.json.gz"):
        var = f.parent.name
        for e in json.load(gzip.open(f))["traceEvents"]:
            if e.get("cat") == "kernel" and any(n in e["name"] for n in names):
                a = e["args"]
                k = (var, e["name"][:58], str(a.get("grid")), str(a.get("block")),
                     a.get("registers per thread"), a.get("shared memory"))
                rows.setdefault(k, []).append(e["dur"])
    return [f"{k[0]:<12} {k[1]:<58} grid {k[2]:<15} block {k[3]:<14} regs {k[4]:<4} smem {k[5]:<7} "
            f"calls {len(v):<4} mean {sum(v) / len(v):8.1f} us" for k, v in sorted(rows.items())]


lines = ["# CUPTI kernel launch configurations (torch profiler traces; 30 speculative rounds per trace)"]
for run, tag in (("20261024-204832-local", "[L3] stock, code 1K"),
                 ("20261025-185811-local", "[L19] stock, code 4K"),
                 ("20261025-190409-local", "[L20] split-KV verify, code 4K"),
                 ("20261026-054627-local", "[L55] final config, code 1K")):
    lines += ["", f"## {run} {tag}"] + launch_configs(
        run, ["_fwd_kernel", "_verify_prefix", "_verify_combine", "gemvx", "Marlin", "_fwd_grouped"])
write("profiler/launch_configs.txt", "\n".join(lines) + "\n")

# 5. Nsight Compute: export the existing draft-head report ([L6]) to CSV (metrics only).
rep = RUNS / "20261024-215917-head" / "traces" / "head_ncu.ncu-rep"
if NCU and rep.exists():
    raw = subprocess.run([NCU, "--import", str(rep), "--page", "details", "--csv"], capture_output=True,
                         text=True, check=True).stdout
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "kernel", "grid", "block", "metric", "unit", "value"])
    for r in csv.DictReader(io.StringIO(raw)):
        w.writerow([r["ID"], r["Kernel Name"][:90], r["Grid Size"], r["Block Size"], r["Metric Name"],
                    r["Metric Unit"], r["Metric Value"]])
    write("profiler/head_ncu_metrics.csv", buf.getvalue())


# 6. Paired comparisons and correctness for the runs behind the headline tables.
def compare(base, runs):
    cmd = [sys.executable, str(REPO / "scripts" / "compare.py"), "--base", base] + [str(RUNS / r) for r in runs]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=True).stdout


COMPARES = {
    "final_vs_stock_L74_L75.txt": ("S2", ["20261026-230007-local", "20261026-231805-local"]),
    "final_vs_previous_best_L74_L75.txt": ("S5K4RH32QMD", ["20261026-230007-local", "20261026-231805-local"]),
    "vocab_pooled_L68-L71.txt": ("S5K4RH32QMD", ["20261026-215748-local", "20261026-220951-local",
                                                "20261026-222407-local", "20261026-223528-local"]),
    "draft_extend_splitkv_L52_L53.txt": ("S2", ["20261026-050219-local", "20261026-052114-local"]),
    "depth_L52_L53.txt": ("S5K4RH32QMD", ["20261026-050219-local", "20261026-052114-local"]),
    "depth_sweep_L35_L36.txt": ("S2", ["20261026-011803-local", "20261026-013301-local"]),
    "adaptive_L64_L65.txt": ("S5K4RH32QMD", ["20261026-211316-local", "20261026-212522-local"]),
}
for name, (base, runs) in COMPARES.items():
    write(f"runs/{name}", f"# scripts/compare.py --base {base} {' '.join(runs)}\n" + compare(base, runs))
shutil.copy(RUNS / "20261026-231805-local" / "correctness.json", OUT / "runs" / "correctness_L75.json")

# 7. Power / clock telemetry recorded by the bench sampler for the final runs.
tele = []
for run in ("20261026-230007-local", "20261026-231805-local"):
    m = json.loads((RUNS / run / "manifest.json").read_text())

    for s in m.get("servers", []):
        keys = ("power_w_p50", "power_limit_w_min_p50", "power_capped_frac", "mem_clock_mhz_p10_p50",
                "sm_clock_mhz_p10_p50")
        tele.append(f"{run} {s.get('variant'):<14} " + " ".join(f"{k}={s.get(k)}" for k in keys))
write("runs/telemetry_L74_L75.txt", "\n".join(tele) + "\n")

# 8. Source snapshot linked from the article: overlay patches + harness (local paths stripped).
CODE = ["patches/fp8_draft_head.diff", "patches/splitkv_verify_cuda_pr39316.diff", "patches/certified_lm_head.diff",
        "scripts/bench.py", "scripts/compare.py", "tests/test_correctness.py", "profiling/analyze.py",
        "profiling/head_bench.py", "profiling/gemm_bench.py", "profiling/splitkv_bench.py",
        "profiling/lmhead_screen.py", "data/build_prompts.py", "data/build_token_map.py", "configs/local.yaml", "configs/adaptive_bs1.json"]
for rel in CODE:
    p = BLOG / "code" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(clean((REPO / rel).read_text()), encoding="utf-8")
    print("wrote", p.relative_to(BLOG))

