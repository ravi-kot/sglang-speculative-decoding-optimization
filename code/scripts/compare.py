"""Compare variants against a baseline from one or more runs (same hardware/config only).

  python scripts/compare.py experiments/runs/<run_id> [more run dirs] [--base A]
  python scripts/compare.py --leaderboard [--base S2]     # best paired result per workload, all runs

Per workload and variant: median effective TPOT, paired TPOT ratio vs baseline with a bootstrap
95% CI (paired by prompt; rounds pooled), throughput gain = (1/ratio - 1) x 100%, decode tok/s =
1000 / TPOT, end-to-end tok/s (includes TTFT), TTFT, acceptance, and greedy agreement.
Refuses to compare runs whose fingerprint (GPU, image/sglang commit, model) differ.
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def boot_ci(pairs, n=2000, seed=0):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choice(pairs) for _ in pairs) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def prefix(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--base", default="A")
    ap.add_argument("--leaderboard", action="store_true")
    a = ap.parse_args()
    if a.leaderboard:
        return leaderboard(a.base)
    recs, keys = [], set()
    for d in map(Path, a.runs):
        m = json.loads((d / "manifest.json").read_text())
        fp = m["fingerprint"]
        keys.add((fp.get("gpu"), fp.get("image") or fp.get("sglang_commit"), str(m["config"]["model"])))
        for l in (d / "requests.jsonl").read_text().splitlines():
            r = json.loads(l)
            r["round"] = f"{d.name}:{r['round']}"  # pair within a run: rounds restart at 0 per run
            recs.append(r)
    if len(keys) > 1:
        raise SystemExit(f"refusing: runs differ in GPU/runtime/model: {keys}")
    by = defaultdict(list)
    for r in recs:
        by[(r["workload"], r["variant"])].append(r)
    print("| workload | var | n | tokens | TPOT p50 ms | TPOT ratio vs base (95% CI) | gain % (CI) | decode tok/s | "
          "e2e tok/s | TTFT p50 ms | accept len | greedy match / prefix |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for wl in sorted({w for w, _ in by}):
        base = {(r["round"], r["prompt"]): r for r in by.get((wl, a.base), [])}
        for v in sorted({v for w, v in by if w == wl}):
            rs = by[(wl, v)]
            tp = [r["eff_tpot_ms"] for r in rs if r.get("eff_tpot_ms")]
            row = [wl, v, len(rs), round(statistics.mean(r["completion_tokens"] for r in rs), 1),
                   round(statistics.median(tp), 3) if tp else "-"]
            pairs = [r["eff_tpot_ms"] / base[(r["round"], r["prompt"])]["eff_tpot_ms"] for r in rs
                     if (r["round"], r["prompt"]) in base and r.get("eff_tpot_ms")
                     and base[(r["round"], r["prompt"])].get("eff_tpot_ms")]
            if v != a.base and pairs:
                lo, hi = boot_ci(pairs)
                med = statistics.median(pairs)
                row.append(f"{med:.3f}x ({lo:.3f}-{hi:.3f})")
                row.append(f"{gain(med):+.1f} ({gain(hi):+.1f}..{gain(lo):+.1f})")
            else:
                row += ["base", "0"] if v == a.base else ["-", "-"]
            row.append(round(1000 / statistics.median(tp), 1) if tp else "-")
            row.append(round(sum(r["completion_tokens"] for r in rs) / (sum(r["e2e_ms"] for r in rs) / 1e3), 1))
            row.append(round(statistics.median(r["ttft_ms"] for r in rs if r.get("ttft_ms")), 1))
            acc = [r["spec_accept_length"] for r in rs if r.get("spec_accept_length")]
            row.append(round(statistics.mean(acc), 3) if acc else "-")
            matched = [(r["output_ids"], base[(r["round"], r["prompt"])]["output_ids"]) for r in rs
                       if (r["round"], r["prompt"]) in base]
            if v != a.base and matched:
                ex = sum(x == y for x, y in matched) / len(matched)
                pf = statistics.mean(prefix(x, y) for x, y in matched)
                row.append(f"{ex:.0%} / {pf:.0f}")
            else:
                row.append("-")
            print("| " + " | ".join(map(str, row)) + " |")


def gain(ratio):
    return (1 / ratio - 1) * 100


def leaderboard(base):
    """Per workload: the variant with the lowest pooled paired TPOT ratio vs the stock base, pooling every
    ok run that contains both (pairs are formed within a run only; no best-run selection)."""
    runs = Path(__file__).resolve().parents[1] / "experiments" / "runs"
    agg, ref = defaultdict(lambda: {"pairs": [], "tpot": [], "acc": [], "runs": set(), "vram": [], "clk": [],
                                    "corr": [0, 0]}), defaultdict(list)
    for d in sorted(runs.glob("*-local")):
        if not (d / "requests.jsonl").exists():
            continue
        m = json.loads((d / "manifest.json").read_text())
        if not str(m.get("status", "")).startswith("ok"):
            continue
        rs = [json.loads(l) for l in (d / "requests.jsonl").read_text().splitlines()]
        corr = json.loads((d / "correctness.json").read_text()) if (d / "correctness.json").exists() else {}
        srv = {s["variant"]: s for s in m["servers"]}
        for wl in {r["workload"] for r in rs}:
            b = {(r["round"], r["prompt"]): r for r in rs if r["workload"] == wl and r["variant"] == base}
            if not b:
                continue
            ref[wl] += [x["eff_tpot_ms"] for x in b.values()]
            for v in {r["variant"] for r in rs if r["workload"] == wl} - {base}:
                vr = [r for r in rs if r["workload"] == wl and r["variant"] == v and r.get("eff_tpot_ms")]
                g = agg[(wl, v)]
                g["pairs"] += [r["eff_tpot_ms"] / b[(r["round"], r["prompt"])]["eff_tpot_ms"] for r in vr
                               if (r["round"], r["prompt"]) in b]
                g["tpot"] += [r["eff_tpot_ms"] for r in vr]
                g["acc"] += [r.get("spec_accept_length", 0) for r in vr]
                g["runs"].add(d.name)
                g["vram"].append(srv.get(v, {}).get("peak_vram_mib") or 0)
                g["clk"].append((srv.get(v, {}).get("sm_clock_mhz_p10_p50") or [0, 0])[1] or 0)
                c = corr.get(f"{v}|{wl}")
                if c:
                    g["corr"][0] += c["agree"]
                    g["corr"][1] += c["tokens"]
    print(f"| workload | stock {base} TPOT | best variant | TPOT ms | decode tok/s | pooled paired ratio (95% CI) | gain % "
          f"| accept | peak VRAM MiB | SM MHz p50 | target-top1 | runs / pairs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for wl in sorted({w for w, _ in agg}):
        cands = {v: g for (w, v), g in agg.items() if w == wl and len(g["pairs"]) >= 8}
        if not cands:
            continue
        v, g = min(cands.items(), key=lambda kv: statistics.median(kv[1]["pairs"]))
        med, (lo, hi) = statistics.median(g["pairs"]), boot_ci(g["pairs"])
        tp = statistics.median(g["tpot"])
        top1 = f"{g['corr'][0] / g['corr'][1]:.4f}" if g["corr"][1] else "n/a"
        print(f"| {wl} | {statistics.median(ref[wl]):.3f} | {v} | {tp:.3f} | {1000 / tp:.1f} | {med:.3f}x ({lo:.3f}-{hi:.3f}) | "
              f"{gain(med):+.1f} | {statistics.mean(g['acc']):.3f} | {max(g['vram'])} | {statistics.median(g['clk']):.0f} | "
              f"{top1} | {len(g['runs'])} / {len(g['pairs'])} |")


if __name__ == "__main__":
    main()
