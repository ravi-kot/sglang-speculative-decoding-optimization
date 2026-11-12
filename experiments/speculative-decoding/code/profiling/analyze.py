"""Critical-path breakdown of a speculative-decoding trace (torch/CUPTI JSON or nsys sqlite).

  python profiling/analyze.py <trace.json[.gz] | report.sqlite> [--out analysis.txt]

Adapted from gdn_bcg_lab/analyze_trace.py. GPU activities (kernels, memcpy, memset) are linked
to the CPU API call that launched them via the CUPTI correlation id. A graph launch groups all
its kernels into one segment; segments are labelled `target` (the graph kind with the most kernels:
target verify/decode) or `draft` (every other graph: MTP draft/draft-extend); non-graph activities are
`eager`. Each target launch opens a new speculative round.

GPU idle = gaps between the merged union of GPU execution intervals (never derived from kernel
durations). Each gap is attributed to the activity that ends it, and marked host-late when that
activity's launch API call began after the GPU went idle (host submission on the critical
path); otherwise it is a dependency/sync/launch-latency gap. Exposed time of an activity is
the part of its interval not overlapped by earlier GPU work.
"""

import argparse
import collections
import gzip
import json
import sqlite3
import statistics
import sys

GAP_MIN_US = 1.0


def load_torch(path):
    op = gzip.open if path.endswith(".gz") else open
    ev = json.load(op(path, "rt"))["traceEvents"]
    api = {e["args"]["correlation"]: e for e in ev
           if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    acts = []
    for e in ev:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
            c = api.get(e["args"].get("correlation"), {})
            acts.append((e["name"], e["ts"], e["ts"] + e["dur"], e["args"].get("correlation"),
                         c.get("name", "?"), c.get("ts")))
    return acts


def load_nsys(path):
    db = sqlite3.connect(path)
    s = dict(db.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    api = {}
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t in tables:
            for corr, start, nid in db.execute(f"SELECT correlationId, start, nameId FROM {t}"):
                api[corr] = (s.get(nid, "?"), start / 1e3)
    acts = []
    q = {"CUPTI_ACTIVITY_KIND_KERNEL": "SELECT shortName, start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL",
         "CUPTI_ACTIVITY_KIND_MEMCPY": "SELECT -1, start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY",
         "CUPTI_ACTIVITY_KIND_MEMSET": "SELECT -2, start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"}
    for t, sql in q.items():
        if t in tables:
            for nid, st, en, corr in db.execute(sql):
                name = {-1: "memcpy", -2: "memset"}.get(nid) or s.get(nid, "?")
                a = api.get(corr, ("?", None))
                acts.append((name, st / 1e3, en / 1e3, corr, a[0], a[1]))
    return acts


def analyze(acts):
    acts = sorted(({"name": n, "ts": t, "end": e, "corr": c, "api": an, "api_ts": at,
                    "graph": "GraphLaunch" in an} for n, t, e, c, an, at in acts), key=lambda a: a["ts"])
    # graph segments: one per launching call
    seg = collections.defaultdict(list)
    for a in acts:
        if a["graph"]:
            seg[a["corr"]].append(a)
    size = {c: len(v) for c, v in seg.items()}
    big = max(size.values(), default=0)
    for c, v in seg.items():
        kind = "target" if size[c] == big else "draft"  # deep drafts can exceed half the target graph
        for a in v:
            a["kind"] = kind
    for a in acts:
        a.setdefault("kind", "eager")

    # merged-interval idle + exposed time + rounds
    rounds, cur, last_target_corr = [], None, None
    idle = collections.Counter()
    gaps, exposed = [], collections.Counter()
    busy_end = acts[0]["ts"]
    for a in acts:
        if a["kind"] == "target" and a["corr"] != last_target_corr:
            last_target_corr = a["corr"]
            cur = {"start": a["ts"], "busy": collections.Counter(), "idle_host": 0.0, "idle_other": 0.0}
            rounds.append(cur)
        gap = a["ts"] - busy_end
        if gap > GAP_MIN_US:
            host_late = a["api_ts"] is not None and a["api_ts"] > busy_end
            idle[(a["kind"], "host_late" if host_late else "dep/sync")] += gap
            gaps.append((gap, a["kind"], host_late, a["name"][:60], a["api"]))
            if cur is not None:
                cur["idle_host" if host_late else "idle_other"] += gap
        ex = max(0.0, a["end"] - max(a["ts"], busy_end))
        exposed[(a["kind"], a["name"][:70])] += ex
        if cur is not None:
            cur["busy"][a["kind"]] += ex
            cur["end"] = max(cur.get("end", 0), a["end"])
        busy_end = max(busy_end, a["end"])
    return acts, rounds, idle, gaps, exposed, size


def report(path):
    acts = load_nsys(path) if path.endswith((".sqlite", ".db")) else load_torch(path)
    if not acts:
        return "no GPU activities found"
    acts, rounds, idle, gaps, exposed, size = analyze(acts)
    t0, t1 = acts[0]["ts"], max(a["end"] for a in acts)
    win, busy = t1 - t0, sum(exposed.values())
    L = [f"trace {path}",
         f"window {win / 1e3:.2f} ms | GPU busy (merged) {busy / 1e3:.2f} ms | idle {(win - busy) / 1e3:.2f} ms "
         f"({100 * (win - busy) / win:.1f}%)",
         f"activities {len(acts)} | graph launches {len(size)} (kernels/launch: "
         f"{dict(collections.Counter(size.values()).most_common(4))})"]
    full = [r for r in rounds[:-1] if "end" in r]  # last round may be truncated
    if full:
        dur = [r2["start"] - r["start"] for r, r2 in zip(rounds, rounds[1:])]
        m = lambda f: statistics.median(f(r) for r in full)
        L += [f"rounds {len(full)} | round p50 {statistics.median(dur):.1f} us = target {m(lambda r: r['busy']['target']):.1f}"
              f" + draft {m(lambda r: r['busy']['draft']):.1f} + eager {m(lambda r: r['busy']['eager']):.1f}"
              f" + idle host-late {m(lambda r: r['idle_host']):.1f} + idle dep/sync {m(lambda r: r['idle_other']):.1f}"]
    L.append("idle attributed to the activity that ends the gap (us):")
    for (k, w), v in sorted(idle.items(), key=lambda kv: -kv[1]):
        L.append(f"  {k:<7} {w:<10} {v:10.1f}  ({100 * v / win:.1f}% of window)")
    L.append("exposed GPU time by kind (us): " + ", ".join(
        f"{k} {sum(v for (kk, _), v in exposed.items() if kk == k):.0f}" for k in ("target", "draft", "eager")))
    L.append("top exposed activities:")
    for (k, n), v in exposed.most_common(10):
        L.append(f"  {v:9.1f} us {100 * v / win:5.1f}%  {k:<6} {n}")
    L.append("largest idle gaps:")
    for g in sorted(gaps, reverse=True)[:6]:
        L.append(f"  {g[0]:8.1f} us before {g[1]:<6} host_late={g[2]} {g[3]} via {g[4]}")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = report(a.trace)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)
