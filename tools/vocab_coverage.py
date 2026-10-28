"""How often does the target emit a token the draft vocabulary cannot propose?

  python tools/vocab_coverage.py <experiment-repo> > data/runs/vocab_coverage.txt

Offline re-analysis of recorded outputs, no GPU: the stock S2 outputs of runs [L52]/[L53]
(greedy, target-consistent) against each hot-token map, and against the 32K map extended
with the request's own prompt tokens (what the context-augmented head adds at prefill).
"""

import collections
import gzip
import json
import sys
from pathlib import Path

import torch

REPO = Path(sys.argv[1])
RUNS = ["20261026-050219-local", "20261026-052114-local"]
SETS = {"mtbench": "mtbench", "code_1k": "agentic_code_1024", "code_4k": "agentic_code_4096"}
MAPS = {"32K corpus": "hot_32768", "64K corpus": "hot_65536", "48K blend": "hot_49152_mix0.5"}

maps = {k: set(torch.load(REPO / "data" / "token_maps" / f"{v}.pt")) for k, v in MAPS.items()}
outs = collections.defaultdict(list)  # workload -> [(prompt_index, output_ids)]
for run in RUNS:
    seen = collections.Counter()
    for line in open(REPO / "experiments" / "runs" / run / "requests.jsonl"):
        r = json.loads(line)
        if r["variant"] == "S2":
            outs[r["workload"]].append((seen[r["workload"]], r["output_ids"]))
            seen[r["workload"]] += 1

print("# Share of stock target output tokens outside the draft vocabulary (lower is better)")
print(f"# outputs: stock S2, runs {', '.join(RUNS)}")
print(f"{'workload':<9} {'tokens':>6} " + " ".join(f"{k:>11}" for k in maps) +
      f" {'32K+prompt':>11} {'misses in prompt':>17} {'max prompt rows':>16}")
for w, items in outs.items():
    prompts = json.load(gzip.open(REPO / "data" / "prompts" / f"{SETS[w]}.json.gz", "rt"))
    tot, miss, ctx_miss, in_prompt, rows = 0, collections.Counter(), 0, 0, 0
    hot = maps["32K corpus"]
    for i, out in items:
        p = set(prompts[i])
        rows = max(rows, len(p - hot))
        for t in out:
            tot += 1
            for k, m in maps.items():
                miss[k] += t not in m
            if t not in hot:
                in_prompt += t in p
                ctx_miss += t not in p
    print(f"{w:<9} {tot:>6} " + " ".join(f"{100 * miss[k] / tot:>10.2f}%" for k in maps) +
          f" {100 * ctx_miss / tot:>10.2f}% {100 * in_prompt / max(miss['32K corpus'], 1):>16.0f}% {rows:>16}")
