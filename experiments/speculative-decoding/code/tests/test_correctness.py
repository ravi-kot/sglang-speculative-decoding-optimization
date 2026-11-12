"""Target-consistency check for speculative outputs (greedy).

  python tests/test_correctness.py configs/local.yaml experiments/runs/<run_id> [--variants S2o,S28] [--verifier A]

Speculative decoding is lossless iff every committed token is the target model's greedy choice
given the committed prefix. For each recorded request, the plain target (variant --verifier, no
speculation) scores prompt + output with teacher forcing; every output token must equal the
target's top-1, except numerical near-ties (|logprob(top1) - logprob(chosen)| <= --tie).
Different draft heads legitimately change verify chunking, hence BF16 rounding and near-tie
outcomes, so exact sequence equality between variants is not the criterion; this test is.
The stock speculative reference (--base) sets the measurement's noise floor (prefill-path verifier
vs decode/verify-path numerics); exit code 1 if a variant's disagreement rate is significantly
higher than stock's (95% CI of the difference excludes 0).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("run_dir")
    ap.add_argument("--variants")
    ap.add_argument("--verifier", default="A")
    ap.add_argument("--tie", type=float, default=0.05)
    ap.add_argument("--base", default="S2", help="stock reference variant for the disagreement-rate comparison")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    run = Path(a.run_dir)
    recs = [json.loads(l) for l in (run / "requests.jsonl").read_text().splitlines()]
    if a.variants:
        recs = [r for r in recs if r["variant"] in a.variants.split(",")]
    sets = {w["name"]: w["set"] for w in cfg["workloads"]}
    prompts = {}
    srv = bench.Server(cfg, a.verifier, cfg["variants"][a.verifier] or [], run / f"server_verify_{a.verifier}.log")
    stats = {}
    try:
        srv.start(cfg.get("startup_timeout_s", 900))
        for r in recs:
            if r.get("finish_reason") == "abort" or not r["output_ids"]:
                continue
            p = prompts.setdefault(r["workload"], bench.load_set(sets[r["workload"]]))[r["prompt"]]
            out = r["output_ids"]
            resp = bench.http(srv.port, "/generate", {
                "input_ids": p + out, "return_logprob": True, "logprob_start_len": len(p) - 1,
                "top_logprobs_num": 1, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
            mi = resp["meta_info"]
            chosen = mi["input_token_logprobs"][1:]          # logprob of each output token given its prefix
            top = mi["input_top_logprobs"][1:]               # target's top-1 at the same positions
            s = stats.setdefault((r["variant"], r["workload"]), {"tokens": 0, "agree": 0, "ties": 0, "bad": 0,
                                                                  "worst_gap": 0.0})
            for (lp_c, tok_c, _), t1 in zip(chosen, top):
                lp_t, tok_t, _ = t1[0]
                s["tokens"] += 1
                if tok_t == tok_c:
                    s["agree"] += 1
                elif lp_t - lp_c <= a.tie:
                    s["ties"] += 1
                else:
                    s["bad"] += 1
                    s["worst_gap"] = max(s["worst_gap"], lp_t - lp_c)
    finally:
        srv.stop()
    worse = []  # variants whose disagreement rate is significantly above the stock reference
    for (v, w), s in sorted(stats.items()):
        line = (f"{v:<6} {w:<9} tokens={s['tokens']:<5} target-top1={s['agree'] / s['tokens']:.4f} "
                f"near-ties={s['ties']} NON-TIE MISMATCH={s['bad']} worst_gap={s['worst_gap']:.3f}")
        ref = stats.get((a.base, w)) if a.base and v != a.base else None
        if ref:  # difference in disagreement rate vs the stock reference, normal-approx 95% CI
            p1, n1 = 1 - s["agree"] / s["tokens"], s["tokens"]
            p0, n0 = 1 - ref["agree"] / ref["tokens"], ref["tokens"]
            se = (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) ** 0.5
            line += f" | disagree-rate vs {a.base}: {100 * (p1 - p0):+.2f} pp (95% CI {100 * (p1 - p0 - 1.96 * se):+.2f}..{100 * (p1 - p0 + 1.96 * se):+.2f})"
            if p1 - p0 - 1.96 * se > 0:
                worse.append(f"{v}|{w}")
        print(line)
    (run / "correctness.json").write_text(json.dumps({f"{v}|{w}": s for (v, w), s in stats.items()}, indent=1))
    if worse:
        print("SIGNIFICANTLY WORSE THAN STOCK:", worse)
    sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
