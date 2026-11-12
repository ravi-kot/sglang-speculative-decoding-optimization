#!/usr/bin/env bash
# Session 17: the bounded boot (user: one more, then benchmarks). Watchdog stops the instance ~2h50m after boot: budget ~2h.
#  1. Run 6c: Run 6 (drafter_b24) + 0.4 epoch, new data order (--seed 7), lr 2e-5; unreadable feature files skipped
#  2. n24f: Nsight of the best stack (block 24 + all exact fixes), 3 regimes -> attribution of the remaining overhead (blog data)
#  3. paired turns_hold128: a16v (control) then a24l (all fixes; Run 6c only if its offline mean beats Run 6 by >= 0.03)
#  4. div17: correctness vs target-only greedy, no server queries
# Stop rule (user): block 24 not above ~960 off -> engineering ends; go to benchmarks.
R=$HOME/lab/profiles/q36_ft2; Q=$HOME/lab/env/q36
cd $R; mkdir -p s16_arms; mv rep_a16v_* rep_a16a_* rep_a24l_* s16_arms/ 2>/dev/null
export FT2_OUT=drafter_b24c FT2_T="--feat $R/feat,$R/feat_swe --turns $R/train_turns.jsonl,$R/swe_turns.jsonl --valfeat $R/featval --valturns $R/val_turns.jsonl"
export FT2_ARGS="--drafter $R/drafter_b24 --block 24 --epochs 0.4 --lr 2e-5 --gamma 7 --soft 0.5 --sel 1.0 --teacher-kl 2.0 --teacher-block 16 --seed 7 --src-repeat 1,4 --anchors 48 --accum 8 --val-every 200 --chain-turns 40 --val-max 400"
bash $Q/runq17.sh f2_train   # ~47 min (Run 6: 2,800 s); unreadable files are skipped now
python3 - <<'PY' > $R/r6c_gate.txt
import re, ast
best = None
for l in open("/home/ubuntu/lab/profiles/q36_ft2/f2_train_drafter_b24c.txt"):
    m = re.search(r"\(by mode (\{.*?\})\), first", l)
    if l.startswith("VAL step") and m:
        d = ast.literal_eval(m.group(1)); s = (d["chain_off"] + d["chain_on"]) / 2
        if best is None or s > best[0]: best = (s, d)
d = best[1] if best else {}
print("R6C", "BETTER" if d and (d["chain_off"] + d["chain_on"]) / 2 > (10.506 + 6.912) / 2 + 0.03 else "NOT_BETTER", {k: d.get(k) for k in ("chain_off", "chain_on", "seg_on_think")})
PY
cat $R/r6c_gate.txt
grep -q "R6C BETTER" $R/r6c_gate.txt && export A24_DRAFT=$R/drafter_b24c
bash $Q/runq17.sh n24f a16v a24l div17
