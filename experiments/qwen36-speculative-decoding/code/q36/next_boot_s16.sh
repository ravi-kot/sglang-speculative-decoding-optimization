#!/usr/bin/env bash
# Session 16 boot (watchdog guard ~2h50m): attribute block 24's +1.56 ms step, train Run 6 further, paired arms.
#  1. nsys n16l (block 16 + AVL + L-skip) and n24l (block 24 + Run 6 + AVL + L-skip): per-kernel critical path, 3 regimes
#  2. Run 6b: Run 6 + 0.4 epoch (same recipe, teacher KL to Run 2 at block 16); still rising at step 1,244
#  3. paired turns_hold128: a16v (control), a16a (L-skip, block 16), a24l (block 24 + L-skip, Run 6b if better offline)
R=$HOME/lab/profiles/q36_ft2; Q=$HOME/lab/env/q36
cd $R; mkdir -p s15_arms; mv rep_a16v_* rep_a16a_* rep_a24v_* rep_a24l_* s15_arms/ 2>/dev/null
bash $Q/runq17.sh n16l n24l
export FT2_OUT=drafter_b24b FT2_T="--feat $R/feat,$R/feat_swe --turns $R/train_turns.jsonl,$R/swe_turns.jsonl --valfeat $R/featval --valturns $R/val_turns.jsonl"
export FT2_ARGS="--drafter $R/drafter_b24 --block 24 --epochs 0.4 --lr 2e-5 --gamma 7 --soft 0.5 --sel 1.0 --teacher-kl 2.0 --teacher-block 16 --seed 7 --src-repeat 1,4 --anchors 48 --accum 8 --val-every 200 --chain-turns 40 --val-max 400"
bash $Q/runq17.sh f2_train
python3 - <<'PY' > $R/r6b_gate.txt
import re, ast
best = None
for l in open("/home/ubuntu/lab/profiles/q36_ft2/f2_train_drafter_b24b.txt"):
    m = re.search(r"\(by mode (\{.*?\})\), first", l)
    if l.startswith("VAL step") and m:
        d = ast.literal_eval(m.group(1)); s = (d["chain_off"] + d["chain_on"]) / 2
        if best is None or s > best[0]: best = (s, d)
d = best[1] if best else {}
print("R6B", "BETTER" if d and (d["chain_off"] + d["chain_on"]) / 2 > (10.506 + 6.912) / 2 + 0.03 else "NOT_BETTER", {k: d.get(k) for k in ("chain_off", "chain_on", "seg_on_think")})
PY
cat $R/r6b_gate.txt
grep -q "R6B BETTER" $R/r6b_gate.txt && export A24_DRAFT=$R/drafter_b24b
bash $Q/runq17.sh a16v a24l a16a
