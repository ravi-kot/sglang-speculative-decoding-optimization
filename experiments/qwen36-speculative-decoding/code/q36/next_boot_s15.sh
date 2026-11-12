#!/usr/bin/env bash
# Session 15 boot: one queue, GPU busy end to end. Order:
#  1. a128g  - our 128-row verify attention (attn128.py) vs FlashInfer fa2, block 16 and 24, config sweep -> a128_flags.env
#  2. moe24  - AVL cost table from the M=24 graph -> avl24.env;   hotcov (CPU) - target tokens outside the 48K draft vocab
#  3. sw24s  - block-24 engine smoke (Run 2 drafter, 8 turns per mode): Bole pads T to 32, dense/BA/KV-fuse fall back above 16
#  4. b24e   - offline chain floor: Run 2 at block 20 / 24 (positions 16+ untrained)
#  5. Run 6  - block-24 drafter: warm start Run 2, teacher KL 2.0 to Run 2 at block 16 (keeps positions 1-15, as Run 2 kept
#              Coral's 1-7), Run 4 data mix, lr 3e-5; gate: chain_off >= 9.90 (Run 2: 9.50) and chain_on >= 6.60 (6.64)
#  6. paired server arms on turns_hold128: a16v (control: block 16 + AVL), a16a (+ attn128), a24v (Run 6 + block 24 + AVL
#     [+ attn128]) if Run 6 passes; div15 vs target-only greedy.
R=$HOME/lab/profiles/q36_ft2; Q=$HOME/lab/env/q36
df -h / | tail -1 | tee $R/disk_s15.txt; lsblk -d -o NAME,SIZE,MODEL | tee -a $R/disk_s15.txt   # instance-store NVMe for larger data runs?
bash $Q/runq17.sh a128g moe24
bash $Q/run17.sh hotcov > $R/hotcov.out 2>&1 &   # CPU only, outside runq (one PID file)
bash $Q/runq17.sh sw24s b24e
wait
export FT2_OUT=drafter_b24 FT2_T="--feat $R/feat,$R/feat_swe --turns $R/train_turns.jsonl,$R/swe_turns.jsonl --valfeat $R/featval --valturns $R/val_turns.jsonl"
export FT2_ARGS="--drafter $R/drafter_b16_t --block 24 --epochs 0.4 --lr 3e-5 --gamma 7 --soft 0.5 --sel 1.0 --teacher-kl 2.0 --teacher-block 16 --src-repeat 1,4 --anchors 48 --accum 8 --val-every 200 --chain-turns 40 --val-max 400"
bash $Q/runq17.sh f2_train
python3 - <<'PY' > $R/r6_gate.txt
import re, ast
best = None
for l in open("/home/ubuntu/lab/profiles/q36_ft2/f2_train_drafter_b24.txt"):
    m = re.search(r"\(by mode (\{.*?\})\), first", l)
    if l.startswith("VAL step") and m:
        d = ast.literal_eval(m.group(1)); s = (d["chain_off"] + d["chain_on"]) / 2
        if best is None or s > best[0]: best = (s, d)
d = best[1] if best else {}
ok = bool(d) and d["chain_off"] >= 9.90 and d["chain_on"] >= 6.60
print("R6_GATE", "PASS" if ok else "FAIL", {k: d.get(k) for k in ("chain_off", "chain_on", "seg_on_think", "seg_off_tool")})
PY
cat $R/r6_gate.txt
if grep -q "R6_GATE PASS" $R/r6_gate.txt; then bash $Q/runq17.sh a16v a16a a24v div15; else bash $Q/runq17.sh a16v a16a div15; fi
