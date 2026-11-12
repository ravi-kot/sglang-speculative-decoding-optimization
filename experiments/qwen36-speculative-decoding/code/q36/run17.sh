#!/usr/bin/env bash
# Drafter training data v2 (RTX): 64 training traces with full per-trace context features (fp8) + 4 held-out validation traces.
# Drafter fine-tuning data (RTX): labels + drafter-input features from BEST3 at 8 concurrent requests on the 24 training traces.
# FINAL blog run (RTX): final stack vs stock SGLang 0.5.21 MTP-4 on the same 128 held-out turns, thinking off and on, one session.
# Session 10 (RTX): BEST2 confirmation at scale on all independent held-out turns + stock SGLang MTP-4 anchor, same session.
# Session 9 (RTX): one-wave attention KV split (Q37_ATTN_ONEWAVE), isolated gate first; control + arm only if it passes.
# Session 8 (RTX): MoE v3 (s7_moe.cu), gated in a captured graph first; control + arm only if it wins in-graph.
# Session 7 (RTX): MoE fix, gated in a captured graph. Session 6 header kept below.
# Session 6 (RTX). Reference = BEST (748 / 606): Session 3 final + KV_PAR + TRACK_GUARD + BA_EXACT + CONV_TILE + KV_FUSE, moe_cfg_both.
# One Track E experiment: narrow-tile dense FP8 GEMMs + 8-CTA BA kernel, enabled only where s5_dense_exact.py proves bitwise equality.
# Usage: run10.sh <item> [client]. Run through ~/lab/runq.sh (PID file); never pkill by pattern from the controlling shell.
set -u
E=$HOME/lab/env; Q=$E/q36; S4=$HOME/lab/profiles/q36_s4; R=$HOME/lab/profiles/q36_ft2; export RUN=$R PYTHONPATH=$E NSYS_TMPDIR=$R/nsys_tmp; mkdir -p $R/nsys $R/nsys_tmp; cd $R
PY=$HOME/lab/venv-bole2/bin/python; VQ=$HOME/lab/venv-q37; OFF=$HOME/lab/profiles/q36_official; I=$1; MODE=${2:-server}
BEST="VENV=$VQ Q37_SYNCFREE=1 Q37_HOTVOCAB=$S4/hotv/hot_ids_48000.pt FP8KV=1 ARM=dfl_bole Q37_KV_PAR=64 Q37_TRACK_GUARD=1 Q37_BA_EXACT=128,1024 Q37_CONV_TILE=64,4 Q37_KV_FUSE=1"
# Session 6 result: BEST + narrow-tile dense GEMMs + 8-CTA BA kernel, Track E (256/256 outputs identical, -0.08 to -0.10 ms, +1%).
BEST2="$BEST Q37_S5_DENSE=64 Q37_S5_BA=128"
WIN="--speculative-draft-window-size 8192"
[ -f $R/s6_flags.env ] && . $R/s6_flags.env
rep() { $PY $Q/swe_replay.py --turns $3 --arm $1 --thinking $2 --out rep_$1_$2${4:-}.jsonl | tail -14 > rep_$1_$2${4:-}.txt; }  # $4: output tag
arm() { env $BEST ${2:-} Q37_ACCEPT_LOG=$R/accept_$1.txt MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$1 CLIENT="bash $Q/run17.sh $1 client" bash $Q/q36_arm.sh
        grep -E "q37 (s5|ba exact|conv|kv fuse)" server_$1.log | sort | uniq -c | cut -c1-160 > used_$1.txt; nvidia-smi --query-gpu=memory.used --format=csv,noheader > mem_after_$1.txt; }
case "$I:$MODE" in
  s6_prep:server) export PATH=$VQ/bin:$PATH
    D=$VQ/lib/python3.12/site-packages/sglang/kernels/jit/csrc/gemm/fp8_blockwise_s5; mkdir -p $D; cp $Q/s5_kernels/fp8_blockwise_scaled_mm_*.cuh $D/
    ( cd $Q/s5_kernels && rm -f s5_ba.so && nvcc -std=c++17 -O3 -arch=sm_120a -w -shared -Xcompiler -fPIC -cudart shared -o s5_ba.so s5_ba.cu ) 2>&1 | grep -iE "error" | cut -c1-200
    ( cd $Q/s5_kernels && rm -f s5_moe.so && nvcc -std=c++17 -O3 -arch=sm_120a -w -shared -Xcompiler -fPIC -cudart shared -o s5_moe.so s5_moe.cu ) 2>&1 | grep -iE "error" | cut -c1-200
    nvidia-smi --query-gpu=name,memory.used,clocks.sm,clocks.mem,power.draw --format=csv | tee gpu_pre.txt
    ( cd /tmp && timeout 900 python $Q/s5_dense_exact.py 64 2>&1 | grep -E "^(BUILD|REAL|BITWISE|TIME|S5_)" | tee $R/s6_dense.txt | cut -c1-220 )
    ( cd /tmp && SGLANG_MOE_CONFIG_DIR=$S4/moe_cfg_both timeout 900 python $Q/s5_moe_exact.py $HOME/lab/profiles/q36_s3/routes.jsonl 20 2>&1 | grep -E "^(QUANT|WEIGHTS|ROUTES|VAR|REPEAT|TIME|S5_|Traceback|\[rank0\]: [A-Za-z]*Error|[A-Za-z]*Error)" | tee $R/s6_moe.txt | cut -c1-220 )
    python - <<'PY' > $R/s6_flags.env
import re
t = open("/home/ubuntu/lab/profiles/q36_s6/s6_dense.txt").read()
flags = []
# A tile is eligible only if every BITWISE line for it says equal; pick the faster eligible tile on out_proj + qkvz pair.
ok = {T: all("equal" in l for l in t.splitlines() if l.startswith("BITWISE") and f"tile {T}:" in l) and f"BUILD tile {T} ok" in t for T in (64,)}
tm = {T: float(m.group(1)) for T in (64,) for m in [re.search(rf"tile{T} ([0-9.]+) us", next((l for l in t.splitlines() if l.startswith("TIME out_o_proj")), ""))] if m}
st = re.search(r"TIME out_o_proj M=8 cold: stock ([0-9.]+)", t)
cand = [T for T in (64,) if ok[T] and T in tm and st and tm[T] < float(st.group(1)) * 0.95]
if cand:
    flags.append(f"Q37_S5_DENSE={min(cand, key=lambda T: tm[T])}")
if re.search(r"BITWISE in_proj_ba s5_ba vs production s4_ba: equal for M=1..16", t):
    flags.append("Q37_S5_BA=128")
print('S6="' + " ".join(flags) + '"')
m = open("/home/ubuntu/lab/profiles/q36_s6/s6_moe.txt").read()
r = re.search(r"S5_MOE_(EXACT|INEXACT) (\d) saves_us (-?[0-9.]+)", m)
rep_ok = "REPEAT 100 calls identical: True" in m
if r and rep_ok and float(r.group(3)) >= 5.0:
    print(f'S6MOE="Q37_S5_MOE={r.group(2)}"'); print(f'S6MOE_TRACK="{"E" if r.group(1) == "EXACT" else "S"}"')
else:
    print('S6MOE=""')
PY
    cat $R/s6_flags.env ;;
  g2_val:server) env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_FEAT_DUMP=$R/featval Q37_FEAT_W=8192 MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh ;;
  g2_val:client) $PY $Q/gen_client.py --turns $R/val_turns.jsonl --mode off --conc 4 --done $R/done_val_off.txt | tail -2; $PY $Q/gen_client.py --turns $R/val_turns.jsonl --mode on --conc 4 --done $R/done_val_on.txt | tail -2 ;;
  # Data expansion (2026-11-10): independent Workato agentic traces (reasoning + tool calls, in no eval set), target-generated labels.
  # Run 4 data (next boot): independent in-domain SWE-agent trajectories (nebius/SWE-rebench-openhands-trajectories, CC-BY-4.0), one per
  # repo, repos overlapping any MLCommons trace excluded; 1 parquet row group (~127 MB) read by HTTP range, no full download.
  r4prep:server) X="$OFF/turns_dev64.jsonl $R/turns_hold128.jsonl $R/val_turns.jsonl $R/train_turns.jsonl"
    PA=$(for p in $VQ/bin/python $PY /usr/bin/python3; do $p -c "import pyarrow" 2>/dev/null && { echo $p; break; }; done); echo "pyarrow python: ${PA:-NONE}" | tee $R/r4prep.txt
    ( cd /tmp && ${PA:-false} -I $Q/swe_fetch.py --exclude $X --n ${SWE_N:-140} --max-groups ${SWE_G:-1} ${SWE_AVOID:+--avoid $SWE_AVOID --skip-groups 11} --out $R/swe_traj${SWE_TAG:-}.jsonl ) 2>&1 | grep -E "excluded|PLAN|SWE_FETCH|row group|Error" | tee -a $R/r4prep.txt
    ( cd /tmp && $PY $Q/build_train.py --data /dev/null --tok $HOME/lab/models/Qwen3.6-35B-A3B-FP8 --exclude $X --source swe --swe $R/swe_traj${SWE_TAG:-}.jsonl --turns-per-trace ${SWE_TPT:-8} --out $R/swe_turns${SWE_TAG:-}.jsonl ) 2>&1 | grep -E "swe|Error" | tee -a $R/r4prep.txt ;;
  g2_swe:server) env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_FEAT_DUMP=$R/feat_swe${SWE_TAG:-} Q37_FEAT_W=${SWE_W:-400000} Q37_FEAT_FP8=1 MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh ;;
  g2_swe:client) for m in ${SWE_MODES:-off on}; do $PY $Q/gen_client.py --turns $R/swe_turns${SWE_TAG:-}.jsonl --mode $m --conc 8 --done $R/done_swe${SWE_TAG:-}_$m.txt | tail -2; done ;;
  g2_wk:server) env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_FEAT_DUMP=$R/feat_wk Q37_FEAT_W=400000 Q37_FEAT_FP8=1 MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh ;;
  g2_wk:client) $PY $Q/gen_client.py --turns $R/wk_turns.jsonl --mode off --conc 8 --done $R/done_wk_off.txt | tail -2; $PY $Q/gen_client.py --turns $R/wk_turns.jsonl --mode on --conc 8 --done $R/done_wk_on.txt | tail -2 ;;
  g2_train:server) env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_FEAT_DUMP=$R/feat Q37_FEAT_W=400000 Q37_FEAT_FP8=1 MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh ;;
  g2_train:client) $PY $Q/gen_client.py --turns $R/train_turns.jsonl --mode off --conc 8 --done $R/done_off.txt | tee -a $R/gen_off.log | tail -2
    $PY $Q/gen_client.py --turns $R/train_turns.jsonl --mode on --conc 8 --done $R/done_on.txt | tee -a $R/gen_on.log | tail -2 ;;
  f2_base:server|f2_train:server) export PATH=$VQ/bin:$PATH
    # GPU must be clear (no server left): wait up to 2 min, else stop
    for n in $(seq 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$U" -lt 2000 ] && break; sleep 5; done
    echo "$(date -u +%T) gpu used ${U} MiB before $I" | tee -a $R/gpu_clear.log; [ "$U" -lt 2000 ] || { echo "GPU not clear; skipping"; exit 1; }
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm --format=csv,noheader -l 10 > $R/gpu_$I${FT2_OUT:+_$FT2_OUT}.csv & GP=$!
    T=${FT2_T:-"--feat $R/feat --turns $R/train_turns.jsonl --valfeat $R/featval --valturns $R/val_turns.jsonl"}
    if [ $I = f2_base ]; then
      for B in 16 8; do ( cd /tmp && python $Q/ft2_train.py $T --out $R/none --block $B --eval-only --metrics $R/ft2_metrics_base$B.jsonl 2>&1 | grep --line-buffered -E "^(train|VAL|Traceback)|Error" | tee -a $R/f2_base.txt ); done
    else
      ( cd /tmp && python $Q/ft2_train.py $T --out $R/${FT2_OUT:-drafter_b16} --metrics $R/ft2_metrics_${FT2_OUT:-drafter_b16}.jsonl ${FT2_ARGS:-} 2>&1 | grep --line-buffered -E "^(train|VAL|steps|step|SAVED|FT2_DONE|Traceback|skip)|Error" | tee $R/f2_train_${FT2_OUT:-drafter_b16}.txt )
    fi
    kill $GP ;;
  # Blog validation queue (2026-11-10). Clean held-out set: turns_hold128.jsonl (32/bin from the 45 traces never used for
  # training or checkpoint selection, dev64 turns excluded). Offline drafter ablation fills the missing (drafter, block) cells;
  # chain eval is deterministic, so earlier cells (Coral 8/16, Run 1 16, Run 2 10/12/16) are reused.
  abl:server) export PATH=$VQ/bin:$PATH
    T="--feat /nonexistent --turns $R/val_turns.jsonl --valfeat $R/featval --valturns $R/val_turns.jsonl --out /tmp/none --eval-only --val-max 400 --chain-turns 40"
    for s in coral:$HOME/lab/models/qwen-swe-drafter:10,12,14 run1:$R/drafter_b16:8,10,12,14 run2:$R/drafter_b16_t:8,14; do
      IFS=: read n d bs <<< "$s"
      for B in ${bs//,/ }; do echo "ABL $n block $B $(date -u +%T)" >> $R/abl.txt
        ( cd /tmp && python $Q/ft2_train.py $T --drafter $d --block $B 2>&1 | grep -E "^VAL|Traceback|Error" | cut -c1-400 >> $R/abl.txt ); done
    done ;;
  stk:server) ARM=mtp4 TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh; nvidia-smi --query-gpu=memory.used --format=csv,noheader > mem_after_$I.txt ;;
  sw8:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1" ;;   # BEST3: Coral's drafter at block 8
  sw10:server|sw12:server|sw14:server|sw16:server) B=${I#sw}; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=$B DFLBLOCK=$B DRAFTM=$R/drafter_b16_t" ;;
  ref:server) ARM=nospec FP8KV=1 MOECFG=$S4/moe_cfg_both TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh ;;   # target-only greedy
  stk:client|sw8:client|sw10:client|sw12:client|sw14:client|sw16:client) rep $I off $R/turns_hold128.jsonl _hold; rep $I on $R/turns_hold128.jsonl _hold ;;
  ref:client) python3 -c "import json,collections as C;c=C.Counter();[print(l,end='') for l in open('$R/turns_hold128.jsonl') if (lambda b:(c.update([b]),c[b]<=16)[1])(json.loads(l)['bin'])]" > $R/turns_hold64.jsonl
    rep $I off $R/turns_hold64.jsonl _hold; rep $I on $R/turns_hold64.jsonl _hold
    for m in off on; do A=""; for x in stk sw8 sw10 sw12 sw14 sw16 a16c a16v b8on; do [ -f rep_${x}_${m}_hold.jsonl ] && A="$A rep_${x}_${m}_hold.jsonl"; done
      $PY $Q/diverge.py --turns $R/turns_hold128.jsonl --mode $m --ref rep_ref_${m}_hold.jsonl --arms $A --margins 48 --out div_$m.json 2>&1 | grep -E "^(DIV|MARGIN)|Error" > div_$m.txt; done ;;
  b16:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 ${B16_DRAFT:+DRAFTM=$B16_DRAFT}" ;;
  b8c:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1" ;;
  # Engine track (2026-11-10): adaptive verification length (Q37_AVL) vs block-16 control, paired on turns_hold128 (clean held-out).
  avlsm:server|a16c:server|a16v:server) X="Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 DRAFTM=${A_DRAFT:-$R/drafter_b16_t}"
    case $I in avlsm) X="$X Q37_AVL=1 Q37_AVL_LOG=1 ${AVL_EXTRA:-}" ;; a16v) X="$X Q37_AVL=1 ${AVL_EXTRA:-}" ;; esac
    arm $I "$X"; grep -E "q37 avl" server_$I.log | tail -3 > avl_$I.txt ;;
  a16v4:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 DRAFTM=$R/drafter_r4 Q37_AVL=1" ;;   # Run 4 drafter + AVL
  a16v4:client) rep $I off $R/turns_hold128.jsonl _hold; rep $I on $R/turns_hold128.jsonl _hold ;;
  avlc:server|avl_m4:server|avl_l5:server|avlr5:server) X="Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 DRAFTM=${A_DRAFT:-$R/drafter_b16_t} Q37_AVL=1"
    case $I in avl_m4) X="$X Q37_AVL_MIN=4" ;; avl_l5) X="$X Q37_AVL_LAM=0.5" ;; avlr5) X="$X DRAFTM=$R/drafter_r5c" ;; esac   # AVL thinking-off tuning, paired with avlc
    arm $I "$X" ;;
  avlc:client|avl_m4:client|avl_l5:client|avlr5:client) rep $I off $R/turns_hold128.jsonl _hold ;;
  # Session 15 (prepared 2026-11-10): own 128-row verify attention (attn128.py, Q37_ATTN128) + block-24 drafter (Run 6).
  a128g:server) export PATH=$VQ/bin:$PATH; SP=$VQ/lib/python3.12/site-packages
    grep -nE "_fixed_batch_size =|self._batch_size =|_paged_kv_indptr_buf =|self._sm_scale =|self._causal =|    def run\(" $SP/flashinfer/prefill.py | head -20 > $R/a128_fi_attrs.txt
    grep -nE "prefill_wrapper_paged\.(forward|run)\(|\.forward\(|\.run\(" $SP/sglang/srt/layers/attention/flashinfer_backend.py | head -8 >> $R/a128_fi_attrs.txt
    for B in 16 24; do ( cd /tmp && timeout 1500 python $Q/s9_attn_gate.py a128 $B 2>&1 | grep -E "^(A128|S9_|Traceback)|Error" | tee $R/a128_gate$B.txt | cut -c1-200 ); done
    python3 - $R > $R/a128_flags.env <<'PY'
import re, sys
R = sys.argv[1]
for B in (16, 24):
    try:
        t = open(f"{R}/a128_gate{B}.txt").read()
    except OSError:
        t = ""
    m = re.search(r"S9_A128 cfg ([0-9,]+) saves_us_per_layer (-?[0-9.]+) at Lr \d+; (.*)", t)
    lr = {int(k[2:]): float(v) for k, v in re.findall(r"(Lr\d+) (-?[0-9.]+)", m.group(3))} if m else {}
    full, half = (float(m.group(2)), lr.get(8, -1e9)) if m else (-1e9, -1e9)
    ok = full >= 5.0 or (full >= -5.0 and half >= 20.0)
    print(f'A128{"" if B == 16 else "_24"}="' + (f"Q37_ATTN128=1 Q37_ATTN128_CFG={m.group(1)}" if ok else "") + f'"   # full {full:.1f} us/layer, Lr8 {half:.1f}')
PY
    cat $R/a128_flags.env ;;
  moe24:server) export PATH=$VQ/bin:$PATH   # AVL cost table for block 24: every L from the M=24 graph (rows >= L on row 0's experts)
    ( cd /tmp && SGLANG_MOE_CONFIG_DIR=$S4/moe_cfg_both timeout 1200 python $Q/moe_real.py --routes $HOME/lab/profiles/q36_s3/routes.jsonl --run 24 --remap 1 2 4 6 8 10 12 14 16 18 20 22 24 2>&1 | grep -E "^GRAPH|Traceback|Error" | tee $R/moe24.txt | cut -c1-200 )
    echo "AVL24=\"Q37_AVL_MEAS=$(grep -oE 'L=[0-9]+:.*x40 layers [0-9.]+' $R/moe24.txt | sed -E 's/L=([0-9]+):.*x40 layers ([0-9.]+)/\1:\2/' | paste -sd,)\"" > $R/avl24.env; cat $R/avl24.env ;;
  b24e:server) export PATH=$VQ/bin:$PATH   # offline chain: Run 2 at blocks 20 / 24 (positions 16+ untrained) = the floor Run 6 must beat
    T="--feat /nonexistent --turns $R/val_turns.jsonl --valfeat $R/featval --valturns $R/val_turns.jsonl --out /tmp/none --eval-only --val-max 400 --chain-turns 40"
    for B in 20 24; do echo "B24E run2 block $B $(date -u +%T)" >> $R/b24e.txt
      ( cd /tmp && python $Q/ft2_train.py $T --drafter $R/drafter_b16_t --block $B 2>&1 | grep -E "^VAL|Traceback|Error" | cut -c1-400 >> $R/b24e.txt ); done ;;
  sw24s:server) . $R/avl24.env; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=24 DFLBLOCK=24 DRAFTM=${B24_DRAFT:-$R/drafter_b16_t} Q37_AVL=1 $AVL24" ;;
  sw24s:client) for m in off on; do $PY $Q/swe_replay.py --turns $R/turns_hold128.jsonl --arm $I --thinking $m --limit 8 --out rep_${I}_${m}.jsonl | tail -3 > rep_${I}_${m}.txt; done ;;
  s5m24:server) export PATH=$VQ/bin:$PATH   # narrow dense GEMM at block 24: bitwise vs stock for M=1..24, timing at M=24
    ( cd /tmp && S5_MMAX=24 S5_TM=24 timeout 900 python $Q/s5_dense_exact.py 64 2>&1 | grep -E "^(BUILD|REAL|BITWISE|TIME|S5_)|Error" | tee $R/s5m24.txt | cut -c1-220 ) ;;
  div17:server) for m in off on; do A=""; for x in a16v a24l; do [ -f rep_${x}_${m}_hold.jsonl ] && A="$A rep_${x}_${m}_hold.jsonl"; done
      $PY $Q/diverge.py --turns $R/turns_hold128.jsonl --mode $m --ref rep_ref_${m}_hold.jsonl --arms $A --margins 0 --out div17_$m.json 2>&1 | grep -E "^DIV|Error" > div17_$m.txt; done; cat div17_*.txt ;;
  # Benchmark (blog data): turns_bench.jsonl = 512 held-out turns (128/bin, 45 never-trained sessions, disjoint from hold128). Resumable.
  bbest:server) . $R/avl24.env; . $R/lskip_flags.env
    arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=24 DFLBLOCK=24 DRAFTM=$R/drafter_b24 Q37_AVL=1 $AVL24 ${LSK24:-} Q37_S5_MMAX=24 Q37_BOLE_WARPS=solvebv:8,solve:2" ;;
  bb16:server) . $R/lskip_flags.env; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 DRAFTM=$R/drafter_b16_t Q37_AVL=1" ;;   # block 16 + AVL (ablation)
  bb8:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1" ;;   # BEST3: block 8, Coral drafter
  bstk:server) ARM=mtp4 TAG=$I CLIENT="bash $Q/run17.sh $I client" bash $Q/q36_arm.sh; nvidia-smi --query-gpu=memory.used --format=csv,noheader > mem_after_$I.txt ;;   # stock SGLang MTP-4
  bbest:client|bb16:client|bb8:client|bstk:client) for m in ${BMODES:-off on}; do rep $I $m $R/turns_bench.jsonl _bench; done ;;
  hotcov:server) $VQ/bin/python - $R $HOME/lab/models/Qwen3.6-35B-A3B-FP8 $S4/hotv/hot_ids_48000.pt <<'PY' | tee $R/hotcov.txt   # CPU: target tokens the 48K draft vocab can never propose
import json, sys, torch, collections
from transformers import AutoTokenizer
R, M, H = sys.argv[1:4]
tk = AutoTokenizer.from_pretrained(M); hot = set(torch.load(H).tolist())
for m in ("off", "on"):
    c, miss, top = collections.Counter(), collections.Counter(), collections.Counter()
    for l in open(f"{R}/rep_a16v_{m}_hold.jsonl", encoding="utf-8"):
        t = json.loads(l).get("text") or ""
        th, rest = t.split("</think>", 1) if "</think>" in t else (("", t) if m == "off" else (t, ""))
        for seg, txt in (("think", th), ("rest", rest)):
            ids = tk.encode(txt, add_special_tokens=False); c[seg] += len(ids)
            out = [i for i in ids if i not in hot]; miss[seg] += len(out); top.update(out)
    print("HOTCOV", m, {k: f"{c[k]} tok {100 * miss[k] / max(c[k], 1):.2f}% outside" for k in c}, "top missing", [(tk.decode([i]), n) for i, n in top.most_common(12)])
PY
    ;;
  a16a:server) export PATH=$VQ/bin:$PATH   # L-aware tiles on FlashInfer (Q37_ATTN_LSKIP): gate (block 16 and 24), then the paired arm
    for B in 16 24; do ( cd /tmp && timeout 900 python $Q/s9_attn_gate.py lskip $B 2>&1 | grep -E "^(LSKIP|S9_|Traceback)|Error" | tee $R/lskip_gate$B.txt | cut -c1-200 ); done
    { grep -q "S9_LSKIP B=16 PASS" $R/lskip_gate16.txt && echo 'LSK="Q37_ATTN_LSKIP=1"' || echo 'LSK=""'; grep -q "S9_LSKIP B=24 PASS" $R/lskip_gate24.txt && echo 'LSK24="Q37_ATTN_LSKIP=1"' || echo 'LSK24=""'; } > $R/lskip_flags.env
    . $R/lskip_flags.env; cat $R/lskip_flags.env; [ -n "$LSK" ] || { echo "lskip gate failed; skipping arm"; exit 0; }
    arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 DRAFTM=${A_DRAFT:-$R/drafter_b16_t} Q37_AVL=1 $LSK"
    grep -E "q37 (attn lskip|lskip)" server_$I.log | sort | uniq -c | head -5 > used_lskip_$I.txt; grep -q "lskip select" used_lskip_$I.txt || echo "LSKIP NOT ENGAGED" | tee -a used_lskip_$I.txt ;;
  a24v:server|a24l:server) . $R/avl24.env; [ -f $R/lskip_flags.env ] && . $R/lskip_flags.env
    # Run 2 already reaches 10.22 / 6.80 at block 24 (b24e, positions 16+ untrained): use Run 6 only if its offline mean beats that
    D24=$(python3 -c "import re,ast;m=re.search(r'(\{.*\})',open('$R/r6_gate.txt').read());d=ast.literal_eval(m.group(1)) if m else {};print('$R/drafter_b24' if d and d.get('chain_off') and (d['chain_off']+d['chain_on'])/2>(10.219+6.803)/2 else '$R/drafter_b16_t')" 2>/dev/null || echo $R/drafter_b16_t)
    [ -n "${A24_DRAFT:-}" ] && D24=$A24_DRAFT
    echo "$I drafter $D24" | tee drafter_$I.txt
    S5M=""; grep -q "S5_DENSE_PASS" $R/s5m24.txt 2>/dev/null && grep -q "equal for M=1..24" $R/s5m24.txt && S5M="Q37_S5_MMAX=24"   # Track E at M<=24
    arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=24 DFLBLOCK=24 DRAFTM=$D24 Q37_AVL=1 $AVL24 ${LSK24:-} $S5M Q37_BOLE_WARPS=solvebv:8,solve:2"   # Bole solve V tile 8, 2 warps: bitwise equal, 27.7 -> 21.6 us/layer at T=24
    grep -E "q37 s5 dense (used|fell)" server_$I.log | head -3 > used_s5_$I.txt
    grep -E "q37 (attn128|avl)" server_$I.log | head -4 > used_a128_$I.txt ;;
  a16a:client|a24v:client|a24l:client) rep $I off $R/turns_hold128.jsonl _hold; rep $I on $R/turns_hold128.jsonl _hold ;;
  div15:server) for m in off on; do A=""; for x in a16v a16a a24v a24l; do [ -f rep_${x}_${m}_hold.jsonl ] && A="$A rep_${x}_${m}_hold.jsonl"; done   # needs boot-1 rep_ref_*_hold.jsonl
      [ -f rep_ref_${m}_hold.jsonl ] && $PY $Q/diverge.py --turns $R/turns_hold128.jsonl --mode $m --ref rep_ref_${m}_hold.jsonl --arms $A --margins 48 --out div15_$m.json 2>&1 | grep -E "^(DIV|MARGIN)|Error" > div15_$m.txt; done ;;
  b8on:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1" ;;   # BEST3 block 8, thinking-on control
  avlsm:client) for m in off on; do $PY $Q/swe_replay.py --turns $R/turns_hold128.jsonl --arm $I --thinking $m --limit 8 --out rep_${I}_${m}.jsonl | tail -3 > rep_${I}_${m}.txt; done ;;
  a16c:client|a16v:client) rep $I off $R/turns_hold128.jsonl _hold; rep $I on $R/turns_hold128.jsonl _hold ;;
  b8on:client) rep $I on $R/turns_hold128.jsonl _hold ;;
  b16t:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_BLOCK=16 DFLBLOCK=16 DRAFTM=$R/drafter_b16_t" ;;
  b8c2:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1" ;;
  b16:client|b8c:client|b16t:client|b8c2:client) rep $I off $OFF/turns_dev64.jsonl _dev64; rep $I on $OFF/turns_dev64.jsonl _dev64 ;;
  ft_smoke:server|ft_gen:server) env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_FEAT_DUMP=$R/feat MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run16.sh $I client" bash $Q/q36_arm.sh
    grep -o "q37 feat dump [^\[]*" server_$I.log | tail -3 > used_$I.txt; nvidia-smi --query-gpu=memory.used --format=csv,noheader > mem_after_$I.txt ;;
  ft_smoke:client) $PY $Q/gen_client.py --turns $R/train_turns.jsonl --mode off --conc 2 --done $R/done_smoke.txt --limit-traces 2 --limit-turns 3 ;;
  ft_gen:client) $PY $Q/gen_client.py --turns $R/train_turns.jsonl --mode off --conc 8 --done $R/done_off.txt | tee -a $R/gen_off.log | tail -3
    $PY $Q/gen_client.py --turns $R/train_turns.jsonl --mode on --conc 8 --done $R/done_on.txt | tee -a $R/gen_on.log | tail -3 ;;
  ft_sanity:server) export PATH=$VQ/bin:$PATH; ( cd /tmp && python $Q/ft_train.py --feat $R/feat --turns $R/train_turns.jsonl --out $R/drafter_ft --eval-only 2>&1 | grep -E "^(files|VAL|ROUNDCHECK|Traceback)|Error" | tee $R/ft_sanity.txt ) ;;
  ft_train:server) export PATH=$VQ/bin:$PATH; ( cd /tmp && python $Q/ft_train.py --feat $R/feat --turns $R/train_turns.jsonl --out $R/${FT_OUT:-drafter_ft} --metrics $R/ft_metrics_${FT_OUT:-drafter_ft}.jsonl ${FT_ARGS:-} 2>&1 | grep --line-buffered -E "^(files|VAL|training|step|SAVED|FT_DONE|Traceback)|Error" | tee $R/ft_train.txt ) ;;
  ft_arm:server) [ -f $R/drafter_ft/model.safetensors ] || { echo "no fine-tuned drafter saved; skipping"; exit 0; }
    env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 DRAFTM=$R/drafter_ft Q37_ACCEPT_LOG=$R/accept_$I.txt MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run16.sh $I client" bash $Q/q36_arm.sh ;;
  ft_ctl:server) env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_ACCEPT_LOG=$R/accept_$I.txt MOECFG=$S4/moe_cfg_both EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run16.sh $I client" bash $Q/q36_arm.sh ;;
  ft_arm:client|ft_ctl:client) rep $I off $S4/turns_ind/turns_ind128.jsonl _ind128; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind64; rep $I off $OFF/turns_dev64.jsonl _dev64 ;;
  final_ours:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 ${FINAL_EXTRA:-}" ;;
  final_stock:server) ARM=mtp4 TAG=$I CLIENT="bash $Q/run15.sh $I client" bash $Q/q36_arm.sh; nvidia-smi --query-gpu=memory.used --format=csv,noheader > mem_after_$I.txt ;;
  final_ours:client|final_stock:client) rep $I off $S4/turns_ind/turns_ind128.jsonl _ind128; rep $I on $S4/turns_ind/turns_ind128.jsonl _ind128 ;;
  s10_best2:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128" ;;
  s10_stock:server) ARM=mtp4 TAG=$I CLIENT="bash $Q/run14.sh $I client" bash $Q/q36_arm.sh; nvidia-smi --query-gpu=memory.used --format=csv,noheader > mem_after_$I.txt ;;
  s10_best2:client|s10_stock:client) rep $I off $S4/turns_ind/turns_ind128.jsonl _ind128; rep $I off $S4/turns_ind/turns_ind64.jsonl _ind64; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind64 ;;
  s10_cfg:server) export PATH=$VQ/bin:$PATH; C=$R/cfgs; mkdir -p $C; rm -f $C/*.pt
    B='"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32'
    while IFS='|' read -r tag n w st; do
      J="{$B, \"BLOCK_SIZE_N\": $n, \"num_warps\": $w, \"num_stages\": $st}"
      ( cd /tmp && timeout 300 python $Q/s10_cfg_gate.py $HOME/lab/profiles/q36_s3/routes.jsonl $tag "$J" "$J" $C 2>&1 | grep -E "^CFG|Error" | cut -c1-220 | tee -a $R/s10_cfg.txt )
    done <<'LIST'
base|64|4|2
n32s2|32|4|2
n32s3|32|4|3
n64s3|64|4|3
n64s4|64|4|4
n128s2|128|4|2
n64w8s3|64|8|3
n128w8s3|128|8|3
LIST
    python3 - <<'PY' > $R/s10_flags.env
import re
t = open("/home/ubuntu/lab/profiles/q36_s10/s10_cfg.txt").read()
r = {m.group(1): (float(m.group(2)), int(m.group(3))) for m in re.finditer(r"CFG (\S+) us_per_layer ([0-9.]+) .*? equal (\d+)/40", t)}
base = r.get("base", (None,))[0]
ok = [(v[0], k) for k, v in r.items() if k != "base" and v[1] == 40 and base and base - v[0] >= 5.0]
print(f'S10CFG="{min(ok)[1]}"' if ok else 'S10CFG=""')
PY
    cat $R/s10_flags.env ;;
  s10_cfgarm:server) . $R/s10_flags.env; [ -n "${S10CFG:-}" ] || { echo "no config passed the in-graph gate"; exit 0; }
    env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_ACCEPT_LOG=$R/accept_$I.txt MOECFG=$R/cfgs/$S10CFG EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run14.sh $I client" bash $Q/q36_arm.sh
    grep -o "Using MoE kernel config from [^ ]*" server_$I.log | sort -u | head -3 > used_$I.txt ;;
  s10_ovl:server) . $R/s10_flags.env; M=$S4/moe_cfg_both; [ -n "${S10CFG:-}" ] && M=$R/cfgs/$S10CFG
    env $BEST Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1 Q37_ACCEPT_LOG=$R/accept_$I.txt MOECFG=$M EXTRA="$WIN" TAG=$I CLIENT="bash $Q/run14.sh $I client" bash $Q/q36_arm.sh
    grep -o "q37 commit overlap [a-z]*" server_$I.log | sort | uniq -c > used_$I.txt ;;
  s10_close:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128" ;;
  s10_close:client) rep $I off $S4/turns_ind/turns_ind128.jsonl _ind128 ;;
  s10_ovl:client) rep $I off $S4/turns_ind/turns_ind128.jsonl _ind128; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind64 ;;
  s10_cfgarm:client) rep $I off $S4/turns_ind/turns_ind128.jsonl _ind128; rep $I off $S4/turns_ind/turns_ind64.jsonl _ind64; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind64 ;;
  s9_gate:server) export PATH=$VQ/bin:$PATH
    ( cd /tmp && timeout 600 python $Q/s9_attn_gate.py 2>&1 | grep -E "^(ATTN|S9_|Traceback)|Error" | tee $R/s9_gate.txt | cut -c1-230 )
    X=$(grep -oE "saves_us_per_target_layer -?[0-9.]+" $R/s9_gate.txt | awk '{print $2}')
    if [ -n "$X" ] && python3 -c "import sys; sys.exit(0 if float('$X') >= 5.0 else 1)"; then echo 'S9ATT="Q37_ATTN_ONEWAVE=1"' > $R/s9_flags.env; else echo 'S9ATT=""' > $R/s9_flags.env; fi
    cat $R/s9_flags.env ;;
  s9_b2:server|s9_b2b:server) . $R/s9_flags.env; [ -n "${S9ATT:-}" ] || { echo "attention gate failed; no control needed"; exit 0; }; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128" ;;
  s9_a1:server) . $R/s9_flags.env; [ -n "${S9ATT:-}" ] || { echo "attention gate failed; skipping"; exit 0; }; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 $S9ATT" ;;
  s9_b2:client|s9_a1:client) rep $I off $OFF/turns_dev64.jsonl; rep $I off $S4/turns_ind/turns_ind128.jsonl _ind; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind ;;
  s9_b2b:client) rep $I off $OFF/turns_dev64.jsonl ;;
  s8_gate:server) export PATH=$VQ/bin:$PATH
    ( cd $Q/s5_kernels && rm -f s7_moe.so && nvcc -std=c++17 -O3 -arch=sm_120a -w -shared -Xcompiler -fPIC -cudart shared -o s7_moe.so s7_moe.cu ) 2>&1 | grep -iE "error" | cut -c1-200
    ( cd /tmp && SGLANG_MOE_CONFIG_DIR=$S4/moe_cfg_both timeout 1200 python $Q/s8_moe_graph.py $HOME/lab/profiles/q36_s3/routes.jsonl 2>&1 | grep -E "^(WEIGHTS|DISTINCT|GRAPH|S8_|Traceback)|Error" | tee $R/s8_gate.txt | cut -c1-200 )
    P=$(grep -oE "S8_PICK [0-9]+ [0-9]+ saves_us -?[0-9.]+" $R/s8_gate.txt); set -- $P
    if [ -n "$P" ] && python3 -c "import sys; sys.exit(0 if float('$5') >= 5.0 else 1)"; then echo "S8MOE=\"Q37_S7_MOE=1 Q37_S7_ST=$2 Q37_S7_GRID=$3\"" > $R/s8_flags.env; else echo 'S8MOE=""' > $R/s8_flags.env; fi
    cat $R/s8_flags.env ;;
  s8_b2:server|s8_b2b:server) . $R/s8_flags.env; [ -n "${S8MOE:-}" ] || { echo "MoE v3 did not win in-graph; no control needed"; exit 0; }; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128" ;;
  s8_m3:server) . $R/s8_flags.env; [ -n "${S8MOE:-}" ] || { echo "MoE v3 did not win in-graph; skipping"; exit 0; }; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 $S8MOE" ;;
  s8_b2:client|s8_m3:client) rep $I off $OFF/turns_dev64.jsonl; rep $I off $S4/turns_ind/turns_ind128.jsonl _ind; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind ;;
  s8_b2b:client) rep $I off $OFF/turns_dev64.jsonl ;;
  s7_b2:server|s7_b2b:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128" ;;
  s7_gate:server) export PATH=$VQ/bin:$PATH
    ( cd $Q/s5_kernels && rm -f s5_moe.so && nvcc -std=c++17 -O3 -arch=sm_120a -w -shared -Xcompiler -fPIC -cudart shared -o s5_moe.so s5_moe.cu ) 2>&1 | grep -iE "error" | cut -c1-200
    ( cd /tmp && SGLANG_MOE_CONFIG_DIR=$S4/moe_cfg_both timeout 1200 python $Q/s7_moe_graph.py $HOME/lab/profiles/q36_s3/routes.jsonl 2>&1 | grep -E "^(WEIGHTS|DISTINCT|GRAPH|S7_|Traceback)|Error" | tee $R/s7_gate.txt | cut -c1-200 )
    P=$(grep -oE "S7_PICK [0-9]+ [0-9]+ saves_us -?[0-9.]+" $R/s7_gate.txt)
    set -- $P
    if [ -n "$P" ] && python3 -c "import sys; sys.exit(0 if float('$5') >= 5.0 else 1)"; then echo "S7MOE=\"Q37_S5_MOE=1 Q37_S5_MOE_NB=$2 Q37_S5_MOE_GRID=$3\"" > $R/s7_flags.env; else echo 'S7MOE=""' > $R/s7_flags.env; fi
    cat $R/s7_flags.env ;;
  s7_m2:server) . $R/s7_flags.env; [ -n "${S7MOE:-}" ] || { echo "MoE did not beat stock in-graph; skipping"; exit 0; }; arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 $S7MOE" ;;
  s7_seq:server) arm $I "Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_SHARED_SEQ=1" ;;
  s7_seq:client|s7_m2:client) rep $I off $OFF/turns_dev64.jsonl; rep $I off $S4/turns_ind/turns_ind128.jsonl _ind; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind ;;
  s7_b2:client) rep $I off $OFF/turns_dev64.jsonl; rep $I off $S4/turns_ind/turns_ind128.jsonl _ind; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind ;;
  s7_b2b:client) rep $I off $OFF/turns_dev64.jsonl ;;
  s6_ctl:server|s6_ctl2:server) arm $I ;;
  s6_m1:server) [ -n "${S6MOE:-}" ] || { echo "MoE kernels did not pass the gate; skipping"; exit 0; }; arm $I "${S6:-} $S6MOE" ;;
  s6_e1:server) [ -n "${S6:-}" ] || { echo "no Track E component passed the gate; skipping"; exit 0; }; arm $I "$S6" ;;
  s6_ctl:client|s6_e1:client|s6_m1:client) rep $I off $OFF/turns_dev64.jsonl; rep $I off $S4/turns_ind/turns_ind128.jsonl _ind; rep $I on $S4/turns_ind/turns_ind64.jsonl _ind ;;
  s6_ctl2:client) rep $I off $OFF/turns_dev64.jsonl ;;
  n16:server|n8:server|n16v:server|n16l:server|n24l:server|n24f:server) cp $OFF/turns_dev64.jsonl turns_dev64.jsonl; V=$VQ; X="Q37_S5_DENSE=64 Q37_S5_BA=128 Q37_COMMIT_OVERLAP=1"; B=""
    [ $I = n16l ] && { X="$X Q37_BLOCK=16 Q37_AVL=1 Q37_ATTN_LSKIP=1"; B="DFLBLOCK=16 DRAFTM=$R/drafter_b16_t"; }
    [ $I = n24l ] && { . $R/avl24.env; X="$X Q37_BLOCK=24 Q37_AVL=1 $AVL24 Q37_ATTN_LSKIP=1"; B="DFLBLOCK=24 DRAFTM=$R/drafter_b24"; }
    [ $I = n24f ] && { . $R/avl24.env; X="$X Q37_BLOCK=24 Q37_AVL=1 $AVL24 Q37_ATTN_LSKIP=1 Q37_S5_MMAX=24 Q37_BOLE_WARPS=solvebv:8,solve:2"; B="DFLBLOCK=24 DRAFTM=${A24_DRAFT:-$R/drafter_b24}"; }
    [ $I = n16 ] && { X="$X Q37_BLOCK=16"; B="DFLBLOCK=16 DRAFTM=$R/drafter_b16_t"; }
    [ $I = n16v ] && { X="$X Q37_BLOCK=16 Q37_AVL=1"; B="DFLBLOCK=16 DRAFTM=$R/drafter_b16_t"; }
    # Session 6 nsys recipe with BEST3 flags; block 16 vs block 8, same three regimes (bs1_profile.py)
    P="sudo env PATH=$V/bin:$PATH HOME=$HOME PYTHONPATH=$E Q37_NVTX=1 Q37_SYNCFREE=1 Q37_KV_FUSE=1 Q37_CONV_TILE=64,4 Q37_BA_EXACT=128,1024 Q37_KV_PAR=64 Q37_TRACK_GUARD=1 $X Q37_HOTVOCAB=$S4/hotv/hot_ids_48000.pt SGLANG_MOE_CONFIG_DIR=$S4/moe_cfg_both NSYS_TMPDIR=$R/nsys_tmp nsys profile -o $R/nsys/$I --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=repeat:3 --trace-fork-before-exec=true"
    env $BEST $X $B MOECFG=$S4/moe_cfg_both EXTRA="$WIN" KILLWAIT=5 PREFIX="$P" TAG=$I CLIENT="$PY $Q/bs1_profile.py --out nsys/regimes_$I.jsonl; sleep 5; sudo pkill -INT -f '[s]glang.launch_server'; sleep 40" bash $Q/q36_arm.sh
    sudo chown -R ubuntu:ubuntu $HOME/.cache $HOME/.triton $R 2>/dev/null
    for n in 1 2 3; do nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output /dev/null nsys/$I.$n.nsys-rep > /dev/null 2>&1; done
    for n in 1 2 3; do [ -f nsys/$I.$n.sqlite ] && { echo "== $I regime $n"; python3 $Q/crit_path.py nsys/$I.$n.sqlite | head -30; }; done > crit_$I.txt; head -32 crit_$I.txt ;;
  s6_nsys:server) cp $OFF/turns_dev64.jsonl turns_dev64.jsonl; V=$VQ; X="${S6:-} ${S6MOE:-}"
    # Session 4 nsys_c recipe (produced the cx traces), plus the Session 6 flags.
    P="sudo env PATH=$V/bin:$PATH HOME=$HOME PYTHONPATH=$E Q37_NVTX=1 Q37_SYNCFREE=1 Q37_KV_FUSE=1 Q37_CONV_TILE=64,4 Q37_BA_EXACT=128,1024 Q37_KV_PAR=64 Q37_TRACK_GUARD=1 $X Q37_HOTVOCAB=$S4/hotv/hot_ids_48000.pt SGLANG_MOE_CONFIG_DIR=$S4/moe_cfg_both NSYS_TMPDIR=$R/nsys_tmp nsys profile -o $R/nsys/e1 --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none --gpu-metrics-devices=0 --gpu-metrics-set=gb20x --gpu-metrics-frequency=100000 --capture-range=cudaProfilerApi --capture-range-end=repeat:3 --trace-fork-before-exec=true"
    env $BEST MOECFG=$S4/moe_cfg_both EXTRA="$WIN" KILLWAIT=5 PREFIX="$P" TAG=$I CLIENT="$PY $Q/bs1_profile.py --out nsys/regimes_e1.jsonl; sleep 5; sudo pkill -INT -f '[s]glang.launch_server'; sleep 40" bash $Q/q36_arm.sh
    sudo chown -R ubuntu:ubuntu $HOME/.cache $HOME/.triton $R 2>/dev/null
    for n in 1 2 3; do nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output /dev/null nsys/e1.$n.nsys-rep > /dev/null 2>&1; done
    for n in 1 2 3; do [ -f nsys/e1.$n.sqlite ] && python3 $Q/crit_path.py nsys/e1.$n.sqlite | head -25; done > crit_e1.txt; head -30 crit_e1.txt ;;
esac
