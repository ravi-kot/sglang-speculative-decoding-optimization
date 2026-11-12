# Session 13 scoreboard: block 16, adaptive verification, drafter diagnostics (RTX PRO 6000, 2026-11-10)

All server numbers: Qwen3.6-35B-A3B-FP8, BS1 greedy, max_tokens 1024, `turns_hold128.jsonl` (128 turns, 32 per context bin,
from the 45 traces never used for drafter training or checkpoint selection). tok/s = MLCommons-bin-reweighted median decode
tok/s (swe_replay). Paired = same turns, per-turn ratio. Archive: `q36_ft2_s13.tgz` (sha256 48aca0aa...7347).

## Throughput (held-out, paired, same boot)

| Config | Thinking off tok/s | Thinking on tok/s | Tokens/round off / on |
|---|---|---|---|
| Stock SGLang 0.5.21 MTP-4 | 330 | (partial run) | – |
| BEST3: block 8, Coral drafter | – | 631 | – / 5.58 |
| Block 16, Run 2 drafter | 891 | 653 | 10.24 / 7.63 |
| **Block 16, Run 2, `Q37_AVL=1`** | **912** | **691** | 10.13 / 7.39 |

Paired ratios:
- AVL vs block-16 control: off median 1.033, geo 1.042 (-0.47 ms/round); on median 1.091, geo 1.086 (-1.30 ms/round).
- AVL vs block-8 BEST3, thinking on: median 1.111, geo 1.127. Block-16 control vs block 8, on: median 0.996, geo 1.038.
- Block 16 vs stock MTP-4, thinking off: geo 2.60x; AVL vs stock (reweighted medians): 2.76x.

## Correctness vs target-only greedy (no speculation), 64 held-out turns per mode

| Config | Off identical | On identical | Median margin at divergence | Malformed tool calls |
|---|---|---|---|---|
| Stock MTP-4 | 44/64 | 18/34 (partial) | 0.25 / 0.125 nats | 0 |
| Block 8 BEST3 | – | 32/64 | 0.125 | 0 |
| Block 16 | 45/64 | 31/64 | 0.125 | 0 |
| Block 16 + AVL | 46/64 | 31/64 | 0.125 | 0 |

Every speculative configuration (stock included) diverges from M=1 target-only decoding at numerical near-ties; at the
divergence point the target's own prefill picks either token about equally often. AVL vs block-16 control: 118/128 (off),
89/128 (on) identical (moved round boundaries change rounding, Track S).

## Adaptive verification length (`Q37_AVL`, ours)

Per round, L = argmax_L E[useful tokens | L] - 0.75 * MoE_ms(L) from the drafter's calibrated path probabilities; verify rows
>= L are remapped onto row 0's experts (no new expert weights), drafts >= L masked before acceptance.
Measured MoE ms/round (graph, real routing, `moe_real.py --remap`): L=16 4.80, 14 4.37, 12 4.06, 10 3.73, 8 3.39, 6 2.87,
4 2.31, 2 1.68. Simulation (`adaptive_sim.py`, measured curve, pooled calibration): +6.5% off, +11.7% on; server: +4.2% / +8.6%
(geo). Nsight: AVL overhead about 0.08 ms/round (40 remap kernels); MoE 6.15 -> 3.23 ms/round in the 101K regime.

## Attention at block > 8 (FlashInfer fa2, head_dim 256, GQA 8)

Not redundant DRAM reads: at block 16 the second 64-row query tile hits L2 (hit rate 10.5% -> 50%, DRAM bytes unchanged,
~65 MB at 101K). Work-list interleaving: no change (bitwise equal). The +35-57% kernel time is per-SM latency (2 CTAs/SM,
twice the work per CTA). Only a 128-row kernel of our own would recover it (~0.3 ms/round at long context).

## Drafter diagnostics (offline chain tokens/round, held-out validation traces, off / on)

| Drafter | B8 | B10 | B12 | B14 | B16 |
|---|---|---|---|---|---|
| Coral | 6.58 / 5.41 | 7.48 / 5.94 | 8.11 / 6.26 | 8.44 / 6.40 | 8.39 / 6.39 |
| Run 1 | 6.44 / 5.19 | 7.34 / 5.66 | 8.04 / 5.95 | 8.66 / 6.16 | 9.21 / 6.29 |
| Run 2 | 6.57 / 5.39 | 7.54 / 5.90 | 8.25 / 6.26 | 8.96 / 6.50 | 9.53 / 6.66 |

- Generalization gap: Run 2 on 4 training traces 9.39 / 6.95 vs held-out 9.53 / 6.66 (none).
- Capacity probe (Run 2 + 372 steps on 4 traces, lr 1e-4): those traces 11.27 / 8.34, held-out 9.31 / 6.44 -> memorization;
  data-diversity limited, not capacity limited.
- Run 3a (hard post-rejection anchors, `--hard-frac 0.5`, 995 steps): 9.52 / 6.64 -> no gain; not objective limited.
- Segment split (Run 2, held-out, chain): thinking-on reasoning 5.22 tok/round, tool calls 8.31, text 8.26; off tool 9.58.
- Next: Run 3b = Run 2 + 120 independent Workato traces (target-generated labels, reasoning + tool calls, 4x sampling weight).

## Reproduce

`q36/run17.sh` items: `a16c`, `a16v`, `b8on`, `stk` (servers); `ref` (target-only + `diverge.py`); `n16`, `n8`, `n16v` (Nsight);
`abl` (ablation); `g2_wk` + `f2_train` (Run 3b). Plots: `q36/blog_plots.py`.

## Boot 2 (2026-11-10, 09:53-13:xx UTC): training levers and AVL tuning

Offline chain tokens/round on held-out validation traces (off / on, reasoning segment on):

| Drafter | Change vs Run 2 (all warm-started from Run 2, teacher KL 2.0, lr 3e-5, ~800-1000 steps) | Off | On | Reasoning |
|---|---|---|---|---|
| Run 2 | baseline | 9.50 | 6.64 | 5.19 |
| Run 3a | hard post-rejection anchors | 9.52 | 6.64 | 5.20 |
| Run 3b | + 120 Workato traces (4x) | 9.49 | 6.66 | 5.21 |
| Run 4 | + 140 in-domain SWE-rebench trajectories (4x) | 9.57 | 6.65 | 5.19 |
| Run 5c | Run 4 + confidence-weighted loss (2.0) | 9.60 | 6.64 | 5.20 |

Diagnostics: 47% of misses are on target-confident tokens (p >= 0.9, 5.5% miss); confident-miss rate by draft position
2.6% (pos 1) -> ~6% (pos 5-13) -> 7.8% (pos 15); 16K / 32K drafter window: 9.28 / 8.88 off vs 9.39 at 8K (no gain).
AVL thinking-off tuning (paired boot, hold128): default 921, min L 4 918, lambda 0.5 909 tok/s -> default kept.
Archive: `q36_ft2_s14.tgz` (sha256 cb0a94b1...e090).

## Sessions 15-16 (2026-11-10/11): L-skip attention, block 24 (Run 6), exact block-24 fixes

Same-boot paired on turns_hold128 vs control a16v (block 16 + AVL, Run 2): control 924 off / 693 on (rw tok/s).

| Arm | Off rw | On rw | Paired off (median / geo) | Paired on (median / geo) |
|---|---|---|---|---|
| a16a: + L-skip (FlashInfer plan variant per AVL L, `Q37_ATTN_LSKIP`) | 933 | (contaminated*) | 1.008 / 1.015 | 1.005 (s15: 1.008 / 1.004) |
| **a24l: block 24 + Run 6 + L-skip + s5 dense M<=24 + Bole solve BV8/2 warps** | **934** | **702.5** | **1.016 / 1.028** | **1.008 / 1.018** |

*diverge.py --margins queried the live a16a server during its last (gt128k, thinking-on) turns: those turns stalled; median unaffected.
- Run 6 = Run 2 -> block 24, teacher KL 2.0 to Run 2 at block 16, 1,244 steps: chain 10.51 / 6.91 (Run 2 block 16: 9.50 / 6.64; Run 2 untrained at block 24: 10.22 / 6.80); reasoning 5.44 vs 5.19. Still rising at the end.
- Block-24 step overhead vs block 16: +1.90 ms (plain) -> +1.56 (L-skip) -> +1.29 off / +0.76 on (all fixes). Nsight: Bole solve +0.39 ms (T padded 24->32), stock dense split-K reduces +0.44 ms, BA / conv fallbacks.
- Exact fixes: s5 narrow dense GEMM bitwise equal to stock for M=1..24 (Q37_S5_MMAX=24); Bole solve V tile 8 + 2 warps bitwise equal, 27.7 -> 21.6 us/layer at T=24 (Q37_BOLE_WARPS=solvebv:8,solve:2).
- Own Triton 128-row verify attention (attn128.py): correct, 1.5-2x slower than FlashInfer fa2 -> rejected.
- Correctness vs target-only greedy (64 turns): a24l 50/64 off, 31/64 on; a16a 48/64, 32/64; control 46/64, 31/64; 0 malformed.
- Hot-vocab coverage: 0.36% of thinking-on reasoning tokens outside the 48K draft vocab (not a limiter).
- Run 6b (continue training) hung on a corrupt feature file (producer thread); ft2_train now skips unreadable files.
Archives: q36_ft2_s15.tgz (dac6e62f...), q36_ft2_s16.tgz (4d8b0aa6...).

## Session 17 (2026-11-11, bounded boot): reproduction, Run 6c, engineering stop
- Control a16v 922 off / 697 on; best (block 24 + Run 6 + AVL + L-skip + s5 M<=24 + Bole BV8) 934 / 702.5 (reproduces s16).
  Paired vs control: off median 1.021 / geo 1.028 (69/128 faster); on 1.009 / 1.016 (70/128). Correctness 50/64 off, 31/64 on, 0 malformed.
- Run 6c (Run 6 + 0.4 epoch, new data order): 10.49 / 6.92 vs Run 6 10.51 / 6.91 -> drafter saturated for this recipe.
- Nsight n24f vs n16l: Bole solve +0.19-0.21 ms (was +0.39), split-K reduce +0.14 (was +0.37), conv +0.13, cuBLAS fallbacks.
- Stop rule (< 960 off) met -> engineering ends. Benchmarks: turns_bench.jsonl (512 turns, 128/bin, 45 sessions, disjoint from hold128).
Archive: q36_ft2_s17.tgz (b8352dd1...). Corrupt feature file: feat_swe/2d6e153bd553_on.pt (skipped).
