# Blog data pack: Qwen3.6-35B-A3B-FP8 speculative decoding on one RTX PRO 6000 (data only)

All numbers: BS1, greedy, max_tokens 1024, MLCommons agentic coding turns, thinking off and on. Speed = MLCommons-bin-reweighted
median decode tok/s (bins <32K 2614, 32–64K 2272, 64–128K 5852, >128K 5039), turns with < 32 output tokens excluded.
95% CIs = trace-clustered bootstrap (2,000 replicates; turns of one trace resampled together). Paired ratios use the same turns.

## Headline benchmark: `bench/` (512 independent held-out turns)
- Turn set: `turns_bench.jsonl`, 128 per bin, 45 sessions never used for drafter training or checkpoint selection, disjoint from
  `turns_hold128.jsonl` (which was used for configuration choices). Built with `build_official.py --seed 17 --per-bin 128 --per-trace 8`.
- Arms (`bench/raw/rep_<arm>_<mode>_bench.jsonl`, one JSON line per turn, incl. output text):
  `bstk` stock SGLang 0.5.21 MTP-4; `bb8` BEST3 (block 8, Coral drafter); `bb16` block 16 + AVL (Run 2 drafter);
  `bbest` block 24 + AVL + L-skip + s5 dense M<=24 + Bole solve BV8 (Run 6 drafter).
- `bench/figs/bench_tables.md`, `bench.csv`, `bench.json`: tok/s with CI, ratio vs stock (CI), per-turn geometric mean, ratio vs the
  previous arm (CI), tokens per round, step ms, tok/s by bin. Figures: `bench_throughput.*`, `bench_speedup_by_context.*`.
- Same session (boot of 2026-11-11 08:05–12:01 UTC), sequential arms, one instance. Archive `bench/bench_final.tgz`
  (sha256 ceec57a8...113c).

| Arm | Off tok/s (95% CI) | On tok/s (95% CI) | Off vs stock | On vs stock |
|---|---|---|---|---|
| Stock MTP-4 | 335 (329–340) | 319 (313–323) | – | – |
| Block 8 (BEST3) | 735 (718–750) | 635 (620–647) | 2.20x (2.14–2.25) | 1.99x (1.96–2.03) |
| Block 16 + AVL | 937 (896–968) | 709 (686–721) | 2.80x (2.70–2.87) | 2.23x (2.17–2.27) |
| Block 24 + all (best) | 975 (932–1003) | 717 (691–748) | 2.91x (2.81–2.99) | 2.25x (2.19–2.34) |

## Engineering experiments on `turns_hold128.jsonl` (128 turns, same-boot paired controls)
- `hold128_experiments.csv`: every server experiment vs its same-boot control (ratio of reweighted medians with CI, per-turn median
  and geometric-mean ratios, turns faster, tokens/round, step ms). Sessions s13–s17; notes flag partial / contaminated runs.
- `correctness.csv`: agreement with target-only greedy decoding (64 held-out turns per mode): identical outputs, first divergent
  token, tool calls, malformed tool calls. Divergences are numerical near-ties (see scoreboard: median margin 0.125 nats).

## Drafter
- `drafter_offline.csv`: offline chain evaluation (server-like rounds) on held-out validation traces, tokens per round off / on,
  thinking-on reasoning / tool segments, best checkpoint of each training run (Run 1–6c) and eval-only block sweeps (Coral, Run 1, Run 2).

## Engine
- `avl_moe_cost.csv`: measured MoE ms per round vs verify length L (graph-timed, real routing) for the M=16 and M=24 graphs.
- `nsys_kernels.csv`: Nsight Systems critical path per configuration and context regime (~18K / ~101K / ~165K), ms per step by kernel.
- `gates.csv`: isolated kernel tests: L-skip plan variants vs full plan, our Triton 128-row attention (rejected: slower than
  FlashInfer), s5 narrow dense GEMM at M=24 (bitwise equal), Bole solve tuning (bitwise equal).

## Provenance
- `raw/s13`–`raw/s17`: per-session replay files, logs and gate outputs extracted from `../q36_ft2_s1*.tgz` (checksums in
  `scoreboard16.md`). `scoreboard16.md`: running engineering log with all session results.
- `scripts/hold_ablation.py <blog_data>` and `scripts/pack_tables.py <blog_data>` regenerate the CSVs from `raw/`;
  `../../q36/blog_plots.py <dir> bench` regenerates `bench/figs` from the bench replay files.
- Not measured: correctness vs target-only on the 512-turn set (only on 64 hold128 turns per mode); TTFT / end-to-end latency are
  in the replay files (`ttft_s`) but not summarized.
