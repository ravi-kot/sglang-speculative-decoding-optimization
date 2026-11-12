"""E3 fallback: isolated reproducer for the Bole decode kernels at the production shapes measured in E2
(Qwen3.5-4B: 24 GDN layers, HV 32, HK 16, DK=DV 128; batch 24 (cap 26 graphs), draft T=8, bf16 state with
136 slots, as in arm F), because in-server ncu filters spent their launch budget on warm-up kernels.
Runs 3 warm verify+commit rounds, then 2 profiled rounds (NVTX range "profiled") with:
  - the request-slot commit (all rows valid), and
  - the radix track-slot commit with only 2 of 24 rows valid (the common case in E2), to measure what the
    masked-but-not-skipped rows cost.
Usage: sudo ncu --nvtx --nvtx-include "profiled/" ... python bole_repro.py"""
import torch

from sglang.srt.layers.attention.linear.bole_spec import BoleSpecVerifier

torch.manual_seed(0)
DEV = "cuda"
import os
BS = int(os.environ.get("BS", 24))
L, SLOTS, HV, HK, D, T = int(os.environ.get("BL", 24)), int(os.environ.get("BSLOTS", 5 * BS + 16)), 32, 16, 128, int(os.environ.get("BT_T", 8))  # Phase 4: BS from env (24 = Phase 3 shape); session 4: Qwen3.6 layers, slots
if os.environ.get("Q37_BOLE_WARPS"):
    import sys
    sys.path.insert(0, os.path.expanduser("~/lab/env/q36"))
    import q37_nvtx
    q37_nvtx.install_bole_warps(os.environ["Q37_BOLE_WARPS"])
temporal = (torch.randn(L, SLOTS, HV, D, D, device=DEV) * 0.05).to(torch.float32 if os.environ.get("BFP32") else torch.bfloat16)
ver = BoleSpecVerifier(temporal, max_bs=BS + 1)
cu = torch.arange(0, (BS + 1) * T, T, device=DEV, dtype=torch.int32)
slots = torch.arange(1, BS + 1, device=DEV, dtype=torch.int32)
steps = (torch.arange(BS, device=DEV, dtype=torch.int32) % T)
track_slots = torch.full((BS,), -1, device=DEV, dtype=torch.int32)
track_steps = torch.full((BS,), -1, device=DEV, dtype=torch.int32)
track_slots[:min(2, BS)] = torch.tensor([100, 101][:min(2, BS)], device=DEV, dtype=torch.int32)
track_steps[:min(2, BS)] = torch.tensor([3, 5][:min(2, BS)], device=DEV, dtype=torch.int32)
A_log = torch.randn(HV, device=DEV) * 0.5
dt_bias = torch.randn(HV, device=DEV) * 0.5


def round_():
    for layer in range(L):
        mixed = torch.randn(1, BS * T, 2 * HK * D + HV * D, device=DEV, dtype=torch.bfloat16)
        q = mixed[..., : HK * D].view(1, BS * T, HK, D)
        k = mixed[..., HK * D: 2 * HK * D].view(1, BS * T, HK, D)
        v = mixed[..., 2 * HK * D:].view(1, BS * T, HV, D)
        ab = torch.randn(BS * T, 2 * HV, device=DEV, dtype=torch.bfloat16)
        ver.target_verify(layer, A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=ab[:, :HV], b=ab[:, HV:],
                          ssm_states=temporal[layer], cache_indices=slots, query_start_loc=cu)
    ver.commit(BS, slots, steps, track_slots, track_steps)


if os.environ.get("BTIME"):   # isolated verify timing (one layer's kernels, graph-free, 200 calls) + output checksum for warp variants
    mixed = torch.randn(1, BS * T, 2 * HK * D + HV * D, device=DEV, dtype=torch.bfloat16)
    q = mixed[..., : HK * D].view(1, BS * T, HK, D); k = mixed[..., HK * D: 2 * HK * D].view(1, BS * T, HK, D)
    v = mixed[..., 2 * HK * D:].view(1, BS * T, HV, D); ab = torch.randn(BS * T, 2 * HV, device=DEV, dtype=torch.bfloat16)
    f = lambda: ver.target_verify(0, A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=ab[:, :HV], b=ab[:, HV:],
                                  ssm_states=temporal[0], cache_indices=slots, query_start_loc=cu)
    o = f(); torch.cuda.synchronize()
    if os.environ.get("BSAVE"):
        torch.save(o.detach().cpu(), os.environ["BSAVE"])
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): f()
    for _ in range(3): g.replay()
    s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); s0.record()
    for _ in range(10): g.replay()
    e0.record(); torch.cuda.synchronize()
    print(f"BOLE_TIME T={T} warps={os.environ.get('Q37_BOLE_WARPS', 'default')}: {s0.elapsed_time(e0) / 200 * 1e3:.2f} us per layer verify; "
          f"checksum {o.float().abs().sum().item():.6e} {o.float().view(-1)[:4].tolist()}", flush=True)
    raise SystemExit
for _ in range(3):
    round_()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("profiled")
for _ in range(2):
    round_()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print("bole repro done")
