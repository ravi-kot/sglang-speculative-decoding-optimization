"""Isolated draft output-head (tied lm_head, [vocab, hidden]) projection: BF16 vs FP8 copies.

  python profiling/head_bench.py collect configs/local.yaml      # real head inputs from the server
  python profiling/head_bench.py bench   configs/local.yaml      # graph-replay timing + accuracy + memory
  python profiling/head_bench.py profile configs/local.yaml      # one call per variant inside cudaProfilerStart/Stop (for ncu)

Variants (weights quantized once, outside timing; BF16 target head is untouched):
  bf16          torch.matmul(x, W.T)  - what SGLang's LogitsProcessor runs today (cuBLAS GEMV at M=1)
  w8a8_torch    per-token FP8 activation quant + torch._scaled_mm, per-row weight scales (vLLM #59973 "fp8")
  w8a8_sgl      sgl_kernel per-token quant + sgl_kernel.fp8_scaled_mm (CUTLASS), per-row weight scales
  w8a16_marlin  FP8 weights, BF16 activations, SGLang FP8 Marlin (weight-only, per-row scales)
Head inputs: final-norm hidden states returned by SGLang (--enable-return-hidden-states) while it
generates greedily on our code/MT-Bench prompts; self-checked by argmax(h @ W.T) == generated token.
These are target-model states, used as a proxy for MTP draft states (same tied head, same norm).
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import bench  # noqa: E402  (reuse server lifecycle, prompts, VRAM gate)

CACHE = REPO / ".cache"
HID = CACHE / "head_hidden.pt"
F8 = torch.float8_e4m3fn


def load_head(cfg):
    from safetensors import safe_open

    for f in sorted(Path(cfg["model"]).glob("*.safetensors")):
        with safe_open(str(f), "pt", device="cuda") as st:
            for k in st.keys():
                if k.endswith("embed_tokens.weight") and "mtp" not in k and "visual" not in k:
                    return st.get_tensor(k).to(torch.bfloat16), k
    raise SystemExit("embed_tokens.weight not found")


# ---------------- collect real head inputs ----------------
def collect(cfg, a):
    used, free = bench.gpu_used_free()
    assert free >= cfg.get("min_free_mib", 0), f"VRAM gate: {free} MiB free"
    out = REPO / "experiments" / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-head-collect")
    out.mkdir(parents=True)
    srv = bench.Server(cfg, "A", ["--enable-return-hidden-states"], out / "server_A.log")
    hs, toks = [], []
    try:
        srv.start(cfg.get("startup_timeout_s", 900))
        for name, n in (("agentic_code_1024", 6), ("mtbench", 8)):
            for ids in bench.load_set(name)[:n]:
                r = bench.http(srv.port, "/generate", {"input_ids": ids, "return_hidden_states": True,
                                                     "sampling_params": {"max_new_tokens": 96, "temperature": 0}})
                rows = []
                for step in r["meta_info"]["hidden_states"]:  # prefill step: [T, D] (or [D]); decode: [D]
                    t = torch.tensor(step, dtype=torch.float32)
                    rows.append(t if t.dim() == 1 else t[-1])  # keep the last position = next-token state
                k = min(len(rows), len(r["output_ids"]))
                hs += rows[:k]
                toks += r["output_ids"][:k]
                if not hs[k:]:
                    print(f"  format: {len(r['meta_info']['hidden_states'])} steps for {len(r['output_ids'])} tokens")
    finally:
        srv.stop()
    H = torch.stack(hs)
    torch.save({"h": H, "tok": torch.tensor(toks)}, HID)
    print(f"collected {H.shape[0]} head inputs (dim {H.shape[1]}) -> {HID}  [server log: {out.name}]")


# ---------------- variants ----------------
def make_variants(W, which):
    N, K = W.shape
    ws = (W.float().abs().amax(1, keepdim=True) / 448).clamp(min=1e-12)  # per-row (per output channel)
    Wq = (W.float() / ws).to(F8)
    v, mem = {}, {"bf16": W.numel() * 2}
    v["bf16"] = lambda x: torch.matmul(x, W.T)

    def tok_quant(x):  # per-token dynamic activation scale (part of the timed call)
        s = (x.abs().amax(1, keepdim=True).float() / 448).clamp(min=1e-12)
        return (x.float() / s).to(F8), s

    if "w8a8_torch" in which:
        WqT, wsT = Wq.t(), ws.t().contiguous()

        def w8a8_torch(x):
            xq, xs = tok_quant(x)
            return torch._scaled_mm(xq, WqT, scale_a=xs, scale_b=wsT, out_dtype=torch.bfloat16)

        v["w8a8_torch"], mem["w8a8_torch"] = w8a8_torch, Wq.numel() + ws.numel() * 4
    if "w8a8_sgl" in which:
        from sgl_kernel import fp8_scaled_mm, sgl_per_token_quant_fp8

        WqT, ws1 = Wq.t(), ws.reshape(-1).contiguous()

        def w8a8_sgl(x):
            xq = torch.empty_like(x, dtype=F8)
            xs = torch.empty(x.shape[0], 1, device=x.device, dtype=torch.float32)
            sgl_per_token_quant_fp8(x, xq, xs)
            return fp8_scaled_mm(xq, WqT, xs, ws1, torch.bfloat16)

        v["w8a8_sgl"], mem["w8a8_sgl"] = w8a8_sgl, Wq.numel() + ws.numel() * 4
    if "w8a16_marlin" in which:
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (apply_fp8_marlin_linear,
                                                                     prepare_fp8_layer_for_marlin)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(Wq.clone(), requires_grad=False)  # (N, K)
        layer.weight_scale = torch.nn.Parameter(ws.reshape(-1).clone(), requires_grad=False)
        layer.output_size_per_partition, layer.input_size_per_partition = N, K
        layer.orig_dtype, layer.bias = torch.bfloat16, None
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        v["w8a16_marlin"] = lambda x: apply_fp8_marlin_linear(x, layer.weight, layer.weight_scale,
                                                              layer.workspace, N, K, None)
        mem["w8a16_marlin"] = layer.weight.numel() * layer.weight.element_size() + layer.weight_scale.numel() * 2
        if "w8a16_marlin_nofp32" in which:  # tuning: skip the FP32 cross-CTA reduction buffer
            v["w8a16_marlin_nofp32"] = lambda x: apply_fp8_marlin_linear(
                x, layer.weight, layer.weight_scale, layer.workspace, N, K, None, use_fp32_reduce=False)
            mem["w8a16_marlin_nofp32"] = mem["w8a16_marlin"]
    for w4 in [w for w in which if w.startswith("w4a16_marlin")]:  # 4-bit weight-only RTN, symmetric; _gNN = group
        gs = int(w4.split("_g")[1]) if "_g" in w4 else 128
        from sgl_kernel.scalar_type import scalar_types
        from sglang.srt.layers.quantization.marlin_utils import (apply_gptq_marlin_linear, marlin_make_empty_g_idx,
                                                                 marlin_make_workspace)
        from sglang.test.test_marlin_utils import marlin_quantize

        qs, ss, step = [], [], 31040  # quantize/pack in N chunks (multiple of the 64-col Marlin tile) to bound memory
        for j in range(0, N, step):
            _, q, s, *_ = marlin_quantize(W[j:j + step].t().contiguous(), scalar_types.uint4b8, gs, False)
            qs.append(q)
            ss.append(s)
        q4, s4 = torch.cat(qs, 1), torch.cat(ss, 1)
        del qs, ss
        e, ws4 = marlin_make_empty_g_idx(W.device), marlin_make_workspace(W.device)
        v[w4] = (lambda q4, s4, e, ws4: lambda x: apply_gptq_marlin_linear(
            x, q4, s4, e, e, e, ws4, scalar_types.uint4b8, N, K, True))(q4, s4, e, ws4)
        mem[w4] = q4.numel() * q4.element_size() + s4.numel() * s4.element_size()
    del Wq
    return v, mem


def graph_of(fn, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(x)
    return g, y


def bench_cmd(cfg, a):
    which = a.variants.split(",")
    W, key = load_head(cfg)
    d = torch.load(HID)
    H = d["h"].to("cuda", torch.bfloat16)
    ref_tok = d["tok"].cuda()
    v, mem = make_variants(W, which)
    res = {"head_key": key, "shape": list(W.shape), "n_inputs": H.shape[0], "variants": {}}

    # accuracy vs the BF16 head on all real inputs, chunked (full-vocab logits are 1 MB per row)
    acc = {n: {"top1": 0, "top5": 0, "abs_sum": 0.0, "abs_max": 0.0, "sq_err": 0.0} for n in v}
    sq_ref, self_ok, n_all = 0.0, 0, H.shape[0]
    with torch.no_grad():
        for i in range(0, n_all, 64):
            ref = v["bf16"](H[i:i + 64]).float()
            self_ok += (ref.argmax(1) == ref_tok[i:i + 64]).sum().item()
            sq_ref += ref.pow(2).sum().item()
            r1, r5 = ref.argmax(1), ref.topk(5, dim=1).indices
            for n, fn in v.items():
                out = fn(H[i:i + 64]).float()
                err = (out - ref).abs()
                s = acc[n]
                s["top1"] += (out.argmax(1) == r1).sum().item()
                o5 = out.topk(5, dim=1).indices
                s["top5"] += (o5.unsqueeze(2) == r5.unsqueeze(1)).any(2).sum().item()
                s["abs_sum"] += err.sum().item()
                s["abs_max"] = max(s["abs_max"], err.max().item())
                s["sq_err"] += err.pow(2).sum().item()
                del out, err
    res["selfcheck_argmax_eq_generated"] = round(self_ok / n_all, 4)
    for n, s in acc.items():
        res["variants"][n] = {
            "weight_mib": round(mem[n] / 2**20, 1), "top1_agree": round(s["top1"] / n_all, 4),
            "top5_overlap": round(s["top5"] / (5 * n_all), 4),
            "logit_abs_err_mean": round(s["abs_sum"] / (n_all * W.shape[0]), 4),
            "logit_abs_err_max": round(s["abs_max"], 3), "logit_rel_err": round((s["sq_err"] / sq_ref) ** 0.5, 5)}

    # timing: CUDA-graph replay, variants interleaved per trial, per-replay events
    for M in a.M:
        x = H[:M].contiguous()
        graphs = {}
        for name, fn in v.items():
            torch.cuda.synchronize()
            m0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            fn(x)
            torch.cuda.synchronize()
            res["variants"][name][f"temp_alloc_kib_M{M}"] = round((torch.cuda.max_memory_allocated() - m0) / 1024, 1)
            graphs[name] = graph_of(fn, x)[0]
        times = {n: [] for n in graphs}
        for _ in range(20):  # clock warm-up
            for g in graphs.values():
                g.replay()
        torch.cuda.synchronize()
        e = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in graphs]
        for t in range(a.trials):
            for (n, g), (e0, e1) in zip(graphs.items(), e):
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                times[n].append(e0.elapsed_time(e1) * 1e3)
        for n, ts in times.items():
            med = statistics.median(ts)
            res["variants"][n][f"us_M{M}"] = round(med, 1)
            res["variants"][n][f"us_M{M}_p10_p90"] = [round(sorted(ts)[len(ts) // 10], 1),
                                                      round(sorted(ts)[9 * len(ts) // 10], 1)]
            res["variants"][n][f"GBps_M{M}"] = round(mem[n] / (med * 1e-6) / 1e9, 1)
        del graphs
    out = REPO / "experiments" / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-head")
    out.mkdir(parents=True)
    res["fingerprint"] = bench.fingerprint(cfg) if a.fingerprint else None
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    with open(REPO / "experiments" / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps({"run_id": out.name, "config": cfg["name"], "git": bench.git_state()["commit"][:12],
                             "hypothesis": "draft-only FP8 tied lm_head copy (isolated projection)",
                             "status": "ok", "metrics": res["variants"], "verdict": a.verdict}) + "\n")
    print(f"RUN {out.name} inputs={res['n_inputs']} selfcheck={res['selfcheck_argmax_eq_generated']}")
    for n, r in res["variants"].items():
        print(f"  {n:<13} " + " ".join(f"M{M}={r[f'us_M{M}']}us({r[f'GBps_M{M}']}GB/s)" for M in a.M)
              + f" top1={r['top1_agree']} top5={r['top5_overlap']} rel_err={r['logit_rel_err']} "
                f"W={r['weight_mib']}MiB tmp={r[f'temp_alloc_kib_M{a.M[0]}']}KiB")


def profile_cmd(cfg, a):  # ncu --profile-from-start off ... python head_bench.py profile ...
    W, _ = load_head(cfg)
    H = torch.load(HID)["h"].to("cuda", torch.bfloat16)
    v, _ = make_variants(W, a.variants.split(","))
    x = H[:a.M[0]].contiguous()
    for fn in v.values():
        for _ in range(3):
            fn(x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, fn in v.items():
        torch.cuda.nvtx.range_push(name)
        fn(x)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print("profiled", list(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "bench", "profile"))
    ap.add_argument("config")
    ap.add_argument("--variants", default="bf16,w8a8_torch,w8a8_sgl,w8a16_marlin")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--fingerprint", action="store_true")
    ap.add_argument("--verdict", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    for k, val in cfg.get("env", {}).items():  # same toolkit env as the server (JIT kernels)
        import os

        os.environ[k] = str(val)
    {"collect": collect, "bench": bench_cmd, "profile": profile_cmd}[a.cmd](cfg, a)
