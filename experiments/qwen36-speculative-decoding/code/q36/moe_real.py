"""Phase 5 session 3: MoE on real weights and real routing. Two modes on the route dump written by q37_nvtx (one JSON
line per MoE layer call in verify order, 40 layers per round, 8 tokens x top-8):
 --stats: per M in {1,4,8,16,32}: distinct experts per layer-round, token-expert pairs, repeats, tile padding at BLOCK_M,
   minimum expert-weight bytes per round (distinct x 3.15 MB x 40 layers) and its floor time at 1.40 TB/s.
 --run M: fused_moe on layer --layer's real FP8 weights (safetensors, no server) with real routed expert sets;
   prints mean kernel-path ms per call and the byte lower bound. Run under ncu for DRAM bytes; plain for timing.
Usage: python moe_real.py --routes routes.jsonl --stats | --run 8 [--layer 20] [--calls 200]"""
import argparse, collections, json, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--routes")
ap.add_argument("--kernel", choices=["gemm", "gemmcmp", "attn", "launch"], help="session 4: isolated real-shape kernels for NCU, no routes needed")
ap.add_argument("--repeat", type=int, default=4)
ap.add_argument("--stats", action="store_true")
ap.add_argument("--run", type=int, default=0)
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--calls", type=int, default=200)
ap.add_argument("--block-m", type=int, default=16)
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--remap", type=int, nargs="*", default=None, help="--run M: also time with rows L..M-1 moved onto row 0's experts (adaptive verify length), for each L")
ap.add_argument("--shared", action="store_true", help="numerical reference: shared expert as slot 256 of fused_moe vs float64")
a = ap.parse_args()
if a.kernel == "launch":
    # Cost of one small kernel inside a captured graph: replay time against kernel count, uninstrumented.
    import torch
    x = torch.zeros(8, 2048, device="cuda", dtype=torch.bfloat16)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    pts = {}
    for n in (250, 500, 1000, 1500):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                x.add_(1.0)
        torch.cuda.synchronize(); ts = []
        for _ in range(60):
            s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); pts[n] = ts[len(ts) // 2]
        print(f"LAUNCH n={n} replay median {pts[n]:.3f} ms, {pts[n] * 1e3 / n:.3f} us per kernel", flush=True)
    print(f"LAUNCH slope {(pts[1500] - pts[250]) * 1e3 / 1250:.3f} us per extra kernel, 8x2048 bf16 add")
    raise SystemExit
if a.kernel == "attn":
    # FlashInfer attention at real shapes: 8 query rows, 16 query heads, 2 kv heads, head dim 256, FP8 KV, bf16 query. Context
    # lengths bracket Coral's bins; layout and page size at 90K only read the same bytes through different addressing.
    import flashinfer, torch
    work = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cases = [(20000, "NHD", 1, 0), (90000, "NHD", 1, 0), (180000, "NHD", 1, 0), (90000, "HND", 1, 0), (90000, "NHD", 16, 0), (90000, "NHD", 64, 0)]
    if os.environ.get("ATTN_SPLITS"):  # KV chunk size per query tile; 0 = FlashInfer's default plan
        cases = [(L, "NHD", 1, c) for L in (20000, 90000, 180000) for c in map(int, os.environ["ATTN_SPLITS"].split(","))]
    for L, lay, ps, split in cases:
        pages = (L + ps - 1) // ps
        shape = (pages, ps, 2, 256) if lay == "NHD" else (pages, 2, ps, 256)
        k = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
        v = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
        q = torch.randn(8, 16, 256, device="cuda", dtype=torch.bfloat16) * 0.1
        w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(work, lay, backend="fa2")
        w.plan(torch.tensor([0, 8], device="cuda", dtype=torch.int32), torch.tensor([0, pages], device="cuda", dtype=torch.int32),
               torch.arange(pages, device="cuda", dtype=torch.int32), torch.tensor([L - (pages - 1) * ps], device="cuda", dtype=torch.int32),
               16, 2, 256, ps, causal=True, sm_scale=1.0 / 16.0, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16, **({"fixed_split_size": split} if split else {}))
        for _ in range(3):
            w.run(q, (k, v))
        torch.cuda.synchronize(); ts = []
        for _ in range(a.calls):
            s.record()
            for _ in range(10):  # single-call event pairs bottom out near 17 us
                w.run(q, (k, v))
            e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 100)
        ts.sort()
        print(f"ATTN L={L} layout={lay} page={ps} split={split} per call median {ts[len(ts) // 2]:.1f} us; K+V bytes in context {2 * L * 512 / 1e6:.0f} MB", flush=True)
        del k, v, w
    raise SystemExit
if a.kernel == "gemmcmp":
    # Dense FP8 GEMMs of one layer through each runner backend the server can select: warm (weights in L2) and cold (a 300 MB read
    # between calls evicts them, its own time subtracted). Output compared with the CUTLASS form the baseline uses.
    import torch
    from safetensors import safe_open
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    M0 = os.path.expanduser("~/lab/models/Qwen3.6-35B-A3B-FP8")
    set_global_server_args_for_scheduler(ServerArgs(model_path=M0))
    from sglang.srt.layers.quantization import fp8_utils as U
    if os.environ.get("Q37_GEMM_STREAMK") or os.environ.get("Q37_S4_GEMM"):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import q37_nvtx
        q37_nvtx.install_gemm_streamk() if os.environ.get("Q37_GEMM_STREAMK") else q37_nvtx.install_s4_gemm()
    idx = json.load(open(M0 + "/model.safetensors.index.json"))["weight_map"]

    def get(n):
        with safe_open(M0 + "/" + idx[n], "pt") as f:
            return f.get_tensor(n).cuda()

    def fp8(p, names):
        return torch.cat([get(f"{p}{n}.weight") for n in names], 0), torch.cat([get(f"{p}{n}.weight_scale_inv") for n in names], 0).float()
    G, F = "model.language_model.layers.20.", "model.language_model.layers.19."
    forms = {"gdn_qkvz": fp8(G + "linear_attn.", ["in_proj_qkv", "in_proj_z"]), "gdn_out": fp8(G + "linear_attn.", ["out_proj"]), "fa_qkv": fp8(F + "self_attn.", ["q_proj", "k_proj", "v_proj"]),
             "fa_o": fp8(F + "self_attn.", ["o_proj"]), "shared_gu": fp8(G + "mlp.shared_expert.", ["gate_proj", "up_proj"]), "shared_d": fp8(G + "mlp.shared_expert.", ["down_proj"])}
    torch.manual_seed(0)
    xs = {k: torch.randn(8, w.shape[1], device="cuda", dtype=torch.bfloat16) * 0.1 for k, (w, _) in forms.items()}
    head = get("lm_head.weight").bfloat16(); ex = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def bench(fn, n=20, cold=False):
        # Captured graph, as in the server: eager Python would hide the kernel behind 40 us of host work per call.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                if cold:
                    ex.matmul(head.T)
                fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(15):
            s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort(); return ts[7] * 1e3 / n
    ex.matmul(head.T); torch.cuda.synchronize()  # cuBLAS must be initialized before graph capture
    ev = bench(lambda: None, cold=True)
    ref = {}
    print(f"evict read {ev:.1f} us", flush=True)
    dump = {}
    for b in os.environ.get("GEMMCMP_B", "cutlass").split(","):
        try:
            if b.startswith("splitk:"):
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import s4_splitk
                from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8 as ptq
                _, P_, BN_, BM_, ST_, WP_ = (b.split(":") + ["2", "4"])[:6] if b.count(":") == 3 else b.split(":")

                def lin(input, weight, block_size, weight_scale, input_scale, bias, P_=int(P_), BN_=int(BN_), BM_=int(BM_), ST_=int(ST_), WP_=int(WP_)):
                    q_, qs_ = ptq(input.view(-1, input.shape[-1]), 128, column_major_scales=False)
                    return s4_splitk.s4_gemm(q_, weight, qs_, weight_scale, P=P_, block_n=BN_, block_m=BM_, num_warps=WP_, num_stages=ST_)
            else:
                lin = U._dispatch_explicit_backend(U.Fp8GemmRunnerBackend(b))
            for k, (w, sc) in forms.items():
                fn = lambda: lin(input=xs[k], weight=w, block_size=[128, 128], weight_scale=sc, input_scale=None, bias=None)
                y = fn(); fn(); ref.setdefault(k, y.clone()); dump[k] = y.cpu()
                d = ((y.double() - ref[k].double()).norm(dim=1) / ref[k].double().norm(dim=1)).max().item()
                print(f"GEMMCMP {b} {k} warm {bench(fn):.1f} us cold {bench(fn, cold=True) - ev:.1f} us rel_diff_vs_cutlass {d:.2e} bitwise {bool((y == ref[k]).all())}", flush=True)
        except Exception as ex2:
            print(f"GEMMCMP {b} failed: {str(ex2)[:150]}", flush=True)
    if os.environ.get("GEMM_DUMP"):
        torch.save(dump, os.environ["GEMM_DUMP"])
    raise SystemExit
if a.kernel == "gemm":
    # The dense GEMMs of one GDN layer, one attention layer, both heads and one drafter layer: real weights, real shapes, M=8.
    # Printed order is the kernel launch order, so NCU rows map back to these names.
    import torch
    from safetensors import safe_open
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    M0 = os.path.expanduser("~/lab/models/Qwen3.6-35B-A3B-FP8")
    set_global_server_args_for_scheduler(ServerArgs(model_path=M0))
    from sglang.srt.layers.quantization.fp8_utils import dispatch_w8a8_block_fp8_linear, initialize_fp8_gemm_config
    initialize_fp8_gemm_config()  # as the scheduler does: auto on SM120 means CUTLASS
    if os.environ.get("Q37_S4_GEMM"):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import q37_nvtx
        q37_nvtx.install_s4_gemm()
    linear = dispatch_w8a8_block_fp8_linear(weight_block_size=[128, 128], act_scale_ue8m0=False)
    idx = json.load(open(M0 + "/model.safetensors.index.json"))["weight_map"]

    def get(n):
        with safe_open(M0 + "/" + idx[n], "pt") as f:
            return f.get_tensor(n).cuda()

    def fp8(p, names):
        w = torch.cat([get(f"{p}{n}.weight") for n in names], 0)
        sc = torch.cat([get(f"{p}{n}.weight_scale_inv") for n in names], 0).float()
        x = torch.randn(8, w.shape[1], device="cuda", dtype=torch.bfloat16) * 0.1
        return lambda: linear(input=x, weight=w, block_size=[128, 128], weight_scale=sc, input_scale=None, bias=None), w.numel()

    def bf16(w):
        x = torch.randn(8, w.shape[1], device="cuda", dtype=torch.bfloat16) * 0.1
        return lambda: x.matmul(w.T), w.numel() * 2
    G, F = "model.language_model.layers.20.", "model.language_model.layers.19."
    head = get("lm_head.weight").bfloat16()
    hot = torch.load(os.path.expanduser("~/lab/profiles/q36_s3/hotv/hot_ids_48000.pt")).cuda()
    with safe_open(os.path.expanduser("~/lab/models/qwen-swe-drafter/model.safetensors"), "pt") as f:
        d = lambda *ns: torch.cat([f.get_tensor(f"layers.0.{n}.weight") for n in ns], 0).cuda().bfloat16()
        forms = {"gdn_qkvz": fp8(G + "linear_attn.", ["in_proj_qkv", "in_proj_z"]), "gdn_ba": bf16(torch.cat([get(G + "linear_attn.in_proj_b.weight"), get(G + "linear_attn.in_proj_a.weight")], 0).bfloat16()),
                 "gdn_out": fp8(G + "linear_attn.", ["out_proj"]), "fa_qkv": fp8(F + "self_attn.", ["q_proj", "k_proj", "v_proj"]), "fa_o": fp8(F + "self_attn.", ["o_proj"]),
                 "mlp_gate": bf16(get(G + "mlp.gate.weight").bfloat16()), "shared_gu": fp8(G + "mlp.shared_expert.", ["gate_proj", "up_proj"]), "shared_d": fp8(G + "mlp.shared_expert.", ["down_proj"]),
                 "lm_head": bf16(head), "hot_head": bf16(head[hot].clone()), "d_qkv": bf16(d("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")), "d_o": bf16(d("self_attn.o_proj")),
                 "d_gu": bf16(d("mlp.gate_proj", "mlp.up_proj")), "d_down": bf16(d("mlp.down_proj"))}
    for name, (fn, nb) in forms.items():
        for _ in range(a.repeat):
            fn()
        torch.cuda.synchronize()
        print(f"GEMM {name} weight_bytes {nb / 1e6:.2f} MB calls {a.repeat}", flush=True)
    raise SystemExit
EXP_BYTES =(1024 * 2048 + 2048 * 512) + 4 * (8 * 16 + 16 * 4) * 2  # w13 + w2 + bf16 block scales (approx)
rows = [json.loads("[" + l.strip().replace("] [", "],[") + "]") for l in open(a.routes)]
rows = [r for r in rows if len(r) == 8 and len(r[0]) == 8]
rounds = [rows[i: i + a.layers] for i in range(0, len(rows) - a.layers + 1, a.layers)]
print("verify layer-calls", len(rows), "rounds", len(rounds))


def group(M):
    """Per layer: token-routing lists of M consecutive tokens (M<=8: first M of each round; M>8: consecutive rounds)."""
    out = collections.defaultdict(list)
    per = (M + 7) // 8
    for j in range(0, len(rounds) - per + 1, per):
        for l in range(a.layers):
            toks = [t for rd in rounds[j: j + per] for t in rd[l]][:M]
            out[l].append(toks)
    return out


if a.stats:
    print("| M | groups | distinct experts/layer (mean, p50, p95) | pairs M*8 | repeat frac | tiles padded (BLOCK_M) useful frac | min weight MB/round | floor ms @1.40 TB/s |")
    for M in (1, 4, 8, 16, 32):
        g = group(M); d = []; useful = []
        for l, vs in g.items():
            for toks in vs:
                c = collections.Counter(e for t in toks for e in t)
                d.append(len(c))
                blocks = sum((n + a.block_m - 1) // a.block_m for n in c.values())
                useful.append(sum(c.values()) / (blocks * a.block_m))
        d.sort(); mean = st.mean(d)
        mb = mean * EXP_BYTES * a.layers / 1e6
        print(f"| {M} | {len(d)} | {mean:.1f}, {d[len(d) // 2]}, {d[int(.95 * (len(d) - 1))]} | {M * 8} | {1 - mean / (M * 8):.3f} | {st.mean(useful):.3f} | {mb:.0f} | {mb * 1e6 / 1.40e12 * 1e3:.2f} |")
    freq = collections.Counter(e for rd in rounds for r in rd[a.layer] for e in r)
    tot = sum(freq.values()); top = [v for _, v in freq.most_common()]
    print(f"layer {a.layer} expert frequency: {len(freq)} of 256 used; top 8 hold {sum(top[:8]) / tot:.2f}, top 32 hold {sum(top[:32]) / tot:.2f}")
    same = [len(set(e for t in rounds[j][l] for e in t) & set(e for t in rounds[j + 1][l] for e in t)) / max(1, len(set(e for t in rounds[j][l] for e in t)))
            for j in range(len(rounds) - 1) for l in range(a.layers)]
    print(f"expert overlap with the previous round (same layer): {st.mean(same):.3f}")

if a.run:
    import torch
    from safetensors import safe_open
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path=os.path.expanduser("~/lab/models/Qwen3.6-35B-A3B-FP8")))
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    M0 = os.path.expanduser("~/lab/models/Qwen3.6-35B-A3B-FP8")
    idx = json.load(open(M0 + "/model.safetensors.index.json"))["weight_map"]
    P = f"model.language_model.layers.{a.layer}.mlp.experts."
    with safe_open(M0 + "/" + idx[P + "0.gate_proj.weight"], "pt") as f:
        def g(i, n):
            return f.get_tensor(f"{P}{i}.{n}")
        w1 = torch.stack([torch.cat([g(i, "gate_proj.weight"), g(i, "up_proj.weight")], 0) for i in range(256)]).cuda()
        w2 = torch.stack([g(i, "down_proj.weight") for i in range(256)]).cuda()
        s1 = torch.stack([torch.cat([g(i, "gate_proj.weight_scale_inv"), g(i, "up_proj.weight_scale_inv")], 0) for i in range(256)]).float().cuda()
        s2 = torch.stack([g(i, "down_proj.weight_scale_inv") for i in range(256)]).float().cuda()
        if a.shared:
            Sh = f"model.language_model.layers.{a.layer}.mlp.shared_expert."
            sw1 = torch.cat([f.get_tensor(Sh + "gate_proj.weight"), f.get_tensor(Sh + "up_proj.weight")], 0).cuda()
            sw2 = f.get_tensor(Sh + "down_proj.weight").cuda()
            ss1 = torch.cat([f.get_tensor(Sh + "gate_proj.weight_scale_inv"), f.get_tensor(Sh + "up_proj.weight_scale_inv")], 0).float().cuda()
            ss2 = f.get_tensor(Sh + "down_proj.weight_scale_inv").float().cuda()
            sg = f.get_tensor(f"model.language_model.layers.{a.layer}.mlp.shared_expert_gate.weight").cuda()
    M = a.run
    if a.shared:
        def deq(w, sc):
            out = w.double()
            bh, bw = w.shape[0] // sc.shape[0], w.shape[1] // sc.shape[1]
            return out * sc.double().repeat_interleave(bh, 0).repeat_interleave(bw, 1)

        def expert(x, w1i, s1i, w2i, s2i):
            gu = x.matmul(deq(w1i, s1i).T)
            gate, up = gu[:, :512], gu[:, 512:]
            return (torch.nn.functional.silu(gate) * up).matmul(deq(w2i, s2i).T)
        rows = [r for r in rows]
        W1, W2 = torch.cat([w1, sw1[None]]), torch.cat([w2, sw2[None]])
        S1, S2 = torch.cat([s1, ss1[None]]), torch.cat([s2, ss2[None]])
        torch.manual_seed(0)
        x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16)
        rc = MoeRunnerConfig(inplace=True)
        errs, errs8, ys = [], [], []
        for toks in (group(M)[a.layer] or [])[:a.calls]:
            ids = torch.tensor(toks, device="cuda", dtype=torch.int32)
            wt = torch.softmax(torch.linspace(3, 0, 8, device="cuda"), 0).float().expand(M, 8).clone()
            sigma = torch.sigmoid(x.float().matmul(sg.float().T))
            ids9, wt9 = torch.cat([ids, torch.full((M, 1), 256, device="cuda", dtype=ids.dtype)], 1), torch.cat([wt, sigma], 1)
            from sglang.srt.layers.moe.topk import StandardTopKOutput
            t = StandardTopKOutput(topk_weights=wt9, topk_ids=ids9, router_logits=None)
            y = fused_moe(x.clone(), W1, W2, t, moe_runner_config=rc, use_fp8_w8a8=True, w1_scale=S1, w2_scale=S2, block_shape=[128, 128])
            t8 = StandardTopKOutput(topk_weights=wt, topk_ids=ids, router_logits=None)
            y8 = fused_moe(x.clone(), w1, w2, t8, moe_runner_config=rc, use_fp8_w8a8=True, w1_scale=s1, w2_scale=s2, block_shape=[128, 128])
            xd = x.double(); ref8 = torch.zeros(M, 2048, device="cuda", dtype=torch.float64)
            for e in ids.unique().tolist():
                m = (ids == e); rw = (m.double() * wt.double()).sum(1, keepdim=True)
                ref8 += rw * expert(xd, w1[e], s1[e], w2[e], s2[e])
            ref = ref8 + sigma.double() * expert(xd, sw1, ss1, sw2, ss2)
            rel = lambda u, v: ((u.double() - v).norm(dim=1) / v.norm(dim=1)).max().item()
            errs.append(rel(y, ref)); errs8.append(rel(y8, ref8)); ys.append(y8.cpu())
            if len(errs) >= 40:
                break
        errs.sort(); errs8.sort()
        if os.environ.get("MOE_DUMP"):  # FP8 activation rounding alone is about 4.5% against float64, so config forms are compared to each other
            torch.save(ys, os.environ["MOE_DUMP"])
        print(f"SHARED M={M} layer={a.layer} n={len(errs)}: 9-slot fused_moe vs float64 rel err median {errs[len(errs) // 2]:.3e} max {errs[-1]:.3e}; routed-only fused_moe median {errs8[len(errs8) // 2]:.3e} max {errs8[-1]:.3e}", flush=True)
        raise SystemExit
    if a.remap is not None:   # graph-timed: 32 calls per replay, 4 real layers rotated (no cross-call L2 reuse), routing per call
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        Ls = [5, 15, 25, 35]; Wt = []
        for Ly in Ls:
            Pq = f"model.language_model.layers.{Ly}.mlp.experts."
            with safe_open(M0 + "/" + idx[Pq + "0.gate_proj.weight"], "pt") as f:
                gg = lambda i, n: f.get_tensor(f"{Pq}{i}.{n}")
                Wt.append((torch.stack([torch.cat([gg(i, "gate_proj.weight"), gg(i, "up_proj.weight")], 0) for i in range(256)]).cuda(),
                           torch.stack([gg(i, "down_proj.weight") for i in range(256)]).cuda(),
                           torch.stack([torch.cat([gg(i, "gate_proj.weight_scale_inv"), gg(i, "up_proj.weight_scale_inv")], 0) for i in range(256)]).float().cuda(),
                           torch.stack([gg(i, "down_proj.weight_scale_inv") for i in range(256)]).float().cuda()))
        G = {Ly: group(M)[Ly] for Ly in Ls}
        x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16) * 0.01; rc = MoeRunnerConfig(inplace=True)
        wt = torch.softmax(torch.linspace(3, 0, 8, device="cuda"), 0).float().expand(M, 8).contiguous()
        for L in a.remap:
            tops, dist = [], []
            for c in range(32):
                Ly = Ls[c % 4]; toks = G[Ly][(c // 4) % len(G[Ly])]
                toks = [t if i < L else toks[0] for i, t in enumerate(toks)]
                tops.append(StandardTopKOutput(topk_weights=wt, topk_ids=torch.tensor(toks, device="cuda", dtype=torch.int32), router_logits=None))
                dist.append(len({e for t in toks for e in t}))
            run = lambda: [fused_moe(x, *Wt[c % 4][:2], tops[c], moe_runner_config=rc, use_fp8_w8a8=True, w1_scale=Wt[c % 4][2], w2_scale=Wt[c % 4][3], block_shape=[128, 128]) for c in range(32)]
            run(); torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                run()
            for _ in range(3): gr.replay()
            torch.cuda.synchronize(); s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(20):
                s0.record(); gr.replay(); e0.record(); torch.cuda.synchronize(); ts.append(s0.elapsed_time(e0) / 32 * 1e3)
            ts.sort(); lb = st.mean(dist) * EXP_BYTES
            print(f"GRAPH M={M} L={L}: {ts[len(ts) // 2]:.1f} us per layer call (graph, 4 layers rotated); distinct {st.mean(dist):.1f}; "
                  f"weights {lb / 1e6:.1f} MB -> {lb / (ts[len(ts) // 2] * 1e-6) / 1e12:.2f} TB/s; x40 layers {ts[len(ts) // 2] * 40 / 1e3:.3f} ms/round", flush=True)
            del gr
        raise SystemExit
    sets0 = group(M)[a.layer] or [[[j for j in range(8)] for _ in range(M)]]
    for L in (a.remap or [M]):
      sets = [[t if i < L else toks[0] for i, t in enumerate(toks)] for toks in sets0]
      x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16)
      cfg = TopKConfig(top_k=8, renormalize=True)
      gates = []
      for toks in sets:
          lg = torch.zeros(M, 256, device="cuda", dtype=torch.float32)
          for i, t in enumerate(toks):
              lg[i, torch.tensor(t, device="cuda")] = torch.linspace(9, 2, 8, device="cuda")
          gates.append(lg)
      top = select_experts(x, gates[0], cfg)
      rc = MoeRunnerConfig(inplace=True)
      dist = [len({e for t in toks for e in t}) for toks in sets]

      def call(i):
          n = select_experts(x, gates[i % len(gates)], cfg)
          top.topk_weights.copy_(n.topk_weights); top.topk_ids.copy_(n.topk_ids)
          return fused_moe(x, w1, w2, top, moe_runner_config=rc, use_fp8_w8a8=True, w1_scale=s1, w2_scale=s2, block_shape=[128, 128])
      for i in range(10):
          call(i)
      torch.cuda.synchronize()
      s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
      ts = []
      for i in range(a.calls):
          s.record(); call(i); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
      ts.sort()
      lb = st.mean(dist) * EXP_BYTES
      print(f"RUN M={M} L={L} layer={a.layer}: path median {ts[len(ts) // 2]:.1f} ms*1e-0; distinct {st.mean(dist):.1f}; min weight bytes per call {lb / 1e6:.1f} MB; floor at 1.40 TB/s {lb / 1.4e12 * 1e6:.1f} us; path median us {ts[len(ts) // 2] * 1e3:.1f}")
