"""Phase 5 session 3: lightweight first-call-to-last-call attribution. Wraps existing scheduler and worker functions in
NVTX ranges (torch.cuda.nvtx push/pop: no GPU sync, no allocation). Loaded only in venv-q37 when Q37_NVTX=1.
Also, with Q37_ROUTE_DUMP=<path> (eager server only): records per-layer expert routing of small verify batches."""
import functools, importlib, json, os, torch

nvtx = torch.cuda.nvtx


def wrap(cls, name, label=None, dyn=None):
    f = getattr(cls, name, None)
    if f is None:
        print("q37_nvtx: missing", cls.__name__, name); return
    base = label or f"{cls.__name__}.{name}"

    @functools.wraps(f)
    def g(*a, **k):
        nvtx.range_push(dyn(a, k, base) if dyn else base)
        try:
            return f(*a, **k)
        finally:
            nvtx.range_pop()
    setattr(cls, name, g)


def mode_of(a, k, base):
    fb = a[1] if len(a) > 1 else k.get("forward_batch") or k.get("batch")
    m = getattr(fb, "forward_mode", None)
    return f"{base}:{getattr(m, 'name', m)}"


def install():
    S = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    for n in ("ingest_requests", "get_next_batch_to_run", "run_batch", "process_batch_result", "launch_batch_sample_if_needed"):
        wrap(S, n)
    D = importlib.import_module("sglang.srt.speculative.dflash_worker_v2").DFlashWorkerV2
    for n in ("forward_batch_generation", "_accept_block", "_update_target_mamba_state_after_verify", "_append_target_hidden_to_draft_kv_by_loc",
              "_rebuild_compact_draft_cache", "_make_next_draft_input_decode", "_compute_compact_draft_seq_lens", "_ensure_draft_block_buffers"):
        wrap(D, n, f"dfl.{n}")
    MR = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    wrap(MR, "forward", "MR.forward", mode_of)
    for n in ("_forward_raw", "init_forward_metadata", "sample"):
        if hasattr(MR, n):
            wrap(MR, n, f"MR.{n}")
    FI = importlib.import_module("sglang.srt.layers.attention.flashinfer_backend")
    for c in ("FlashInferAttnBackend",):
        if hasattr(FI, c):
            wrap(getattr(FI, c), "init_forward_metadata", "fi.init_forward_metadata")
    print("q37_nvtx installed", flush=True)


def install_route_dump(path):
    T = importlib.import_module("sglang.srt.layers.moe.topk").TopK
    f = T.forward
    out = open(path, "a")

    def g(self, x, *a, **k):
        r = f(self, x, *a, **k)
        ids = getattr(r, "topk_ids", None)
        if ids is None and isinstance(r, tuple):
            ids = r[1]
        if ids is not None and ids.shape[0] <= 64:
            out.write(" ".join(map(str, ids.reshape(ids.shape[0], -1).tolist())) + "\n"); out.flush()
        return r
    T.forward = g
    print("q37 route dump ->", path, flush=True)


AVL = {}


LSK_N = 16384


def lskip_images(w, lens, q_len, group, slack=None):
    """Q37_ATTN_LSKIP helper: copies of the wrapper's just-planned FlashInfer int workspace (split plan) that compute only the
    first t query tiles (t = 1 .. tiles-1) with a finer KV split (stable_chunk for t tiles, so one wave still fills the SMs).
    Rows of tiles >= t get zero merge sets, which FlashInfer's merge writes as exact zeros. Image tiles-1 is the plan itself.
    Layout from FlashInfer's PrefillPlanInfo: [padded, rows, rows_off, cta_q, req, qo_tile, kv_tile, merge, o_indptr, chunk,
    v, s, valid, graph, split]. Returns a pinned uint8 [tiles, N] tensor, or None when the plan is not split."""
    pi = [int(x) for x in w._plan_info]
    if len(pi) != 15 or not pi[14]:
        return None
    pad, cta = pi[0], pi[3]
    N = LSK_N                                     # fixed: the graph copies this many bytes; the plan's own arrays end well before
    tiles = (q_len * group + cta - 1) // cta
    if tiles < 2 or pi[12] + pad > N:
        return None
    buf = getattr(w, "_q37_lsk_pin", None)
    if buf is None or buf[0].shape != (tiles, N):
        buf = [torch.empty(tiles, N, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    buf.reverse(); w._q37_lsk_pin = buf          # alternate: the other copy may still be queued behind GPU work
    img = buf[0]; img[:] = w._pin_memory_int_workspace_buffer[:N]
    for t in range(1, tiles):
        c = stable_chunk(lens, SLACK if slack is None else slack, t, pad)
        if not c:
            continue                              # keep the full plan for this variant
        nch = max((u + c - 1) // c for u in lens); items = t * nch
        v = img[t - 1]
        i32 = lambda o, n: v[o: o + 4 * n].view(torch.int32)
        i32(pi[4], items).zero_()
        i32(pi[5], items).copy_(torch.arange(t, dtype=torch.int32).repeat_interleave(nch))
        i32(pi[6], items).copy_(torch.arange(nch, dtype=torch.int32).repeat(t))
        i32(pi[8], 2).copy_(torch.tensor([0, q_len * nch], dtype=torch.int32))
        i32(pi[9], 1).fill_(c)
        i32(pi[7], q_len + 1).copy_((torch.arange(q_len + 1).clamp(max=t * cta // group) * nch).to(torch.int32))
        v[pi[12]: pi[12] + pad].copy_((torch.arange(pad) < items).to(torch.uint8))
    return img


def install_attn_lskip():
    """Q37_ATTN_LSKIP=1 (needs Q37_AVL, Q37_SYNCFREE): verify-length-aware target attention on FlashInfer's own kernel. Block 16 runs
    two 64-row query tiles (tokens 0-7, 8-15) per KV chunk. Each round the host also plans "first t tiles only" variants with a
    finer KV split (lskip_images); inside the verify graph, once per forward, the variant for this round's AVL length L is copied
    into the wrapper's int workspace (index_select + copy, no host sync). With L <= 8 the kernel then runs block-8 work; rows of
    skipped tiles merge to exact zeros and are never used (drafts >= L are masked). Rows < L see the same causal attention, only the
    KV chunking (summation order) differs: Track S. Uses FlashInfer's block_valid_mask (CTAs past the work list exit)."""
    FI = importlib.import_module("sglang.srt.layers.attention.flashinfer_backend")
    import flashinfer
    Q = importlib.import_module("sglang.srt.models.qwen3_5")
    C = Q.Qwen3_5MoeForConditionalGeneration
    st = {"verify": False, "sel": set(), "n": 0}
    fw = C.forward

    @functools.wraps(fw)
    def fw2(self, *a, **k):
        fb = k.get("forward_batch") or next((x for x in a if hasattr(x, "forward_mode")), None)
        st["verify"] = bool(fb is not None and fb.forward_mode.is_target_verify()); st["sel"] = set()
        try:
            return fw(self, *a, **k)
        finally:
            st["verify"] = False
    C.forward = fw2
    f = FI.fast_prefill_plan

    @functools.wraps(f)
    def h(self, *a, **k):
        r = f(self, *a, **k)
        try:
            q = int(k["max_q_len"]); bs = len(a[0]) - 1
            if q == BLOCK and bs == 1 and int(a[7]) == 1 and a[4] // a[5] * q > 64 and not torch.cuda.is_current_stream_capturing():
                lens = (k["kv_lens_host"] + q).tolist()
                d = getattr(self, "_q37_lsk_dev", None)
                img = lskip_images(self, lens, q, a[4] // a[5]) if d is not None else None
                if d is not None and (img is None or img.shape != d.shape):   # no variants this round: every row = the plan itself
                    buf = getattr(self, "_q37_lsk_pin2", None)
                    if buf is None:
                        buf = self._q37_lsk_pin2 = [torch.empty(d.shape, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
                    buf.reverse(); img = buf[0]; img[:] = self._pin_memory_int_workspace_buffer[: d.shape[1]]
                if d is not None:
                    d.copy_(img, non_blocking=True)
        except Exception as ex:   # never break planning: the full plan is already in place
            if st["n"] < 3:
                st["n"] += 1; print("q37 lskip plan skipped:", type(ex).__name__, str(ex)[:160], flush=True)
        return r
    FI.fast_prefill_plan = h
    Wc = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper
    plan0 = Wc.plan

    @functools.wraps(plan0)
    def plan(self, *a, **k):   # capture-time real plan(): make the variant buffer now (eager, outside the graph), every row = this plan
        r = plan0(self, *a, **k)
        try:
            if (getattr(self, "is_cuda_graph_enabled", False) and getattr(self, "_fixed_batch_size", 0) == 1 and len(a) >= 6
                    and not torch.cuda.is_current_stream_capturing() and getattr(self, "_q37_lsk_dev", None) is None):
                qi = a[0]; q = int(qi[1] - qi[0]); g = int(a[4]) // int(a[5]); cta = int(self._plan_info[3])
                tiles = (q * g + cta - 1) // cta
                if q == BLOCK and tiles >= 2:
                    ws = self._int_workspace_buffer
                    self._q37_lsk_dev = ws[:LSK_N].unsqueeze(0).repeat(tiles, 1)
                    self._q37_lsk_lut = torch.tensor([max(0, (min(L, q) * g + cta - 1) // cta - 1) for L in range(q + 1)],
                                                     dtype=torch.long, device=ws.device)
                    print(f"q37 lskip buffer: tiles {tiles} cta {cta} group {g}", flush=True)
        except Exception as ex:
            print("q37 lskip buffer skipped:", type(ex).__name__, str(ex)[:160], flush=True)
        return r
    Wc.plan = plan
    if getattr(Wc, "begin_forward", None) is plan0:   # FlashInfer aliases begin_forward = plan at class definition; SGLang calls that
        Wc.begin_forward = plan
    run0 = Wc.run

    @functools.wraps(run0)
    def run(self, q, *a, **k):
        d = getattr(self, "_q37_lsk_dev", None)
        if st["verify"] and d is not None and "L" in AVL and id(self) not in st["sel"] and q.shape[0] == BLOCK:
            st["sel"].add(id(self))
            self._int_workspace_buffer[: d.shape[1]].copy_(d.index_select(0, self._q37_lsk_lut.index_select(0, AVL["L"][:1].long())).view(-1))
            if st["n"] < 6:
                st["n"] += 1; print(f"q37 lskip select in verify: variants {tuple(d.shape)}", flush=True)
        return run0(self, q, *a, **k)
    Wc.run = run
    print("q37 attn lskip installed", flush=True)


def install_attn128():
    """Q37_ATTN128=1 [Q37_ATTN128_CFG=BLOCK_N,warps,stages,min_chunk] [Q37_ATTN128_NOL=1]: target-verify attention through our
    128-row split-KV kernel (attn128.py) instead of FlashInfer fa2's two 64-row tiles. Only for one request (bs 1), causal, NHD FP8
    KV, head_dim 256, Q37_BLOCK query rows, inside the target's verify forward; everything else stays on FlashInfer. With Q37_AVL
    (and not _NOL) the kernel computes only the rows of tokens < L and zeroes the rest (unused: drafts >= L are masked)."""
    import sys
    import flashinfer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attn128 import verify_attn, _cfg
    Q = importlib.import_module("sglang.srt.models.qwen3_5")
    C = Q.Qwen3_5MoeForConditionalGeneration
    st = {"verify": False, "n": 0}
    fw = C.forward

    @functools.wraps(fw)
    def fw2(self, *a, **k):
        fb = k.get("forward_batch") or next((x for x in a if hasattr(x, "forward_mode")), None)
        st["verify"] = bool(fb is not None and fb.forward_mode.is_target_verify())
        try:
            return fw(self, *a, **k)
        finally:
            st["verify"] = False
    C.forward = fw2
    Wc = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper
    run0 = Wc.run
    lawarem = not os.environ.get("Q37_ATTN128_NOL")
    f8 = torch.float8_e4m3fn

    @functools.wraps(run0)
    def run(self, q, paged_kv_cache, *a, **k):
        kv = paged_kv_cache
        ks, vs = k.get("k_scale"), k.get("v_scale")
        if (st["verify"] and not a and isinstance(kv, (tuple, list)) and q.dim() == 3 and q.shape[0] == BLOCK and q.shape[2] == 256
                and q.dtype == torch.bfloat16 and kv[0].dtype == f8 and kv[1].dtype == f8 and q.shape[1] % kv[0].shape[-2] == 0
                and (getattr(self, "_batch_size", None) or getattr(self, "_fixed_batch_size", None)) == 1
                and getattr(self, "_causal", True) and getattr(self, "_kv_layout", "NHD") == "NHD"
                and not getattr(self, "_logits_soft_cap", None) and getattr(self, "_window_left", -1) in (None, -1)
                and not k.get("return_lse") and k.get("lse") is None and k.get("sinks") is None and k.get("q_scale") is None
                and k.get("window_left") in (None, -1) and not isinstance(ks, torch.Tensor) and not isinstance(vs, torch.Tensor)):
            o = verify_attn(q, kv[0], kv[1], self._paged_kv_indptr_buf, self._paged_kv_indices_buf, self._paged_kv_last_page_len_buf,
                            getattr(self, "_sm_scale", None) or q.shape[-1] ** -0.5, ks or 1.0, vs or 1.0,
                            L=AVL["L"] if (lawarem and "L" in AVL) else None, out=k.get("out"))
            if st["n"] < 2:
                st["n"] += 1
                print(f"q37 attn128 verify q {tuple(q.shape)} kv {tuple(kv[0].shape)} cfg {_cfg()} L-aware {lawarem and 'L' in AVL}", flush=True)
            return o
        return run0(self, q, paged_kv_cache, *a, **k)
    Wc.run = run
    print("q37 attn128 installed", flush=True)


def install_avl():
    """Q37_AVL=1: adaptive verification length (ours). The block-16 graphs stay; per round the draft picks L from the drafter's own
    path probabilities (calibrated, Q37_AVL_CAL) by maximizing E[useful tokens | L] - lam * MoE_ms(L) (measured: moe_real.py --remap).
    Verify rows L..15 are remapped onto row 0's experts in every MoE layer, so they read no new expert weights; drafts at positions
    >= L are masked before acceptance, so at most L tokens commit and the bonus comes from a row with exact MoE. Rows < L compute
    exactly what they would at full length (row-independent kernels, causal attention and recurrence), and the bonus is the target's
    argmax, so the greedy output equals block-16's; only round boundaries move."""
    import torch
    W = importlib.import_module("sglang.srt.speculative.dflash_worker_v2")
    Dm = importlib.import_module("sglang.srt.models.dflash")
    T = importlib.import_module("sglang.srt.layers.moe.topk").TopK
    Q = importlib.import_module("sglang.srt.models.qwen3_5")
    B = BLOCK
    lam = float(os.environ.get("Q37_AVL_LAM", "0.75")); lmin = int(os.environ.get("Q37_AVL_MIN", "2"))
    cal = [float(x) for x in os.environ.get("Q37_AVL_CAL", "0.05,0.101,0.201,0.296,0.381,0.493,0.591,0.689,0.791,0.981").split(",")]
    meas = {1: 1.253, 2: 1.684, 4: 2.308, 6: 2.867, 8: 3.387, 10: 3.727, 12: 4.062, 14: 4.371, 16: 4.798}   # ms per round, M=16 graph
    if os.environ.get("Q37_AVL_MEAS"):   # "L:ms,..." e.g. block 24: measured M=20/24 points from moe_real.py --remap
        meas.update({int(x.split(":")[0]): float(x.split(":")[1]) for x in os.environ["Q37_AVL_MEAS"].split(",")})
    ks = sorted(meas)

    def interp(L):   # linear between measured points, last slope beyond them
        if L in meas:
            return meas[L]
        lo = max([x for x in ks if x < L], default=ks[0]); hi = min([x for x in ks if x > L], default=None)
        if hi is None:
            lo, hi = ks[-2], ks[-1]
        return meas[lo] + (meas[hi] - meas[lo]) * (L - lo) / (hi - lo)
    Tv = [interp(L) for L in range(1, B + 1)]
    st = {"verify": False, "n": 0, "hist": [0] * (B + 1)}
    def alloc(dev):   # static buffers, made before any graph capture: their addresses are baked into the draft and verify graphs
        if "L" in AVL:
            return
        assert not torch.cuda.is_current_stream_capturing(), "q37 avl buffers must exist before graph capture"
        nmax = 64
        AVL.update(L=torch.full((nmax,), B, dtype=torch.int32, device=dev), keep=torch.ones(nmax, B, 1, dtype=torch.bool, device=dev),
                   keepc=torch.ones(nmax, B, dtype=torch.bool, device=dev), cal=torch.tensor(cal, device=dev),
                   T=torch.tensor(Tv, device=dev), ar=torch.arange(B, device=dev), one=torch.ones(nmax, 1, device=dev))
    wi = W.DFlashWorkerV2.__init__

    @functools.wraps(wi)
    def wi2(self, *a, **k):
        alloc(torch.device("cuda", torch.cuda.current_device()))
        return wi(self, *a, **k)
    W.DFlashWorkerV2.__init__ = wi2
    lat = W._selector_lattice

    def lat2(draft_model, pred_hidden, anchor_token_ids):
        c, s = lat(draft_model, pred_hidden, anchor_token_ids)
        AVL["cs"] = (c, s)
        return c, s
    W._selector_lattice = lat2
    sp = Dm.CandidateSelector.sample_path

    def sp2(self, *, candidate_ids, scores, **k):
        tokens, q = sp(self, candidate_ids=candidate_ids, scores=scores, **k)
        bs, g, K = candidate_ids.shape
        if "L" not in AVL:
            alloc(tokens.device)
        pidx = (candidate_ids == tokens.reshape(bs, g, 1)).float().argmax(-1)                       # chosen candidate per position
        c0 = scores[:, 0, 0].float().softmax(-1).amax(-1, keepdim=True)
        rows = scores[:, 1:].gather(2, pidx[:, :-1, None, None].expand(-1, -1, 1, K))[:, :, 0]
        conf = torch.cat([c0, rows.float().softmax(-1).amax(-1)], 1)                                  # [bs, B-1]
        p = AVL["cal"][(conf * 10).long().clamp_(0, 9)]
        E = torch.cat([AVL["one"][:bs], 1 + p.cumprod(1).cumsum(1)], 1)                               # E[useful | L], L = 1..B
        L = (E - lam * AVL["T"]).argmax(1).add_(1).clamp_(min=lmin).to(torch.int32)
        AVL["L"][:bs].copy_(L)
        kc = AVL["ar"][None] < L[:, None]
        AVL["keepc"][:bs].copy_(kc); AVL["keep"][:bs, :, 0].copy_(kc)
        return tokens, q
    Dm.CandidateSelector.sample_path = sp2

    f = T.forward

    def g(self, x, *a, **k):
        r = f(self, x, *a, **k)
        ids = getattr(r, "topk_ids", None)
        if st["verify"] and ids is not None and "keep" in AVL and ids.shape[0] % B == 0 and hasattr(r, "_replace"):
            n = ids.shape[0]; bs = n // B
            i3 = ids.view(bs, B, -1)
            r = r._replace(topk_ids=torch.where(AVL["keep"][:bs], i3, i3[:, :1]).view(n, -1))   # one kernel per MoE layer
        return r
    T.forward = g
    C = Q.Qwen3_5MoeForConditionalGeneration
    fw = C.forward

    @functools.wraps(fw)
    def fw2(self, *a, **k):
        fb = k.get("forward_batch") or next((x for x in a if hasattr(x, "forward_mode")), None)
        st["verify"] = bool(fb is not None and fb.forward_mode.is_target_verify())
        try:
            return fw(self, *a, **k)
        finally:
            st["verify"] = False
    C.forward = fw2
    ab = W.DFlashWorkerV2._accept_block
    dbg = os.environ.get("Q37_AVL_LOG")

    @functools.wraps(ab)
    def ab2(self, *a, candidates=None, **k):
        if candidates is not None and candidates.shape[1] == B and "keepc" in AVL:
            bs = candidates.shape[0]
            candidates = torch.where(AVL["keepc"][:bs], candidates, int(self._mask_token_id))
            if dbg:
                for x in AVL["L"][:bs].tolist():
                    st["hist"][x] += 1
                st["n"] += 1
                if st["n"] % 500 == 0:
                    print("q37 avl L histogram", st["hist"], flush=True)
        return ab(self, *a, candidates=candidates, **k)
    W.DFlashWorkerV2._accept_block = ab2
    print(f"q37 avl installed: block {B}, lam {lam}, min L {lmin}, cal {cal}", flush=True)


BLOCK = int(os.environ.get("Q37_BLOCK", 8))
SLACK = 6 * BLOCK  # most any exact kv length can sit below its host bound: reserve 2 blocks + query block, doubled (48 at block 8)


def stable_chunk(ub, slack, q_tiles, padded, min_chunk=128, target=None):
    """Chunk size whose tiles fit the padded grid, with no boundary in (u - slack, u] for any request. None when every
    request fits one default chunk (the default plan then agrees with the kernel); 0 when no such size exists and the
    caller must read the exact lengths. At least one request must keep 2+ chunks: the captured graph expects a split plan."""
    top = max(ub)
    if top <= min_chunk:
        return None
    ok = lambda c: all((u + c - 1) // c == (max(u - slack, 1) + c - 1) // c for u in ub)
    # Session 8: target < padded asks for at most `target` tiles per kv head (Q37_ATTN_ONEWAVE: one wave of SMs instead of two)
    cap = min(padded, target) if target else padded
    c = min_chunk
    while sum(q_tiles * ((u + c - 1) // c) for u in ub) > cap:
        c += 1 + c // 64
    if os.environ.get("Q37_ATTN_CHUNK") and 48000 <= top <= 150000:  # isolated: 69 vs 77 us per layer at 90K; default stays best outside this range
        c = max(c, 1024)
    for c in range(c, min(top, c + 512)):  # bounded: the host must not stall searching
        if ok(c):
            return c
    for c in range(min_chunk - 1, slack, -1):
        if ok(c) and top > c and sum(q_tiles * ((u + c - 1) // c) for u in ub) <= cap:
            return c
    return 0


NSM = int(os.environ.get("Q37_NSM", "188"))  # SMs on the RTX PRO 6000 Server Edition


def install_syncfree():
    """Q37_SYNCFREE=1: FlashInfer plans verify/draft from the reserved host bound instead of waiting for the exact seq
    lens. Kernels read exact lengths from device indptr; chunk_start/end clamp to kv_len, so tiles over the bound are
    empty and merge neutral. The plan's pinned copy can now sit queued behind running GPU work, so each wrapper alternates
    two pinned buffers: the host never rewrites one still waiting to be copied (lookahead is at most one step)."""
    FI = importlib.import_module("sglang.srt.layers.attention.flashinfer_backend")
    if not os.environ.get("Q37_SF_CPU"):  # debug: keep exact host lengths, test only the plan path
        FI.FlashInferAttnBackend.needs_cpu_seq_lens = False
    f = FI.fast_prefill_plan

    @functools.wraps(f)
    def g(self, *a, **k):
        p = self._pin_memory_int_workspace_buffer
        o = getattr(self, "_q37_pin2", None)
        if o is None:
            o = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
        self._q37_pin2, self._pin_memory_int_workspace_buffer = p, o
        # The draft's compact bound leaves out the query block the device adds to kv; over-estimating is always safe.
        q = k["max_q_len"]
        lens = k["kv_lens_host"] + q
        ptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
        ptr[1:] = torch.cumsum(lens, dim=0)
        k.update(kv_lens_host=lens, kv_indptr_host=ptr, max_kv_len=int(lens.max()))
        # The kernel derives its output stride from the exact length, the merge layout from the plan's. They must agree on
        # the chunk count, so choose a chunk size with no boundary between any request's lower and upper length.
        pi = self._plan_info
        tgt = (NSM // a[5]) if os.environ.get("Q37_ATTN_ONEWAVE") else None   # one wave: tiles x kv heads <= SMs
        C = stable_chunk(lens.tolist(), SLACK, (q * a[4] // a[5] + pi[3] - 1) // pi[3], pi[0], target=tgt)
        if C:
            k["fixed_split_size"] = C
        elif C == 0:
            ex = a[1].cpu(); ex = ex[1:] - ex[:-1]
            ptr = torch.zeros(len(ex) + 1, dtype=torch.int32)
            ptr[1:] = torch.cumsum(ex, dim=0)
            k.update(kv_lens_host=ex.to(torch.int32), kv_indptr_host=ptr, max_kv_len=int(ex.max()))
        if os.environ.get("Q37_SF_DEBUG"):
            ex = a[1].cpu(); ex = (ex[1:] - ex[:-1]).tolist()
            u = lens.tolist(); c = C or 128
            bad = any((x + c - 1) // c != (y + c - 1) // c or x < y - SLACK or x > y for x, y in zip(ex, u))
            print("q37plan", id(self) % 9973, "exact", ex, "bound", u, "chunk", C, "BAD" if bad else "ok", flush=True)
        return f(self, *a, **k)
    FI.fast_prefill_plan = g

    # The drafter's sliding-window wrappers kept plain plan(): three blocking .to("cpu") per wrapper, which is what now
    # serializes the step. They get the same sync-free plan, installed after capture's real plan() built _cached_module.
    init = FI.FlashInferAttnBackend.init_forward_metadata_out_graph

    @functools.wraps(init)
    def init2(self, forward_batch, in_capture=False):
        r = init(self, forward_batch, in_capture)
        if in_capture and forward_batch.forward_mode.is_target_verify() and self.prefill_backend == "fa2"                 and self.dispatch_reason == FI.WrapperDispatch.SLIDING_WINDOW:
            for w in self.prefill_cuda_graph_metadata[forward_batch.batch_size]:
                w.begin_forward = functools.partial(FI.fast_prefill_plan, w)
        return r
    FI.FlashInferAttnBackend.init_forward_metadata_out_graph = init2
    # update_sliding_window does not forward its host lengths to call_begin_forward; the fast plan needs them.
    U = FI.FlashInferIndicesUpdaterPrefill
    usw, cbf = U.update_sliding_window, U.call_begin_forward

    @functools.wraps(usw)
    def usw2(self, *a, **k):
        self._q37_cpu = k["seq_lens_cpu"] if "seq_lens_cpu" in k else a[2]
        return usw(self, *a, **k)

    @functools.wraps(cbf)
    def cbf2(self, *a, **k):
        if k.get("seq_lens_cpu") is None:
            k["seq_lens_cpu"] = getattr(self, "_q37_cpu", None)
        return cbf(self, *a, **k)
    U.update_sliding_window, U.call_begin_forward = usw2, cbf2
    if os.environ.get("Q37_SF_SYNC"):  # debug: planning from the bound, but no run-ahead of the published lengths
        O = importlib.import_module("sglang.srt.managers.overlap_utils").FutureMap
        rs = O.resolve_seq_lens_cpu

        def rs2(self, batch):
            if self.publish_ready is not None:
                self.publish_ready.synchronize()
            return rs(self, batch)
        O.resolve_seq_lens_cpu = rs2
    print("q37 syncfree plan installed", flush=True)


def install_op_trace(path, skip=30, n=4):
    """Q37_OP_TRACE=<path>: diagnostic only. After `skip` decode rounds, logs every eager torch op of `n` whole rounds
    (get_next_batch + run_batch + process_result) with its shapes and the innermost sglang call site."""
    import traceback
    from torch.utils._python_dispatch import TorchDispatchMode
    out = open(path, "w")
    state = dict(c=0)

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ts = [a for a in args if isinstance(a, torch.Tensor)]
            if ts and any(t.is_cuda for t in ts) or (kwargs or {}).get("device"):
                fr = [f for f in traceback.extract_stack(limit=14) if "sglang" in f.filename or "flashinfer" in f.filename]
                out.write(json.dumps([state["c"], func.__name__, [list(t.shape) for t in ts], [str(t.device) for t in ts],
                                      [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in fr[-2:]]]) + "\n")
            return func(*args, **(kwargs or {}))
    S = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    for name in ("get_next_batch_to_run", "run_batch", "process_batch_result"):
        f = getattr(S, name)

        def mk(f, name):
            @functools.wraps(f)
            def g(*a, **k):
                if name == "run_batch":
                    state["c"] += 1
                if skip <= state["c"] < skip + n:
                    out.write(json.dumps(["#", name, state["c"]]) + "\n")
                    with M():
                        return f(*a, **k)
                out.flush()
                return f(*a, **k)
            return g
        setattr(S, name, mk(f, name))
    print("q37 op trace ->", path, flush=True)


def install_hot_vocab(path):
    """Q37_HOTVOCAB=<hot_ids.pt>: the drafter's candidate head reads only these rows of the shared lm_head. It proposes;
    verify still checks every token against the full head, so this can move acceptance but never correctness."""
    D = importlib.import_module("sglang.srt.models.dflash")
    f = D.candidate_topk
    hot = torch.load(path)

    @functools.wraps(f)
    def g(hidden, lm_head, k, *, with_partition=False):
        if with_partition:
            return f(hidden, lm_head, k, with_partition=True)
        w = getattr(lm_head, "_q37_hot_w", None)
        if w is None:
            lm_head._q37_hot_i = hot.to(lm_head.weight.device)
            w = lm_head._q37_hot_w = lm_head.weight[lm_head._q37_hot_i].detach().clone()
        vals, j = D._radix_topk(torch.matmul(hidden.to(w.dtype), w.T), k)
        return lm_head._q37_hot_i[j.long()], vals, None
    D.candidate_topk = g
    print("q37 hot vocabulary installed", len(hot), flush=True)


def install_accept_log(path):
    """Q37_ACCEPT_LOG=<path>: per-round counters from the server's own request state, read after process_batch_result has
    already synced: rid, verification rounds so far, output tokens so far. A round's accepted tokens = the output delta."""
    S = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    f = S.process_batch_result
    out = open(path, "a", buffering=1)  # line flush: the server is killed at the end of an arm

    @functools.wraps(f)
    def g(self, batch, result, *a, **k):
        r = f(self, batch, result, *a, **k)
        for q in getattr(batch, "reqs", None) or ():
            out.write(f"{q.rid} {q.spec_verify_ct} {len(q.output_ids)} {int(q.finished())}" + chr(10))
        return r
    S.process_batch_result = g
    print("q37 accept log ->", path, flush=True)


def install_gemm_streamk():
    """Q37_GEMM_STREAMK=1: FP8 block GEMM built from our copy of the source (fp8_blockwise_s4) with StreamK at M<=16. Same weights,
    same dtypes, same accumulator; only how the K walk is shared across SMs changes."""
    G = importlib.import_module("sglang.kernels.ops.gemm.fp8_blockwise_gemm")
    U = importlib.import_module("sglang.kernels.jit.utils")
    G._jit_fp8_blockwise_module = U.cache_once(lambda: U.load_jit(
        "fp8_blockwise_scaled_mm_s4", cuda_files=["gemm/fp8_blockwise_s4/fp8_blockwise_scaled_mm_entry.cuh"],
        cuda_wrappers=[("fp8_blockwise_scaled_mm", "fp8_blockwise_scaled_mm")], extra_dependencies=["cutlass"], extra_cuda_cflags=G._fp8_blockwise_cuda_flags()))
    print("q37 FP8 GEMM StreamK variant installed", flush=True)


def install_s4_gemm():
    """Q37_S4_GEMM=1: our K-partition FP8 GEMM (s4_splitk) for target-verify forwards at M<=16. Prefill and every other caller keep the
    CUTLASS kernel. Same weights, dtypes and quantized inputs. Q37_S4_ALWAYS=1 (isolated tests only) drops the verify check."""
    import s4_splitk
    U = importlib.import_module("sglang.srt.layers.quantization.fp8_utils")
    f = U.fp8_blockwise_scaled_mm
    state = dict(verify=bool(os.environ.get("Q37_S4_ALWAYS")))
    narrow = os.environ.get("Q37_S4_GEMM") == "narrow"  # flushed NCU: wide GEMMs are DRAM-limited and the join only adds time there
    exact = os.environ.get("Q37_S4_GEMM") == "exact"
    tile = tuple(map(int, os.environ.get("Q37_S4_TILE", "32,8,4,4").split(",")))  # block_n, block_m, stages, warps
    seen = set()

    def g(mat_a, mat_b, scales_a, scales_b, out_dtype):
        M, K = mat_a.shape
        N = mat_b.shape[1]
        if not state["verify"] or M > 16 or K % 128 or N % 128 or out_dtype != torch.bfloat16 or (narrow and (N >= 8192 or K <= 512)):
            return f(mat_a, mat_b, scales_a, scales_b, out_dtype=out_dtype)
        if exact:  # same K order and scale order as the CUTLASS kernel, so bitwise equal; only the tiles on M and N change
            if (M, N, K) not in seen:
                seen.add((M, N, K)); print("q37 s4 exact", M, N, K, tile, flush=True)
            return s4_splitk.s4_gemm(mat_a, mat_b.T, scales_a, scales_b.T, P=1, block_n=tile[0], block_m=tile[1] if M <= tile[1] else 16, num_stages=tile[2], num_warps=tile[3])
        P = s4_splitk.partitions(N, K)
        if (M, N, K) not in seen:
            seen.add((M, N, K)); print("q37 s4 gemm", M, N, K, "P", P, flush=True)
        return s4_splitk.s4_gemm(mat_a, mat_b.T, scales_a, scales_b.T, P=P)
    U.fp8_blockwise_scaled_mm = g
    if not os.environ.get("Q37_S4_ALWAYS"):
        Q = importlib.import_module("sglang.srt.models.qwen3_5")
        C = Q.Qwen3_5MoeForConditionalGeneration
        fw = C.forward

        @functools.wraps(fw)
        def fw2(self, *a, **k):
            fb = k.get("forward_batch") or next((x for x in a if hasattr(x, "forward_mode")), None)
            state["verify"] = bool(fb is not None and fb.forward_mode.is_target_verify())
            try:
                return fw(self, *a, **k)
            finally:
                state["verify"] = False
        C.forward = fw2
        if os.environ.get("Q37_S4_PREFETCH"):
            # Producer: read-only touch of out_proj's weights on the alt stream, right after the input projections. Consumer: out_proj, after
            # conv, factor and solve, which leave DRAM nearly idle. Nothing waits on it and no tensor changes, so it cannot affect results.
            G = Q.Qwen3_5GatedDeltaNet
            fip = G._forward_input_proj
            cap = importlib.import_module("sglang.srt.model_executor.runner").get_is_capture_mode

            @functools.wraps(fip)
            def fip2(self, hidden_states):
                r = fip(self, hidden_states)
                if state["verify"] and cap() and self.alt_stream is not None:
                    w = self.out_proj.weight
                    if not hasattr(self, "_q37_sink"):
                        self._q37_sink = torch.empty(4096, device=w.device, dtype=torch.float32)
                    with torch.cuda.stream(self.alt_stream):
                        s4_splitk.touch(w, self._q37_sink)
                return r
            G._forward_input_proj = fip2
    print("q37 s4 gemm installed", flush=True)


def install_state_check(path):
    """Q37_STATE_CHECK=<path>: correctness debug, syncs every round, never timed. Per verify round: context length, drafted tokens, top-5
    target logits and row norms for every position, accepted length, and per-layer norm and sum of the request's committed GDN state."""
    D = importlib.import_module("sglang.srt.speculative.dflash_worker_v2").DFlashWorkerV2
    out = open(path, "a", buffering=1)
    acc, upd = D._accept_block, D._update_target_mamba_state_after_verify

    @functools.wraps(acc)
    def a(self, *x, **k):
        lg = k["next_token_logits"].float()
        v, i = torch.topk(lg, 5, dim=-1)
        self._q37_row = dict(prefix=k["prefix_lens"].tolist(), cand=k["candidates"].view(-1).tolist(), top_i=i.tolist(), top_v=v.tolist(), norm=lg.norm(dim=-1).tolist())
        return acc(self, *x, **k)

    @functools.wraps(upd)
    def u(self, *x, **k):
        r = upd(self, *x, **k)
        torch.cuda.synchronize()
        row = getattr(self, "_q37_row", {})
        b = k["batch"]
        slot = b.req_pool_indices[:k["commit_lens"].shape[0]].long()
        t = self.target_worker.model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[:, slot].double()
        row.update(commit=k["commit_lens"].tolist(), pre=k["seq_lens_pre_verify"].tolist(), post=k["seq_lens_post_verify"].tolist(),
                   state_norm=t.flatten(1).norm(dim=1).tolist(), state_sum=t.flatten(1).sum(dim=1).tolist())
        out.write(json.dumps(row) + chr(10))
        return r
    D._accept_block, D._update_target_mamba_state_after_verify = a, u
    print("q37 state check ->", path, flush=True)


def install_prefetch(kinds):
    """Q37_PF=out,next: read-only L2 prefetch on the alt stream, launched after a GDN layer's input projections, while conv, factor and
    solve leave DRAM nearly idle. out = this layer's out_proj; next = the next layer's input projection. Verify graphs only. Nothing
    waits on it and no tensor it reads or writes reaches the model, so outputs must stay bit-identical to the same kernels without it."""
    import s4_splitk
    Q = importlib.import_module("sglang.srt.models.qwen3_5")
    cap = importlib.import_module("sglang.srt.model_executor.runner").get_is_capture_mode
    C, G = Q.Qwen3_5MoeForConditionalGeneration, Q.Qwen3_5GatedDeltaNet
    state = dict(verify=False)
    nxt = {}
    fw, fip = C.forward, G._forward_input_proj

    def input_weight(layer):
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            w = getattr(la, "_fused_in_proj_weight", None)
            return w if w is not None else la.in_proj_qkvz.weight
        for m in layer.modules():
            if hasattr(m, "qkv_proj"):
                return m.qkv_proj.weight
        return None

    @functools.wraps(fw)
    def fw2(self, *a, **k):
        if not nxt:
            L = [m for m in self.modules() if isinstance(m, (Q.Qwen3_5LinearDecoderLayer, Q.Qwen3_5AttentionDecoderLayer))]
            for x, y in zip(L, L[1:]):
                if isinstance(x, Q.Qwen3_5LinearDecoderLayer):
                    nxt[x.linear_attn] = input_weight(y)
            print("q37 prefetch: GDN layers mapped", len(nxt), "next-layer MB", sorted({round(w.numel() / 1e6, 1) for w in nxt.values() if w is not None}), flush=True)
        fb = k.get("forward_batch") or next((x for x in a if hasattr(x, "forward_mode")), None)
        state["verify"] = bool(fb is not None and fb.forward_mode.is_target_verify())
        try:
            return fw(self, *a, **k)
        finally:
            state["verify"] = False

    @functools.wraps(fip)
    def fip2(self, hidden_states):
        on = state["verify"] and cap() and self.alt_stream is not None
        if on and not hasattr(self, "_q37_sink"):
            self._q37_sink = torch.empty(4096, device=self.out_proj.weight.device, dtype=torch.float32)
        if on and "ba" in kinds:  # in_proj_ba's 0.26 MB ahead of it, on its own stream; in_proj_qkvz on the main stream does not wait for this
            self.alt_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.alt_stream):
                s4_splitk.touch(self.in_proj_ba.weight, self._q37_sink)
        r = fip(self, hidden_states)
        if on:
            ws = ([self.out_proj.weight] if "out" in kinds else []) + ([nxt[self]] if "next" in kinds and nxt.get(self) is not None else [])
            with torch.cuda.stream(self.alt_stream):
                for w in ws:
                    s4_splitk.touch(w, self._q37_sink)
        return r
    C.forward, G._forward_input_proj = fw2, fip2
    print("q37 prefetch installed", kinds, flush=True)


def install_kv_par(blocks):
    """Q37_KV_PAR=<n>: the per-round KV index copy on a (batch, n) grid instead of one CTA walking the whole context. Same integers in the
    same places, so outputs must stay bit-identical; only how many SMs share the copy changes."""
    mods = [importlib.import_module(m) for m in ("sglang.srt.mem_cache.kv_index_translator", "sglang.srt.speculative.dflash_info")]

    class Par:
        def __init__(self, k):
            self.k = k

        def __getitem__(self, grid):
            def launch(*a, **kw):
                if kw.get("TOKEN_BLOCK_PARALLEL") or len(grid) != 1 or grid[0] > 8:
                    return self.k[grid](*a, **kw)
                return self.k[(grid[0], blocks)](*a, TOKEN_BLOCK_PARALLEL=True, **kw)
            return launch
    n = 0
    for m in mods:
        if hasattr(m, "create_flashinfer_kv_indices_triton"):
            m.create_flashinfer_kv_indices_triton = Par(m.create_flashinfer_kv_indices_triton); n += 1
    print("q37 kv index copy parallel", blocks, "modules", n, flush=True)


def install_track_guard():
    """Q37_TRACK_GUARD=1: skip the radix-track Bole commit when no request can cross a track boundary this round. Every row of that launch
    would be invalid, so the kernel writes nothing. The bound comes from host request lengths, which lag the device by at most one round:
    device length before this verify <= host + 2 blocks, after it <= host + 3 blocks; 4 blocks leaves margin. Q37_TRACK_CHECK=1 syncs and verifies each skip."""
    D = importlib.import_module("sglang.srt.speculative.dflash_worker_v2")
    B = importlib.import_module("sglang.srt.layers.attention.linear.bole_spec").BoleSpecVerifier
    upd, com = D.DFlashWorkerV2._update_target_mamba_state_after_verify, B.commit
    check = os.environ.get("Q37_TRACK_CHECK")
    st = dict(skip=False, skipped=0, ran=0, bad=0)

    @functools.wraps(upd)
    def u(self, *a, **k):
        b = k["batch"]
        I = D.mamba_track_grid(b.tree_cache.page_size)
        ls = [len(q.origin_input_ids) + len(q.output_ids) for q in b.reqs]
        st["skip"] = bool(ls) and all((L + 4 * BLOCK) // I == L // I for L in ls)
        return upd(self, *a, **k)

    @functools.wraps(com)
    def c(self, buffer_bs, slots, steps, track_slots=None, track_steps=None):
        if track_slots is not None and st["skip"]:
            st["skipped"] += 1
            if check and bool((track_steps >= 0).any()):
                st["bad"] += 1; print("q37 track guard WRONG SKIP", track_steps.tolist(), flush=True)
            if check and st["skipped"] % 500 == 0:
                print("q37 track guard skipped", st["skipped"], "ran", st["ran"], "wrong", st["bad"], flush=True)
            return com(self, buffer_bs, slots, steps)
        st["ran"] += 1
        return com(self, buffer_bs, slots, steps, track_slots, track_steps)
    D.DFlashWorkerV2._update_target_mamba_state_after_verify, B.commit = u, c
    print("q37 track guard installed", flush=True)


def install_bole_warps(spec):
    """Q37_BOLE_WARPS=factor:8,solve:8: thread count of the Bole verify kernels. Same operations on the same values, only the block size."""
    M = importlib.import_module("sglang.srt.layers.attention.linear.bole_spec")
    kv = dict(x.split(":") for x in spec.split(","))
    bv = int(kv.pop("solvebv", 0))   # solvebv:16 -> the solve's V-column tile (positional arg 18, grid axis 0 = cdiv(dv, BV)); columns are independent
    want = {f"bole_{k}_kernel": int(v) for k, v in kv.items()}
    if bv:
        want.setdefault("bole_solve_kernel", 0)

    class W:
        def __init__(self, k, n, bv=0):
            self.k, self.n, self.bv = k, n, bv

        def __getitem__(self, grid):
            def launch(*a, **kw):
                if self.n:
                    kw["num_warps"] = self.n
                g = grid
                if self.bv and len(a) > 18 and a[18] == 32:
                    a = list(a); a[18] = self.bv; g = ((a[15] + self.bv - 1) // self.bv,) + tuple(grid[1:])
                return self.k[g](*a, **kw)
            return launch
    for name, n in want.items():
        setattr(M, name, W(getattr(M, name), n, bv if name == "bole_solve_kernel" else 0))
    print("q37 bole warps", want, "solve BV", bv or 32, flush=True)


def install_ba_tiny():
    """Q37_BA_TINY=1: in_proj_ba (N=64, K=2048, M<=16, bf16) through the package's tiny_gemm_bf16 instead of cuBLAS: prior art, not our kernel.
    Other BF16 GEMMs are untouched. Reduction order differs from cuBLAS, so outputs are not expected to be bit-identical."""
    U = importlib.import_module("sglang.srt.layers.quantization.unquant")
    T = importlib.import_module("sglang.kernels.ops.gemm.tiny_gemm")
    C = U.UnquantizedLinearMethod
    f = C.apply
    n = dict(c=0)

    @functools.wraps(f)
    def g(self, layer, x, bias=None):
        if bias is None and x.is_cuda and x.ndim == 2 and x.shape[0] <= 16 and layer.weight.shape == (64, 2048) and x.dtype == torch.bfloat16:
            if not n["c"]:
                print("q37 in_proj_ba tiny gemm used", tuple(x.shape), flush=True)
            n["c"] += 1
            return T.tiny_gemm_bf16(x, layer.weight)
        return f(self, layer, x, bias)
    C.apply = g
    print("q37 in_proj_ba tiny gemm installed", flush=True)


def install_ba_exact():
    """Q37_BA_EXACT=threads[,span]: in_proj_ba (N=64, K=2048, M<=16, bf16) through our s4_ba kernel: same WMMA sequence as the cuBLAS kernel and bitwise equal
    to it, but weight tiles are copied to shared memory without blocking first, so qkvz saturating DRAM beside it cannot stall each K step."""
    import ctypes
    U = importlib.import_module("sglang.srt.layers.quantization.unquant")
    C = U.UnquantizedLinearMethod
    f = C.apply
    lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s4_kernels", "s4_wmma.so"))
    lib.s4_ba.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int]
    T, L = (list(map(int, os.environ["Q37_BA_EXACT"].split(","))) + [1024])[:2]
    n = dict(c=0, off=0)

    @functools.wraps(f)
    def g(self, layer, x, bias=None):
        w = layer.weight
        if bias is None and x.is_cuda and x.ndim == 2 and x.shape[0] <= 16 and w.shape == (64, 2048) and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            if x.is_contiguous() and w.is_contiguous() and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0:
                if not n["c"]:
                    print("q37 ba exact used", tuple(x.shape), T, L, flush=True)
                n["c"] += 1
                o = x.new_empty((x.shape[0], 64))
                rc = lib.s4_ba(o.data_ptr(), x.data_ptr(), w.data_ptr(), x.shape[0], 2048, 64, T, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), L)
                assert rc == 0, rc
                return o
            n["off"] += 1
            print("q37 ba exact fell back, unaligned or strided", n["off"], flush=True)
        return f(self, layer, x, bias)
    C.apply = g
    print("q37 ba exact installed", T, L, flush=True)


def install_conv_tile():
    """Q37_CONV_TILE=block_n,warps: conv update kernel with a narrower feature tile, so more SMs share it. Per-element arithmetic does not change."""
    M = importlib.import_module("sglang.kernels.ops.mamba.causal_conv1d_triton")
    k0 = M._causal_conv1d_update_kernel
    bn, nw = map(int, os.environ["Q37_CONV_TILE"].split(","))

    class P:
        def __getitem__(self, grid):
            def run(*a, **k):
                k = dict(k)
                k["BLOCK_N"] = bn
                k["num_warps"] = nw
                return k0[grid](*a, **k)
            return run
    M._causal_conv1d_update_kernel = P()
    print("q37 conv tile installed", bn, nw, flush=True)


def install_kv_fuse():
    """Q37_KV_FUSE=1: FP8 KV write as one kernel (s4_kv) when both layer scales are exactly 1.0, so the divides do nothing. Anything else uses the original path."""
    import s4_kv
    Mp = importlib.import_module("sglang.srt.mem_cache.memory_pool")
    C = Mp.MHATokenToKVPool
    f = C.set_kv_buffer
    fp8 = torch.float8_e4m3fn
    n = dict(c=0, off=0)

    @functools.wraps(f)
    def g(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None, dcp_kv_mask=None):
        if (type(self) is C and k_scale is not None and v_scale is not None and dcp_kv_mask is None and layer is not None and getattr(layer, "k_scale_float", None) == 1.0
                and getattr(layer, "v_scale_float", None) == 1.0 and not self.is_quantized_kv_cache and not self.use_hnd and self.kv_cache_layout != "vectorized_5d"
                and self.dtype == fp8 and cache_k.dtype == torch.bfloat16 and cache_v.dtype == torch.bfloat16 and cache_k.ndim == 3 and cache_k.shape == cache_v.shape and cache_k.shape[0] <= 16):
            loc = Mp.unwrap_write_loc(loc_info)[0]
            i = (layer_id_override if layer_id_override is not None else layer.layer_id) - self.start_layer
            if not n["c"]:
                print("q37 kv fuse used", tuple(cache_k.shape), self.k_buffer[i].shape, flush=True)
            n["c"] += 1
            N, H, D = cache_k.shape
            s4_kv.store(cache_k, cache_v, self.k_buffer[i].view(-1, H * D).view(fp8), self.v_buffer[i].view(-1, H * D).view(fp8), loc)
            return
        n["off"] += 1
        if n["off"] <= 3:
            print("q37 kv fuse original path", type(self).__name__, cache_k.shape, getattr(layer, "k_scale_float", None), flush=True)
        return f(self, layer, loc_info, cache_k, cache_v, k_scale, v_scale, layer_id_override, dcp_kv_mask)
    C.set_kv_buffer = g
    print("q37 kv fuse installed", flush=True)


def install_mono():
    """Q37_MONO=1 [Q37_MONO_CFG=...]: FlashInfer MonoMoE (prior art) for routed experts at M<=8, one stored weight copy. Gate/up is repacked in
    place after load. M>8 (prefill) restores the original row order for one layer into a shared scratch and runs the unchanged Triton path."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mono_lib as L
    if os.environ.get("Q37_MONO_CFG"):
        L.variant(os.environ["Q37_MONO_CFG"])
    from flashinfer.fused_moe.monomoe import alloc_scratchpad, has_monomoe, mono_moe
    nopair = "USE_PAIR_LAYOUT=false" in os.environ.get("Q37_MONO_CFG", "")
    Q = importlib.import_module("sglang.srt.models.qwen2_moe")
    B = Q.Qwen2MoeSparseMoeBlock
    R = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    f, load = B._forward_router_experts, R.load_model
    S = {}
    n = dict(mono=0, tri=0)

    @functools.wraps(load)
    def load2(self, *a, **k):
        out = load(self, *a, **k)
        blocks = [m for m in self.model.modules() if isinstance(m, B)]
        if not blocks:
            return out
        if not has_monomoe():
            print("q37 mono UNAVAILABLE, Triton kept", flush=True)
            return out
        dev = blocks[0].experts.w13_weight.device
        p, inv = L.row_perm(dev)
        c = 0
        for b in blocks:
            e = b.experts
            w = e.w13_weight
            if b.enable_shared_expert_fusion or tuple(w.shape) != L.SHAPE_UP or w.dtype != torch.float8_e4m3fn or tuple(e.w2_weight.shape) != (L.E, L.K, L.N):
                print("q37 mono layer skipped", tuple(w.shape), w.dtype, flush=True)
                continue
            if not nopair:
                L.repack(w, p)
            e._q37_mono = dict(topk=b.topk.topk_config.top_k, renorm=b.topk.topk_config.renormalize)
            c += 1
        S.update(inv=inv, pad=alloc_scratchpad(dev), scratch=torch.nn.Parameter(torch.empty(L.SHAPE_UP, dtype=torch.float8_e4m3fn, device=dev), requires_grad=False) if not nopair else None)
        print(f"q37 mono ready: {c} of {len(blocks)} layers, repacked={not nopair}, module {L.module_path()}", flush=True)
        return out
    R.load_model = load2

    @functools.wraps(f)
    def g(self, hidden_states, *, defer_finalize=False):
        e = self.experts
        st = getattr(e, "_q37_mono", None)
        if st is None or defer_finalize:
            return f(self, hidden_states, defer_finalize=defer_finalize)
        M = hidden_states.shape[0]
        if 0 < M <= 8:
            logits, _ = self.gate(hidden_states)
            if not n["mono"]:
                print("q37 mono used", tuple(hidden_states.shape), logits.dtype, st, flush=True)
            n["mono"] += 1
            return mono_moe(hidden_states, logits, e.w13_weight, e.w13_weight_scale_inv, e.w2_weight, e.w2_weight_scale_inv, st["topk"],
                            scoring_func="softmax", renormalize=st["renorm"], out=hidden_states.new_empty((M, L.K)), scratchpad=S["pad"], interleave_up=False)
        if nopair or not M:
            return f(self, hidden_states, defer_finalize=False)
        if not n["tri"]:
            print("q37 mono prefill path", tuple(hidden_states.shape), flush=True)
        n["tri"] += 1
        w = e._parameters["w13_weight"]
        L.restore_into(S["scratch"], w, S["inv"])
        e._parameters["w13_weight"] = S["scratch"]
        try:
            return f(self, hidden_states, defer_finalize=False)
        finally:
            e._parameters["w13_weight"] = w
    B._forward_router_experts = g
    print("q37 mono installed", flush=True)


def install_dump_hidden(path):
    """Q37_DUMP_HIDDEN=<path>: debug only, never timed. Saves up to 800 small hidden batches, for the offline verify-head bound check."""
    P = importlib.import_module("sglang.srt.layers.logits_processor").LogitsProcessor
    f = P._get_logits
    keep = []

    @functools.wraps(f)
    def g(self, hidden_states, lm_head, *a, **k):
        if not torch.cuda.is_current_stream_capturing() and len(keep) < 800 and hidden_states.ndim == 2 and hidden_states.shape[0] <= 8 and lm_head.weight.shape[0] > 100000:
            keep.append(hidden_states.detach().float().cpu())
            if len(keep) % 100 == 0:
                torch.save(keep, path)
        return f(self, hidden_states, lm_head, *a, **k)
    P._get_logits = g
    print("q37 dump hidden installed", flush=True)


def install_lattice_dump(path):
    """Q37_LATTICE_DUMP=<dir>: debug only. Every BS1 draft writes its selector lattice (candidate ids [7,K], edge scores [7,K,K], anchor)
    into a device ring from inside the captured draft graph (no host sync). When a request finishes, the new rows are saved as one part.
    Rows are aligned to output ids offline by anchor token, so warm-up and capture rows are harmless."""
    W = importlib.import_module("sglang.srt.speculative.dflash_worker_v2")
    S = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    f = W._selector_lattice
    N = 32768
    st = {}
    os.makedirs(path, exist_ok=True)

    def g(draft_model, pred_hidden, anchor_token_ids):
        ids, sc = f(draft_model, pred_hidden, anchor_token_ids)
        if ids.shape[0] == 1:
            if not st:
                d = ids.device
                st.update(ids=torch.zeros((N,) + tuple(ids.shape[1:]), dtype=torch.int32, device=d),
                          sc=torch.zeros((N,) + tuple(sc.shape[1:]), dtype=torch.float32, device=d),
                          an=torch.zeros((N,), dtype=torch.int64, device=d), ctr=torch.zeros((1,), dtype=torch.int64, device=d), last=0, part=0)
            i = torch.remainder(st["ctr"], N)
            st["ids"].index_copy_(0, i, ids[:1].to(torch.int32))
            st["sc"].index_copy_(0, i, sc[:1].float())
            st["an"].index_copy_(0, i, anchor_token_ids[:1].reshape(1).to(torch.int64))
            st["ctr"].add_(1)
        return ids, sc
    W._selector_lattice = g

    h = S.process_batch_result

    @functools.wraps(h)
    def hb(self, batch, result, *a, **k):
        r = h(self, batch, result, *a, **k)
        if st and any(q.finished() for q in (getattr(batch, "reqs", None) or ())):
            c = int(st["ctr"].item()); lo = max(st["last"], c - N)
            if c > lo:
                j = torch.arange(lo, c, device=st["ctr"].device) % N
                torch.save({"start": lo, "end": c, "ids": st["ids"][j].cpu(), "sc": st["sc"][j].cpu(), "an": st["an"][j].cpu(),
                            "rids": [q.rid for q in batch.reqs if q.finished()],
                            "out": {q.rid: list(q.output_ids) for q in batch.reqs if q.finished()},
                            "prompt": {q.rid: list(q.origin_input_ids) for q in batch.reqs if q.finished()}}, os.path.join(path, f"part_{st['part']:05d}.pt"))
                st["last"] = c; st["part"] += 1
        return r
    S.process_batch_result = hb
    print("q37 lattice dump ->", path, flush=True)


def install_pld():
    """Q37_PLD=1 (optional Q37_PLD_MIN_N=2..4): device-side prompt lookup replaces the drafter's 7-token chain when the history's
    suffix occurred before (pld.py). Same verify width, same target computation; only the proposal changes."""
    W = importlib.import_module("sglang.srt.speculative.dflash_worker_v2")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pld import PLD
    C = W.DFlashWorkerV2
    init, fbg, acc = C.__init__, C.forward_batch_generation, C._accept_block
    min_n = int(os.environ.get("Q37_PLD_MIN_N", "2"))
    bias = float(os.environ.get("Q37_PLD_BIAS", "0"))
    PB = {}
    if bias:
        # The lattice is built inside the captured draft graph: add bias where a candidate equals the lookup token at its slot.
        # PB["prop"] is a static buffer (allocated at capture) that the host-side proposal fills before every replay.
        f_lat = W._selector_lattice

        def lat(draft_model, pred_hidden, anchor_token_ids):
            ids, sc = f_lat(draft_model, pred_hidden, anchor_token_ids)
            if "prop" not in PB:
                PB["prop"] = torch.full((64, ids.shape[1]), -1, dtype=torch.int64, device=ids.device)
            hit = (ids == PB["prop"][: ids.shape[0], :, None]).to(sc.dtype)
            return ids, sc + bias * hit[:, :, None, :]
        W._selector_lattice = lat

    @functools.wraps(init)
    def init2(self, *a, **k):
        init(self, *a, **k)
        self._q37_pld = None  # the request pool is allocated after the worker is built: created at the first prefill
        self._q37_pool = None
        dr = self.draft_model_runner
        f0 = dr.forward

        def fwd(forward_batch, *a2, **k2):
            is_block = self._q37_pld is not None and forward_batch.spec_info is getattr(self, "_draft_block_spec_info", None)
            if bias and is_block:
                bs = forward_batch.batch_size
                self._q37_pool = forward_batch.req_pool_indices
                if "prop" in PB:
                    self._q37_pld.propose_into(self._q37_pool, PB["prop"][:bs].view(-1))
                return f0(forward_batch, *a2, **k2)
            out = f0(forward_batch, *a2, **k2)
            if (not bias and self._q37_pld is not None and forward_batch.spec_info is getattr(self, "_draft_block_spec_info", None) and self._draft_sampler is not None
                    and getattr(out, "can_run_graph", False)):
                bs = forward_batch.batch_size
                self._q37_pool = forward_batch.req_pool_indices
                self._q37_pld.propose(self._q37_pool, self._draft_sampler.out[: bs * 7])
            return out
        dr.forward = fwd
        print("q37 pld worker wrapped, min_n", min_n, flush=True)

    @functools.wraps(fbg)
    def fbg2(self, batch, *a, **k):
        # Decided before the call: the worker switches a decode batch to TARGET_VERIFY, which is_extend() also reports.
        is_prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        res = fbg(self, batch, *a, **k)
        if is_prefill:
            if self._q37_pld is None:
                r2t = self.target_worker.model_runner.req_to_token_pool.req_to_token
                self._q37_pld = PLD(r2t.shape[0], r2t.shape[1] + 1024, r2t.device, min_n=min_n)
                print("q37 pld buffers: slots", r2t.shape[0], "maxlen", r2t.shape[1] + 1024, flush=True)
            nxt = res.next_token_ids
            slots = batch.req_pool_indices_cpu.tolist() if batch.req_pool_indices_cpu is not None else batch.req_pool_indices.tolist()
            for i, q in enumerate(batch.reqs):
                self._q37_pld.seed(int(slots[i]), list(q.origin_input_ids) + list(q.output_ids), nxt[i])
        return res

    @functools.wraps(acc)
    def acc2(self, *a, **k):
        r = acc(self, *a, **k)
        if self._q37_pool is not None:
            self._q37_pld.append(self._q37_pool, r[3], r[1])
            self._q37_pool = None
        return r
    C.__init__, C.forward_batch_generation, C._accept_block = init2, fbg2, acc2
    print("q37 pld hooks set", flush=True)


def install_hidden_ring(path):
    """Q37_HIDDEN_RING=<file>: debug only. Verify-head inputs (8 rows) go into a device ring from inside the captured graph; when a
    request finishes, up to 2,000 rounds are saved as a list of [8, H] float tensors (the head_bound.py input format)."""
    P = importlib.import_module("sglang.srt.layers.logits_processor").LogitsProcessor
    S = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    f = P._get_logits
    N = 2000
    st = {}

    @functools.wraps(f)
    def g(self, hidden_states, lm_head, *a, **k):
        if hidden_states.ndim == 2 and hidden_states.shape[0] == 8 and lm_head.weight.shape[0] > 200000:
            if not st:
                st.update(buf=torch.zeros((N, 8, hidden_states.shape[1]), dtype=hidden_states.dtype, device=hidden_states.device),
                          ctr=torch.zeros((1,), dtype=torch.int64, device=hidden_states.device))
            st["buf"].index_copy_(0, torch.remainder(st["ctr"], N), hidden_states[None])
            st["ctr"].add_(1)
        return f(self, hidden_states, lm_head, *a, **k)
    P._get_logits = g
    h = S.process_batch_result

    @functools.wraps(h)
    def hb(self, batch, result, *a, **k):
        r = h(self, batch, result, *a, **k)
        if st and any(q.finished() for q in (getattr(batch, "reqs", None) or ())):
            c = min(int(st["ctr"].item()), N)
            torch.save([x.float().cpu() for x in st["buf"][:c]], path)
        return r
    S.process_batch_result = hb
    print("q37 hidden ring ->", path, flush=True)


def install_s5_dense():
    """Q37_S5_DENSE=64|32: decode-verify FP8 dense GEMMs (M<=16, K>=2048, N>=2048: out_proj, o_proj, qkvz, attention qkv) through our
    narrow-tile variant of the SM120 swapAB kernel (s5_kernels): weight tile 64 or 32 rows instead of 128, same K walk, promotion and
    epilogue, weight scales repeated per sub-block so each row uses its original value. Expected bit-exact (Track E; s5_dense_exact.py).
    Shared-expert GEMMs (N=1024 or K=512) keep the stock kernel so they do not take SMs from the routed experts beside them.
    Any shape or scale not prepared at load falls back to the stock kernel and is counted."""
    T = int(os.environ["Q37_S5_DENSE"])
    G = importlib.import_module("sglang.kernels.ops.gemm.fp8_blockwise_gemm")
    U = importlib.import_module("sglang.kernels.jit.utils")
    R = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    name = f"fp8_blockwise_scaled_mm_n{T}"
    flags = list(G._fp8_blockwise_cuda_flags()) + (["-DS5_TILE32"] if T == 32 else [])
    new = U.cache_once(lambda: U.load_jit(f"fp8_blockwise_scaled_mm_s5_{T}", cuda_files=["gemm/fp8_blockwise_s5/fp8_blockwise_scaled_mm_entry.cuh"],
                                          cuda_wrappers=[(name, name)], extra_dependencies=["cutlass"], extra_cuda_cflags=flags))
    orig = G._jit_fp8_blockwise_module
    rep = {}
    n = dict(used=0, fb_shape=0, fb_scale=0)

    MM = int(os.environ.get("Q37_S5_MMAX", "16"))   # 24 only after s5_dense_exact.py proves M=17..24 bitwise equal (block 24)

    def eligible(M, K, N):
        return M <= MM and K >= 2048 and N >= 2048 and N % 128 == 0 and K % 128 == 0

    class Proxy:
        def __init__(self, base):
            self.base = base

        def __getattr__(self, k):
            return getattr(self.base, k)

        def fp8_blockwise_scaled_mm(self, out, a, b, sa, sb):
            M, K, N = a.shape[0], b.shape[0], b.shape[1]
            if eligible(M, K, N) and out.dtype == torch.bfloat16:
                r = rep.get(sb.data_ptr())
                if r is not None and r.shape[1] == N // T and r.shape[0] == sb.shape[0]:
                    if not n["used"]:
                        print(f"q37 s5 dense used M={M} K={K} N={N} tile {T}", flush=True)
                    n["used"] += 1
                    return getattr(new(), name)(out, a, b, sa, r)
                n["fb_scale"] += 1
                if n["fb_scale"] <= 3:
                    print(f"q37 s5 dense fell back (scale not prepared) K={K} N={N} sb {tuple(sb.shape)} {tuple(sb.stride())}", flush=True)
            elif M <= MM and K >= 2048 and N >= 2048:
                n["fb_shape"] += 1
            return self.base.fp8_blockwise_scaled_mm(out, a, b, sa, sb)

    G._jit_fp8_blockwise_module = U.cache_once(lambda: Proxy(orig()))
    load = R.load_model

    @functools.wraps(load)
    def load2(self, *a, **k):
        out = load(self, *a, **k)
        c = 0
        for mod in self.model.modules():
            w, s_ = getattr(mod, "weight", None), getattr(mod, "weight_scale_inv", None)
            if w is None or s_ is None or not isinstance(w, torch.Tensor) or w.dtype != torch.float8_e4m3fn or w.ndim != 2 or s_.ndim != 2:
                continue
            N, K = w.shape
            if not eligible(1, K, N) or tuple(s_.shape) != (N // 128, K // 128):
                continue
            # The kernel takes the scale as [K/128, N/T] K-major, like the original [K/128, N/128] view of weight_scale_inv [N/128, K/128].
            r = s_.float().repeat_interleave(128 // T, dim=0).contiguous().t()
            rep[s_.data_ptr()] = r
            c += 1
        print(f"q37 s5 dense: {c} weight scales prepared, tile {T}", flush=True)
        return out
    R.load_model = load2
    print("q37 s5 dense installed, tile", T, flush=True)
    return n


def install_s5_ba():
    """Q37_S5_BA=threads: in_proj_ba (N=64, K=2048, M<=16, bf16) through s5_ba: one CTA per 8 output columns (8 CTAs instead of 4), same
    mma.m16n8k16 sequence per element as s4_ba / cuBLAS, loads overlapped with the MMA chain. Expected bit-exact (s5_dense_exact.py)."""
    import ctypes
    Un = importlib.import_module("sglang.srt.layers.quantization.unquant")
    C = Un.UnquantizedLinearMethod
    f = C.apply
    lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_kernels", "s5_ba.so"))
    lib.s5_ba.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    T = int(os.environ["Q37_S5_BA"])
    n = dict(c=0)

    @functools.wraps(f)
    def g(self, layer, x, bias=None):
        w = layer.weight
        if (bias is None and x.is_cuda and x.ndim == 2 and x.shape[0] <= 16 and w.shape == (64, 2048) and x.dtype == torch.bfloat16
                and w.dtype == torch.bfloat16 and x.is_contiguous() and w.is_contiguous() and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0):
            if not n["c"]:
                print("q37 s5 ba used", tuple(x.shape), T, flush=True)
            n["c"] += 1
            o = x.new_empty((x.shape[0], 64))
            rc = lib.s5_ba(o.data_ptr(), x.data_ptr(), w.data_ptr(), x.shape[0], 2048, 64, T, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
            assert rc == 0, rc
            return o
        return f(self, layer, x, bias)
    C.apply = g
    print("q37 s5 ba installed", T, flush=True)


def install_s5_moe():
    """Q37_S5_MOE=<variant 0-7>: routed experts at decode verify (M<=8, top-8 of 256, FP8 block 128, silu, no bias) through our two
    persistent kernels (s5_kernels/s5_moe.cu): stock activation quant -> K1 gate/up + SiLU-mul -> stock quant -> K2 down + routed weight
    + fixed-order slot sum. Replaces moe_align, both Triton GEMMs, silu_and_mul and moe_sum. Anything else runs the stock path.
    The variant picks the rounding details (see s5_moe.cu); s5_moe_exact.py chooses it by bitwise comparison with the stock path."""
    import ctypes, inspect
    nbl = {1: 0, 2: 1, 4: 2}[int(os.environ.get("Q37_S5_MOE_NB", "1"))]
    var = int(os.environ["Q37_S5_MOE"]) | (nbl << 4)          # bits 4-5: log2 of 128-K blocks per pipeline step (Session 7)
    grid = int(os.environ.get("Q37_S5_MOE_GRID", "188"))       # CTAs; fewer leaves SMs for the shared expert beside it
    Fm = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")
    Km = importlib.import_module("sglang.kernels.ops.moe.fused_moe_triton_kernels")
    R = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    src = inspect.getsource(Km.invoke_fused_moe_kernel)
    quant = next((getattr(Km, nm) for nm in ("sglang_per_token_group_quant_fp8", "per_token_group_quant_fp8") if nm in src and hasattr(Km, nm)), None)
    if quant is None:
        print("q37 s5 moe UNAVAILABLE: activation quant function not found, stock kept", flush=True)
        return
    lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_kernels", "s5_moe.so"))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.s5_moe_up.argtypes = [ci, ci, vp, vp, vp, ci, ci, vp, vp, vp, ci, vp, ci]
    lib.s5_moe_down.argtypes = [ci, ci, vp, vp, vp, ci, ci, vp, vp, vp, ci, vp, vp, vp, vp, ci]
    f = Fm.fused_experts
    B = {}
    n = dict(used=0, stock=0)

    load = R.load_model

    @functools.wraps(load)
    def load2(self, *a, **k):
        out = load(self, *a, **k)
        if "c2" not in B and any(p.dtype == torch.float8_e4m3fn for p in self.model.parameters()):
            d = next(self.model.parameters()).device
            B.update(c2=torch.empty((64, 512), dtype=torch.bfloat16, device=d), c3=torch.empty((64, 2048), dtype=torch.bfloat16, device=d),
                     cnt=torch.zeros((128,), dtype=torch.int32, device=d))
            print("q37 s5 moe buffers ready", flush=True)
        return out
    R.load_model = load2

    @functools.wraps(f)
    def g(hidden_states, w1, w2, topk_output, moe_runner_config, b1=None, b2=None, use_fp8_w8a8=False, use_int8_w8a8=False,
          use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, w1_scale=None, w2_scale=None, w1_zp=None, w2_zp=None,
          a1_scale=None, a2_scale=None, block_shape=None, a1_q=None, fuse_swiglu_interleaved=False):
        c = moe_runner_config
        tw, ids = topk_output[0], topk_output[1]
        M = hidden_states.shape[0]
        ok = ("c2" in B and 1 <= M <= 8 and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous() and use_fp8_w8a8
              and not (use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16 or per_channel_quant or fuse_swiglu_interleaved)
              and list(block_shape or []) == [128, 128] and b1 is None and b2 is None and a1_scale is None and a2_scale is None
              and tuple(w1.shape) == (256, 1024, 2048) and tuple(w2.shape) == (256, 2048, 512)
              and w1_scale is not None and tuple(w1_scale.shape) == (256, 8, 16) and w2_scale is not None and tuple(w2_scale.shape) == (256, 16, 4)
              and w1_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32 and w1.is_contiguous() and w2.is_contiguous()
              and tuple(ids.shape) == (M, 8) and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
              and tw.dtype == torch.float32 and tw.is_contiguous()
              and getattr(c, "activation", "silu") == "silu" and getattr(c, "is_gated", True) and not getattr(c, "apply_router_weight_on_input", False)
              and not getattr(c, "no_combine", False) and getattr(c, "routed_scaling_factor", None) in (None, 1.0)
              and getattr(c, "gemm1_alpha", None) is None and getattr(c, "gemm1_clamp_limit", None) is None and getattr(c, "swiglu_limit", None) is None
              and (getattr(c, "num_experts", None) is None or c.num_experts == c.num_local_experts))
        if not ok:
            n["stock"] += 1
            return f(hidden_states, w1, w2, topk_output, moe_runner_config, b1, b2, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                     per_channel_quant, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape, a1_q, fuse_swiglu_interleaved)
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        if a1_q is not None:
            xq, xs = a1_q, a1_scale
        else:
            xq, xs = quant(hidden_states, 128)
        c2 = B["c2"][: M * 8]
        rc = lib.s5_moe_up(var, grid, st, xq.data_ptr(), xs.data_ptr(), xs.stride(0), xs.stride(1), w1.data_ptr(), w1_scale.data_ptr(),
                           ids.data_ptr(), int(ids.dtype == torch.int64), c2.data_ptr(), M)
        assert rc == 0, rc
        aq, as_ = quant(c2, 128)
        out = hidden_states if getattr(c, "inplace", False) else torch.empty_like(hidden_states)
        rc = lib.s5_moe_down(var, grid, st, aq.data_ptr(), as_.data_ptr(), as_.stride(0), as_.stride(1), w2.data_ptr(), w2_scale.data_ptr(),
                             ids.data_ptr(), int(ids.dtype == torch.int64), tw.data_ptr(), B["c3"].data_ptr(), out.data_ptr(), B["cnt"].data_ptr(), M)
        assert rc == 0, rc
        if not n["used"]:
            print(f"q37 s5 moe used M={M} variant {var} quant {quant.__name__} ids {ids.dtype} xs stride {tuple(xs.stride())}", flush=True)
        n["used"] += 1
        return out
    Fm.fused_experts = g
    print("q37 s5 moe installed, variant", var & 7, "nb", 1 << nbl, "grid", grid, flush=True)
    return n, B


def install_shared_seq():
    """Q37_SHARED_SEQ=1: run each MoE block's shared expert and routed experts in sequence on one stream (the model's own non-dual-stream
    path) instead of overlapping them on the alt stream. Same kernels and arithmetic; removes SM and DRAM contention with the routed GEMMs."""
    Qm = importlib.import_module("sglang.srt.models.qwen2_moe")
    R = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    load = R.load_model

    @functools.wraps(load)
    def load2(self, *a, **k):
        out = load(self, *a, **k)
        n = 0
        for m in self.model.modules():
            if isinstance(m, Qm.Qwen2MoeSparseMoeBlock) and getattr(m, "alt_stream", None) is not None:
                m.alt_stream = None; n += 1
        print(f"q37 shared expert in sequence: {n} MoE blocks", flush=True)
        return out
    R.load_model = load2


def install_s7_moe():
    """Q37_S7_MOE=1 [Q37_S7_ST=2|3, Q37_S7_GRID=376]: routed experts at decode verify through the Session 8 kernels (s5_kernels/s7_moe.cu):
    stock quant -> up (warp per unit, SiLU-mul in registers) -> stock quant -> down (stream-K over row block x expert). Same eligibility
    as install_s5_moe; anything else runs the stock path. S track (summation by expert, not by slot)."""
    import ctypes, inspect
    st_ = int(os.environ.get("Q37_S7_ST", "2"))
    grid = int(os.environ.get("Q37_S7_GRID", "376"))
    Fm = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")
    Km = importlib.import_module("sglang.kernels.ops.moe.fused_moe_triton_kernels")
    R = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    src = inspect.getsource(Km.invoke_fused_moe_kernel)
    quant = next((getattr(Km, nm) for nm in ("sglang_per_token_group_quant_fp8", "per_token_group_quant_fp8") if nm in src and hasattr(Km, nm)), None)
    if quant is None:
        print("q37 s7 moe UNAVAILABLE: activation quant function not found, stock kept", flush=True)
        return
    lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_kernels", "s7_moe.so"))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.s7_moe_up.argtypes = [ci, ci, vp, vp, vp, ci, ci, vp, vp, vp, ci, vp, ci]
    lib.s7_moe_down.argtypes = [ci, ci, vp, vp, vp, ci, ci, vp, vp, vp, ci, vp, vp, vp, vp, ci]
    f = Fm.fused_experts
    B = {}
    n = dict(used=0, stock=0)
    load = R.load_model

    @functools.wraps(load)
    def load2(self, *a, **k):
        out = load(self, *a, **k)
        if "c2" not in B and any(p.dtype == torch.float8_e4m3fn for p in self.model.parameters()):
            d = next(self.model.parameters()).device
            B.update(c2=torch.empty((64, 512), dtype=torch.bfloat16, device=d), part=torch.zeros((grid * 4 * 2 * 128,), dtype=torch.float32, device=d),
                     cnt=torch.zeros((128,), dtype=torch.int32, device=d))
            print("q37 s7 moe buffers ready", flush=True)
        return out
    R.load_model = load2

    @functools.wraps(f)
    def g(hidden_states, w1, w2, topk_output, moe_runner_config, b1=None, b2=None, use_fp8_w8a8=False, use_int8_w8a8=False,
          use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, w1_scale=None, w2_scale=None, w1_zp=None, w2_zp=None,
          a1_scale=None, a2_scale=None, block_shape=None, a1_q=None, fuse_swiglu_interleaved=False):
        c = moe_runner_config
        tw, ids = topk_output[0], topk_output[1]
        M = hidden_states.shape[0]
        ok = ("c2" in B and 1 <= M <= 8 and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous() and use_fp8_w8a8
              and not (use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16 or per_channel_quant or fuse_swiglu_interleaved)
              and list(block_shape or []) == [128, 128] and b1 is None and b2 is None and a1_scale is None and a2_scale is None
              and tuple(w1.shape) == (256, 1024, 2048) and tuple(w2.shape) == (256, 2048, 512)
              and w1_scale is not None and tuple(w1_scale.shape) == (256, 8, 16) and w2_scale is not None and tuple(w2_scale.shape) == (256, 16, 4)
              and w1_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32 and w1.is_contiguous() and w2.is_contiguous()
              and tuple(ids.shape) == (M, 8) and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
              and tw.dtype == torch.float32 and tw.is_contiguous()
              and getattr(c, "activation", "silu") == "silu" and getattr(c, "is_gated", True) and not getattr(c, "apply_router_weight_on_input", False)
              and not getattr(c, "no_combine", False) and getattr(c, "routed_scaling_factor", None) in (None, 1.0)
              and getattr(c, "gemm1_alpha", None) is None and getattr(c, "gemm1_clamp_limit", None) is None and getattr(c, "swiglu_limit", None) is None
              and (getattr(c, "num_experts", None) is None or c.num_experts == c.num_local_experts))
        if not ok:
            n["stock"] += 1
            return f(hidden_states, w1, w2, topk_output, moe_runner_config, b1, b2, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                     per_channel_quant, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape, a1_q, fuse_swiglu_interleaved)
        stm = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        xq, xs = (a1_q, a1_scale) if a1_q is not None else quant(hidden_states, 128)
        c2 = B["c2"][: M * 8]
        i64 = int(ids.dtype == torch.int64)
        rc = lib.s7_moe_up(st_, grid, stm, xq.data_ptr(), xs.data_ptr(), xs.stride(0), xs.stride(1), w1.data_ptr(), w1_scale.data_ptr(), ids.data_ptr(), i64, c2.data_ptr(), M)
        assert rc == 0, rc
        aq, as_ = quant(c2, 128)
        out = hidden_states if getattr(c, "inplace", False) else torch.empty_like(hidden_states)
        rc = lib.s7_moe_down(st_, grid, stm, aq.data_ptr(), as_.data_ptr(), as_.stride(0), as_.stride(1), w2.data_ptr(), w2_scale.data_ptr(),
                             ids.data_ptr(), i64, tw.data_ptr(), B["part"].data_ptr(), out.data_ptr(), B["cnt"].data_ptr(), M)
        assert rc == 0, rc
        if not n["used"]:
            print(f"q37 s7 moe used M={M} st {st_} grid {grid} quant {quant.__name__} ids {ids.dtype}", flush=True)
        n["used"] += 1
        return out
    Fm.fused_experts = g
    print("q37 s7 moe installed, st", st_, "grid", grid, flush=True)


def install_commit_overlap():
    """Q37_COMMIT_OVERLAP=1: after accept, the GDN/Bole state commit (_update_target_mamba_state_after_verify) runs on a side stream while
    the drafter KV materialization of the committed tokens (_append_target_hidden_to_draft_kv_by_loc) runs on the current stream. They read
    the same accept results and write different pools (mamba state vs draft KV). The side stream first waits for all queued work and is
    joined right after the KV append, before the round returns, so the next verify sees the committed state. Same kernels, same values."""
    W = importlib.import_module("sglang.srt.speculative.dflash_worker_v2")
    C = W.DFlashWorkerV2
    up, app = C._update_target_mamba_state_after_verify, C._append_target_hidden_to_draft_kv_by_loc
    st = {}
    n = dict(c=0)

    @functools.wraps(up)
    def up2(self, *a, **k):
        if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
            return up(self, *a, **k)
        side = st.get("s")
        if side is None:
            side = st["s"] = torch.cuda.Stream()
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            r = up(self, *a, **k)
        st["pending"] = (cur, side)
        if not n["c"]:
            print("q37 commit overlap used", flush=True)
        n["c"] += 1
        return r

    @functools.wraps(app)
    def app2(self, *a, **k):
        try:
            return app(self, *a, **k)
        finally:
            pend = st.pop("pending", None)
            if pend is not None:
                pend[0].wait_stream(pend[1])
    fbg = C.forward_batch_generation

    @functools.wraps(fbg)
    def fbg2(self, *a, **k):
        try:
            return fbg(self, *a, **k)
        finally:   # safety: never leave the commit unjoined past the round
            pend = st.pop("pending", None)
            if pend is not None:
                pend[0].wait_stream(pend[1])
    C._update_target_mamba_state_after_verify, C._append_target_hidden_to_draft_kv_by_loc, C.forward_batch_generation = up2, app2, fbg2
    print("q37 commit overlap installed", flush=True)


def install_feat_dump(path):
    """Q37_FEAT_DUMP=<dir> [Q37_FEAT_W=2048]: drafter fine-tuning data. At the drafter's KV-append point (the only place target features
    reach the drafter), keep per request: the last W prompt tokens' features (prefill) and every verify row's features with its position
    and the round's commit length (decode). Features are the drafter's own input: its frozen fc + hidden_norm applied to the raw
    concatenated target layers. When a request finishes, one file <rid>.pt: ctx positions/features, verify positions/features/commit
    lengths, prompt length, output ids. Debug/data only: adds one small projection per round."""
    W = importlib.import_module("sglang.srt.speculative.dflash_worker_v2")
    S = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    C = W.DFlashWorkerV2
    win = int(os.environ.get("Q37_FEAT_W", "2048"))
    os.makedirs(path, exist_ok=True)
    fbg, app, pbr, acc = C.forward_batch_generation, C._append_target_hidden_to_draft_kv_by_loc, S.process_batch_result, C._accept_block
    st = {"cur": None}
    buf = {}
    n = dict(saved=0)

    @functools.wraps(fbg)
    def fbg2(self, batch, *a, **k):
        reqs = list(getattr(batch, "reqs", None) or [])
        st["cur"] = ([q.rid for q in reqs], list(getattr(batch, "extend_lens", None) or []))
        try:
            return fbg(self, batch, *a, **k)
        finally:
            st["cur"] = None

    @functools.wraps(acc)
    def acc2(self, *a, **k):   # soft labels: the target's top-16 next-token log-probs for every verify row (fp16 values, int32 ids)
        r = acc(self, *a, **k)
        cur = st["cur"]
        lg = k.get("next_token_logits")
        if cur is not None and lg is not None:
            bs = int(k.get("bs", len(cur[0])))
            with torch.no_grad():
                lp = torch.log_softmax(lg.float(), -1)
                v, ix = lp.topk(16, dim=-1)
                blk = v.shape[0] // max(bs, 1)
                v = v.view(bs, blk, 16).to(torch.float16).clone(); ix = ix.view(bs, blk, 16).to(torch.int32).clone()
                for i, rid in enumerate(cur[0][:bs]):
                    b = buf.setdefault(rid, {"cp": [], "cf": [], "vp": [], "vf": [], "vc": [], "tv": [], "ti": []})
                    b.setdefault("tv", []).append(v[i]); b.setdefault("ti", []).append(ix[i])
        return r

    @functools.wraps(app)
    def app2(self, *, target_hidden, cache_loc, positions, cache_loc_2d=None, commit_lens=None, extend_lens=None):
        r = app(self, target_hidden=target_hidden, cache_loc=cache_loc, positions=positions, cache_loc_2d=cache_loc_2d, commit_lens=commit_lens, extend_lens=extend_lens)
        cur = st["cur"]
        if cur is None or target_hidden is None or target_hidden.numel() == 0:
            return r
        rids, ext = cur
        with torch.no_grad():
            if extend_lens is not None:   # prefill chunk: rows of each request in batch order; keep the last `win`
                off = 0
                for rid, L in zip(rids, ext):
                    L = int(L); keep = min(win, L)
                    if keep > 0:
                        rows = target_hidden[off + L - keep: off + L]
                        f = self.draft_model.project_target_hidden(rows).to(torch.bfloat16)
                        b = buf.setdefault(rid, {"cp": [], "cf": [], "vp": [], "vf": [], "vc": []})
                        b["cp"].append(positions[off + L - keep: off + L].to(torch.int64).clone()); b["cf"].append(f)
                        cpos = torch.cat(b["cp"]); cf = torch.cat(b["cf"])
                        b["cp"], b["cf"] = [cpos[-win:]], [cf[-win:]]
                    off += L
            elif cache_loc_2d is not None and commit_lens is not None:   # verify: all rows; commit lengths filter them at save time
                bs, blk = cache_loc_2d.shape[0], cache_loc_2d.shape[1]
                f = self.draft_model.project_target_hidden(target_hidden[: bs * blk]).to(torch.bfloat16).view(bs, blk, -1)
                pos = positions[: bs * blk].view(bs, blk).to(torch.int64)
                for i, rid in enumerate(rids[:bs]):
                    b = buf.setdefault(rid, {"cp": [], "cf": [], "vp": [], "vf": [], "vc": []})
                    # clone: positions / commit_lens are persistent worker buffers that later rounds overwrite
                    b["vp"].append(pos[i].clone()); b["vf"].append(f[i]); b["vc"].append(commit_lens[i].reshape(1).to(torch.int32).clone())
        return r

    @functools.wraps(pbr)
    def pbr2(self, batch, result, *a, **k):
        out = pbr(self, batch, result, *a, **k)
        for q in (getattr(batch, "reqs", None) or ()):
            if q.finished() and q.rid in buf:
                b = buf.pop(q.rid)
                rec = {"rid": q.rid, "prompt_len": len(q.origin_input_ids), "output_ids": list(q.output_ids),
                       "ctx_ids": None, "ctx_pos": torch.cat(b["cp"]).cpu() if b["cp"] else None,
                       "ctx_feat": (torch.cat(b["cf"]).to(torch.float8_e4m3fn if os.environ.get("Q37_FEAT_FP8") else torch.bfloat16).cpu()) if b["cf"] else None,
                       "ver_pos": torch.stack(b["vp"]).cpu() if b["vp"] else None, "ver_feat": torch.stack(b["vf"]).cpu() if b["vf"] else None,
                       "ver_commit": torch.cat(b["vc"]).cpu() if b["vc"] else None,
                       "top_lp": torch.stack(b["tv"]).cpu() if b.get("tv") else None, "top_id": torch.stack(b["ti"]).cpu() if b.get("ti") else None}
                if rec["ctx_pos"] is not None:
                    ids = q.origin_input_ids
                    rec["ctx_ids"] = torch.tensor([ids[int(p_)] if int(p_) < len(ids) else -1 for p_ in rec["ctx_pos"]], dtype=torch.int32)
                torch.save(rec, os.path.join(path, f"{q.rid}.pt"))
                n["saved"] += 1
                if n["saved"] % 100 == 1:
                    print(f"q37 feat dump saved {n['saved']} (last {q.rid}: ctx {0 if rec['ctx_pos'] is None else len(rec['ctx_pos'])}, rounds {0 if rec['ver_pos'] is None else len(rec['ver_pos'])}, out {len(rec['output_ids'])})", flush=True)
        return out
    C.forward_batch_generation, C._append_target_hidden_to_draft_kv_by_loc, S.process_batch_result, C._accept_block = fbg2, app2, pbr2, acc2
    print("q37 feat dump ->", path, "window", win, flush=True)


def install_from_env():
    E = os.environ
    if E.get("Q37_MONO"):
        install_mono()
    if E.get("Q37_DUMP_HIDDEN"):
        install_dump_hidden(E["Q37_DUMP_HIDDEN"])
    if E.get("Q37_KV_FUSE"):
        install_kv_fuse()
    if E.get("Q37_CONV_TILE"):
        install_conv_tile()
    if E.get("Q37_BA_EXACT"):
        install_ba_exact()
    if E.get("Q37_FEAT_DUMP"):
        install_feat_dump(E["Q37_FEAT_DUMP"])
    if E.get("Q37_COMMIT_OVERLAP"):
        install_commit_overlap()
    if E.get("Q37_SHARED_SEQ"):
        install_shared_seq()
    if E.get("Q37_S7_MOE"):
        install_s7_moe()
    if E.get("Q37_S5_MOE"):
        install_s5_moe()
    if E.get("Q37_S5_BA"):
        install_s5_ba()
    if E.get("Q37_S5_DENSE"):
        install_s5_dense()
    if E.get("Q37_BA_TINY"):
        install_ba_tiny()
    if E.get("Q37_BOLE_WARPS"):
        install_bole_warps(E["Q37_BOLE_WARPS"])
    if E.get("Q37_TRACK_GUARD"):
        install_track_guard()
    if E.get("Q37_KV_PAR"):
        install_kv_par(int(E["Q37_KV_PAR"]))
    if E.get("Q37_PF"):
        install_prefetch(E["Q37_PF"].split(","))
    if E.get("Q37_STATE_CHECK"):
        install_state_check(E["Q37_STATE_CHECK"])
    if E.get("Q37_S4_GEMM"):
        install_s4_gemm()
    if E.get("Q37_GEMM_STREAMK"):
        install_gemm_streamk()
    if E.get("Q37_HIDDEN_RING"):
        install_hidden_ring(E["Q37_HIDDEN_RING"])
    if E.get("Q37_PLD"):
        install_pld()
    if E.get("Q37_LATTICE_DUMP"):
        install_lattice_dump(E["Q37_LATTICE_DUMP"])
    if E.get("Q37_ACCEPT_LOG"):
        install_accept_log(E["Q37_ACCEPT_LOG"])
    if E.get("Q37_NVTX"):
        install()
    if E.get("Q37_ROUTE_DUMP"):
        install_route_dump(E["Q37_ROUTE_DUMP"])
    if E.get("Q37_AVL"):
        install_avl()
    if E.get("Q37_SYNCFREE"):
        install_syncfree()
    if E.get("Q37_ATTN128"):
        install_attn128()
    if E.get("Q37_ATTN_LSKIP"):
        install_attn_lskip()
    if E.get("Q37_OP_TRACE"):
        install_op_trace(E["Q37_OP_TRACE"])
    if E.get("Q37_HOTVOCAB"):
        install_hot_vocab(E["Q37_HOTVOCAB"])
