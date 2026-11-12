"""Session 5 isolated gate for the Track E dense experiment (no server). Run on the instance after the s5 kernels are built.
1. Bitwise: stock SM120 swapAB FP8 GEMM vs our narrow-tile variants (64, 32 rows) for M = 1..16 on the decode shapes, random FP8 data and
   real checkpoint weights where a weight of that shape exists. Scales: activation per token per 128 K (column-major, as the CUTLASS path
   gets them), weight per 128x128 block, repeated per sub-block for the variants.
2. Bitwise: in_proj_ba (64 x 2048 bf16) cuBLAS (F.linear) vs s4_ba vs s5_ba, M = 1..16.
3. Timing, L2 evicted before every call (256 MB write, its own time subtracted): each GEMM alone at M=8, and the qkvz + BA pair the way the
   model runs it (qkvz on the current stream, BA on a side stream, joined), stock pair vs new pair.
Prints one line per check; ends with S5_DENSE_PASS or S5_DENSE_FAIL. Usage: python s5_dense_exact.py [64,32]"""
import ctypes, importlib, json, os, sys
import torch
import torch.nn.functional as F

Q = os.path.dirname(os.path.abspath(__file__))
TILES = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "64").split(",")]
MMAX = int(os.environ.get("S5_MMAX", "16"))   # dense bitwise check rows (the narrow kernel takes M <= 32)
dev = "cuda"
G = importlib.import_module("sglang.kernels.ops.gemm.fp8_blockwise_gemm")
U = importlib.import_module("sglang.kernels.jit.utils")
stock = G._jit_fp8_blockwise_module()
mods = {}
for T in TILES:
    name = f"fp8_blockwise_scaled_mm_n{T}"
    try:
        mods[T] = getattr(U.load_jit(f"fp8_blockwise_scaled_mm_s5_{T}", cuda_files=["gemm/fp8_blockwise_s5/fp8_blockwise_scaled_mm_entry.cuh"],
                                     cuda_wrappers=[(name, name)], extra_dependencies=["cutlass"],
                                     extra_cuda_cflags=list(G._fp8_blockwise_cuda_flags()) + (["-DS5_TILE32"] if T == 32 else [])), name)
        print(f"BUILD tile {T} ok", flush=True)
    except Exception as e:
        print(f"BUILD tile {T} FAILED: {str(e)[-400:]}", flush=True)
ok = True

# Runtime decode shapes (N, K) from the best-stack trace grids: out_proj / o_proj (1,16,1), qkvz (1,96,1), attention qkv (1,72,1).
SHAPES = {"out_o_proj": (2048, 4096), "qkvz": (12288, 2048), "attn_qkv": (9216, 2048)}
M0 = os.path.expanduser("~/lab/models/Qwen3.6-35B-A3B-FP8")
real = {}
try:
    from safetensors import safe_open
    idx = json.load(open(M0 + "/model.safetensors.index.json"))["weight_map"]
    want = {v: k for k, v in SHAPES.items()}
    for key, fn in idx.items():
        if not key.endswith(".weight_scale_inv") or ".layers.0." not in key and ".layers.3." not in key:
            continue
        wk = key.replace("_scale_inv", "")
        with safe_open(M0 + "/" + idx[wk], "pt") as f:
            w = f.get_slice(wk); shp = tuple(w.get_shape())
            if shp in want and want[shp] not in real:
                real[want[shp]] = (f.get_tensor(wk).to(dev), safe_open(M0 + "/" + fn, "pt").get_tensor(key).to(dev).float(), wk)
    print("REAL weights found:", {k: v[2] for k, v in real.items()}, flush=True)
except Exception as e:
    print("REAL weights unavailable:", str(e)[:200], flush=True)


def act(M, K, g):
    a = (torch.randn(M, K, generator=g, device=dev) * 2).to(torch.float8_e4m3fn)
    sa = (torch.rand(K // 128, M, generator=g, device=dev) * 0.02 + 1e-3).t()  # column-major [M, K/128]
    return a, sa


def run_stock(a, w, sa, sw, out=None):
    out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device=dev) if out is None else out
    stock.fp8_blockwise_scaled_mm(out, a, w.t(), sa, sw.t())
    return out


def run_new(T, a, w, sa, sw_rep_t, out=None):
    out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device=dev) if out is None else out
    mods[T](out, a, w.t(), sa, sw_rep_t)
    return out


g = torch.Generator(device=dev).manual_seed(0)
for tag, (N, K) in SHAPES.items():
    cases = [("random", (torch.randn(N, K, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn), torch.rand(N // 128, K // 128, generator=g, device=dev) * 0.01 + 1e-4)]
    if tag in real:
        cases.append(("real", real[tag][0], real[tag][1]))
    for src, w, sw in cases:
        for T in mods:
            rep = sw.repeat_interleave(128 // T, dim=0).contiguous().t()
            bad = []
            for M in range(1, MMAX + 1):
                a, sa = act(M, K, g)
                if not torch.equal(run_stock(a, w, sa, sw), run_new(T, a, w, sa, rep)):
                    bad.append(M)
            ok &= not bad
            print(f"BITWISE {tag} {src} N={N} K={K} tile {T}: {f'equal for M=1..{MMAX}' if not bad else 'DIFFERS at M=' + str(bad)}", flush=True)

# in_proj_ba
s4 = ctypes.CDLL(os.path.join(Q, "s4_kernels", "s4_wmma.so")); s4.s4_ba.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int]
s5 = ctypes.CDLL(os.path.join(Q, "s5_kernels", "s5_ba.so")); s5.s5_ba.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
wba = (torch.randn(64, 2048, generator=g, device=dev) * 0.05).to(torch.bfloat16)


def ba4(x):
    o = x.new_empty(x.shape[0], 64); assert s4.s4_ba(o.data_ptr(), x.data_ptr(), wba.data_ptr(), x.shape[0], 2048, 64, 128, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), 1024) == 0; return o


def ba5(x):
    o = x.new_empty(x.shape[0], 64); assert s5.s5_ba(o.data_ptr(), x.data_ptr(), wba.data_ptr(), x.shape[0], 2048, 64, 128, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)) == 0; return o


bad = []
for M in range(1, 17):
    for rep_ in range(3):
        x = torch.randn(M, 2048, generator=g, device=dev).to(torch.bfloat16)
        r = F.linear(x, wba); b4 = ba4(x); b5 = ba5(x)
        if not torch.equal(b4, b5):
            bad.append((M, torch.equal(r, b4), torch.equal(r, b5), torch.equal(b4, b5)))
ok &= not bad
print(f"BITWISE in_proj_ba s5_ba vs production s4_ba: {'equal for M=1..16' if not bad else 'DIFFERS ' + str(bad[:6])}", flush=True)

# Timing, L2 evicted before each call.
ev = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
side = torch.cuda.Stream()


def timed(fn, n=50):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5): ev.fill_(1); fn()
    torch.cuda.synchronize(); tot = 0.0
    for _ in range(n):
        ev.fill_(1); s.record(); fn(); e.record(); torch.cuda.synchronize(); tot += s.elapsed_time(e)
    return tot / n * 1e3


M = int(os.environ.get("S5_TM", "8"))   # timing rows (block 24: 24)
for tag, (N, K) in SHAPES.items():
    w = (torch.randn(N, K, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn); sw = torch.rand(N // 128, K // 128, generator=g, device=dev) * 0.01 + 1e-4
    a, sa = act(M, K, g); o = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    line = f"TIME {tag} M={M} cold: stock {timed(lambda: run_stock(a, w, sa, sw, o)):.1f} us"
    for T in mods:
        rep = sw.repeat_interleave(128 // T, dim=0).contiguous().t()
        line += f", tile{T} {timed(lambda: run_new(T, a, w, sa, rep, o)):.1f} us"
    print(line, flush=True)

x = torch.randn(M, 2048, generator=g, device=dev).to(torch.bfloat16)
if M <= 16: print(f"TIME in_proj_ba M={M} cold: cuBLAS {timed(lambda: F.linear(x, wba)):.1f} us, s4_ba {timed(lambda: ba4(x)):.1f} us, s5_ba {timed(lambda: ba5(x)):.1f} us", flush=True)
N, K = SHAPES["qkvz"]
w = (torch.randn(N, K, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn); sw = torch.rand(N // 128, K // 128, generator=g, device=dev) * 0.01 + 1e-4
a, sa = act(M, K, g); o = torch.empty(M, N, dtype=torch.bfloat16, device=dev)


def pair(gemm, ba):
    def f():
        main = torch.cuda.current_stream(); side.wait_stream(main)
        with torch.cuda.stream(side):
            ba(x)
        gemm(); main.wait_stream(side)
    return f


if M <= 16:   # the BA kernels hold 16 rows
    line = f"TIME qkvz+ba pair M={M} cold: stock+s4 {timed(pair(lambda: run_stock(a, w, sa, sw, o), ba4)):.1f} us"
    for T in mods:
        rep = sw.repeat_interleave(128 // T, dim=0).contiguous().t()
        line += f", tile{T}+s5 {timed(pair(lambda: run_new(T, a, w, sa, rep, o), ba5)):.1f} us"
    print(line, flush=True)
print("S5_DENSE_PASS" if ok and mods else "S5_DENSE_FAIL", flush=True)
