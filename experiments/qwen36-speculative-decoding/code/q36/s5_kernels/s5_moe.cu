// Session 6, our routed-MoE kernels for decode verify (M <= 8 tokens, top-8 of 256 experts, FP8 block-128 weights), SM120.
// Idea credit: weight-major / SwapAB scheduling from MonoMoE (arXiv 2609.04244, FlashInfer, Apache-2.0); this is an independent
// mma.sync implementation, not a port. Replaces, per layer: moe_align + Triton up GEMM + silu_and_mul + Triton down GEMM + moe_sum
// with two persistent kernels. The two activation quantizations stay the stock kernels (called between the launches).
//
//   K1 s5_moe_up:   cache2[p, :] = bf16(silu(bf16(gate_p)) * bf16(up_p)),  gate/up = xq[t(p)] . w13[e(p)]  (block-128 promoted)
//   K2 s5_moe_down: cache3[p, :] = bf16((a2q[p] . w2[e(p)]) * topk_w[p]);  out[t, r] = bf16(sum_slot cache3[t*8+slot, r])
//
// Weights sit on the MMA M axis (16 rows), tokens on N (8 = the verify width), mma.sync.m16n8k32 e4m3 -> f32.
// Per 128-K block: 4 chained MMAs into a zeroed temporary, then acc += tmp * a_scale * w_scale, the Triton kernel's order.
// Each CTA rebuilds the (expert -> pairs) table from top-k ids (no align kernel) and streams a balanced, statically assigned list of
// weight tiles through a cp.async ring that runs ahead across tile and expert boundaries. No cross-CTA waits: the down sum is done by
// whichever warp finishes a 16-row block last (atomic counter, reset by that warp), so nothing depends on CTA residency.
// Rounding details that could not be read from the installed source are runtime variants, chosen by a bitwise check against the
// stock path (s5_moe_exact.py): bit 0 FMA in the promotion, bit 1 expf (else __expf), bit 2 SiLU rounded to bf16 before the multiply.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
using bf = __nv_bfloat16;

constexpr int K_HID = 2048, N_INT = 512, E_ALL = 256, TOPK = 8, MAXP = 64;
constexpr int ROWPAD = 144;      // 128-byte K chunk + 16 B pad: rows land 4 banks apart
constexpr int XPAD = K_HID + 16; // token rows of the quantized activation in shared memory
constexpr int APAD = N_INT + 16;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void wait_n() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ uint32_t ld32s(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool FMA> __device__ __forceinline__ float promote(float acc, float t, float sa, float sw) {
  if (FMA) return fmaf(__fmul_rn(t, sa), sw, acc);
  return __fadd_rn(acc, __fmul_rn(__fmul_rn(t, sa), sw));
}

// (expert -> pair list) in first-appearance order of the route-major pairs p = t*8 + slot; identical in every CTA.
struct Route { int ne; int ex[MAXP]; int cnt[MAXP]; int pr[MAXP][TOPK]; };
__device__ void build_route(Route& R, int* map, const void* ids, int ids64, int P) {
  for (int i = threadIdx.x; i < E_ALL; i += blockDim.x) map[i] = -1;
  __syncthreads();
  if (threadIdx.x == 0) {
    int ne = 0;
    for (int p = 0; p < P; ++p) {
      int e = ids64 ? (int)reinterpret_cast<const int64_t*>(ids)[p] : reinterpret_cast<const int*>(ids)[p];
      if (e < 0 || e >= E_ALL) continue;
      int s = map[e];
      if (s < 0) { s = ne++; map[e] = s; R.ex[s] = e; R.cnt[s] = 0; }
      if (R.cnt[s] < TOPK) R.pr[s][R.cnt[s]++] = p;
    }
    R.ne = ne;
  }
  __syncthreads();
}

// ---------------------------------------------------------------- K1: gate/up + SiLU-mul
// Session 7: NB = 128-K scale blocks moved per pipeline step (1, 2, 4). Larger steps amortize the per-step wait + barrier; the
// arithmetic (per-block promotion, ascending order) is identical for every NB.
template <int NB> struct Up { static constexpr int RP = NB * 128 + 16, ST = NB == 1 ? 8 : (NB == 2 ? 4 : 3), SPU = 16 / NB; };
template <int VAR, int NB>
__global__ void __launch_bounds__(128, 1) s5_moe_up_kernel(
    const uint8_t* __restrict__ xq, const float* __restrict__ xs, int xs_rs, int xs_cs,
    const uint8_t* __restrict__ w13, const float* __restrict__ s13, const void* ids, int ids64, bf* __restrict__ cache2, int M) {
  extern __shared__ __align__(16) uint8_t sm[];
  Route& R = *reinterpret_cast<Route*>(sm);
  int* map = reinterpret_cast<int*>(sm + sizeof(Route));
  uint8_t* sx = sm + ((sizeof(Route) + E_ALL * 4 + 15) / 16) * 16;          // [8][XPAD]
  float* sxs = reinterpret_cast<float*>(sx + 8 * XPAD);                       // [8][16]
  constexpr int RP = Up<NB>::RP, ST1 = Up<NB>::ST, SPU = Up<NB>::SPU;
  uint8_t* ring = reinterpret_cast<uint8_t*>(sxs + 8 * 16);                   // [ST1][32][RP]
  float* ex = reinterpret_cast<float*>(ring + ST1 * 32 * RP);                 // [16][8] gate values for the epilogue
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, g = lane >> 2, q = lane & 3;
  const int P = M * TOPK;
  build_route(R, map, ids, ids64, P);
  for (int i = t; i < M * (K_HID / 16); i += blockDim.x) {
    const int r = i / (K_HID / 16), c = (i % (K_HID / 16)) * 16;
    cp16(sx + r * XPAD + c, xq + (size_t)r * K_HID + c);
  }
  for (int i = t; i < M * 16; i += blockDim.x) sxs[(i / 16) * 16 + i % 16] = xs[(i / 16) * xs_rs + (i % 16) * xs_cs];
  commit();
  const int units = R.ne * (N_INT / 16);
  const int mine = units > (int)blockIdx.x ? (units - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  const int steps = mine * SPU;
  auto load = [&](int s) {
    if (s < steps) {
      const int u = blockIdx.x + (s / SPU) * gridDim.x, kg = s % SPU, es = u >> 5, j = u & 31, e = R.ex[es];
      uint8_t* dst = ring + (s % ST1) * 32 * RP;
      for (int i = t; i < 32 * 8 * NB; i += blockDim.x) {
        const int r = i / (8 * NB), c = (i % (8 * NB)) * 16;
        const int row = (r < 16) ? j * 16 + r : N_INT + j * 16 + (r - 16);
        cp16(dst + r * RP + c, w13 + ((size_t)e * (2 * N_INT) + row) * K_HID + kg * NB * 128 + c);
      }
    }
    commit();
  };
  for (int s = 0; s < ST1 - 1; ++s) load(s);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < steps; ++s) {
    wait_n<ST1 - 2>();
    __syncthreads();
    load(s + ST1 - 1);
    const int u = blockIdx.x + (s / SPU) * gridDim.x, kg = s % SPU, es = u >> 5, j = u & 31, e = R.ex[es], nt = R.cnt[es];
    if (warp < 2) {
      const uint8_t* A0 = ring + (s % ST1) * 32 * RP + warp * 16 * RP;
      const int tg = g < nt ? R.pr[es][g] / TOPK : 0;
      const int rb = (warp == 0 ? 0 : N_INT / 128) + (j * 16) / 128;
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const int kc = kg * NB + nb;
        const uint8_t* A = A0 + nb * 128;
        const uint8_t* B = sx + tg * XPAD + kc * 128;
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int k0 = kk * 32;
          const uint32_t a0 = ld32s(A + g * RP + k0 + q * 4), a1 = ld32s(A + (g + 8) * RP + k0 + q * 4);
          const uint32_t a2 = ld32s(A + g * RP + k0 + 16 + q * 4), a3 = ld32s(A + (g + 8) * RP + k0 + 16 + q * 4);
          const uint32_t b0 = g < nt ? ld32s(B + k0 + q * 4) : 0u, b1 = g < nt ? ld32s(B + k0 + 16 + q * 4) : 0u;
          mma_e4m3(tmp, a0, a1, a2, a3, b0, b1);
        }
        const float sw = s13[((size_t)e * (2 * N_INT / 128) + rb) * (K_HID / 128) + kc];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int n = q * 2 + (i & 1);
          const float sa = n < nt ? sxs[(R.pr[es][n] / TOPK) * 16 + kc] : 0.f;
          acc[i] = promote<(VAR & 1) != 0>(acc[i], tmp[i], sa, sw);
        }
      }
    }
    if (kg == SPU - 1) {   // epilogue of unit u: rows j*16..+15 of the intermediate for the expert's tokens
      if (warp == 0) for (int i = 0; i < 4; ++i) ex[(g + 8 * (i >> 1)) * 8 + q * 2 + (i & 1)] = acc[i];
      __syncthreads();
      if (warp == 1) {
        for (int i = 0; i < 4; ++i) {
          const int n = q * 2 + (i & 1), r = g + 8 * (i >> 1);
          if (n < nt) {
            const float gv = __bfloat162float(__float2bfloat16(ex[r * 8 + n]));
            const float uv = __bfloat162float(__float2bfloat16(acc[i]));
            float sl = gv / (1.0f + ((VAR & 2) ? expf(-gv) : __expf(-gv)));
            if (VAR & 4) sl = __bfloat162float(__float2bfloat16(sl));
            cache2[(size_t)R.pr[es][n] * N_INT + j * 16 + r] = __float2bfloat16(__fmul_rn(sl, uv));
          }
        }
      }
      for (int i = 0; i < 4; ++i) acc[i] = 0.f;
    }
  }
  wait_n<0>();
}

// ---------------------------------------------------------------- K2: down + routed weight + fixed-order sum
template <int NB> struct Dn { static constexpr int RP = NB * 128 + 16, ST = NB == 1 ? 12 : (NB == 2 ? 6 : 4), SPU = 4 / NB; };
template <int VAR, int NB>
__global__ void __launch_bounds__(128, 1) s5_moe_down_kernel(
    const uint8_t* __restrict__ aq, const float* __restrict__ as, int as_rs, int as_cs,
    const uint8_t* __restrict__ w2, const float* __restrict__ s2, const void* ids, int ids64, const float* __restrict__ tw,
    bf* __restrict__ cache3, bf* __restrict__ out, int* __restrict__ cnt, int M) {
  extern __shared__ __align__(16) uint8_t sm[];
  Route& R = *reinterpret_cast<Route*>(sm);
  int* map = reinterpret_cast<int*>(sm + sizeof(Route));
  uint8_t* sa2 = sm + ((sizeof(Route) + E_ALL * 4 + 15) / 16) * 16;           // [64][APAD]
  float* sas = reinterpret_cast<float*>(sa2 + MAXP * APAD);                   // [64][4]
  float* stw = sas + MAXP * 4;                                                // [64]
  constexpr int RP = Dn<NB>::RP, ST2 = Dn<NB>::ST, SPU = Dn<NB>::SPU;
  uint8_t* ring = reinterpret_cast<uint8_t*>(stw + MAXP);                     // [ST2][16][RP]
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, g = lane >> 2, q = lane & 3;
  const int P = M * TOPK;
  build_route(R, map, ids, ids64, P);
  for (int i = t; i < P * (N_INT / 16); i += blockDim.x) {
    const int r = i / (N_INT / 16), c = (i % (N_INT / 16)) * 16;
    cp16(sa2 + r * APAD + c, aq + (size_t)r * N_INT + c);
  }
  for (int i = t; i < P * 4; i += blockDim.x) sas[i] = as[(i / 4) * as_rs + (i % 4) * as_cs];
  for (int i = t; i < P; i += blockDim.x) stw[i] = tw[i];
  commit();
  const int units = R.ne * (K_HID / 16);
  const int mine = units > (int)blockIdx.x ? (units - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  const int steps = mine * SPU;
  auto load = [&](int s) {
    if (s < steps) {
      const int u = blockIdx.x + (s / SPU) * gridDim.x, kg = s % SPU, es = u >> 7, rb = u & 127, e = R.ex[es];
      uint8_t* dst = ring + (s % ST2) * 16 * RP;
      for (int i = t; i < 16 * 8 * NB; i += blockDim.x) {
        const int r = i / (8 * NB), c = (i % (8 * NB)) * 16;
        cp16(dst + r * RP + c, w2 + ((size_t)e * K_HID + rb * 16 + r) * N_INT + kg * NB * 128 + c);
      }
    }
    commit();
  };
  for (int s = 0; s < ST2 - 1; ++s) load(s);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < steps; ++s) {
    wait_n<ST2 - 2>();
    __syncthreads();
    load(s + ST2 - 1);
    if (warp == 0) {
      const int u = blockIdx.x + (s / SPU) * gridDim.x, kg = s % SPU, es = u >> 7, rb = u & 127, e = R.ex[es], nt = R.cnt[es];
      const uint8_t* A0 = ring + (s % ST2) * 16 * RP;
#pragma unroll
      for (int nb = 0; nb < NB; ++nb) {
        const int kc = kg * NB + nb;
        const uint8_t* A = A0 + nb * 128;
        const uint8_t* B = sa2 + (g < nt ? R.pr[es][g] : 0) * APAD + kc * 128;
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int k0 = kk * 32;
          const uint32_t a0 = ld32s(A + g * RP + k0 + q * 4), a1 = ld32s(A + (g + 8) * RP + k0 + q * 4);
          const uint32_t a2 = ld32s(A + g * RP + k0 + 16 + q * 4), a3 = ld32s(A + (g + 8) * RP + k0 + 16 + q * 4);
          const uint32_t b0 = g < nt ? ld32s(B + k0 + q * 4) : 0u, b1 = g < nt ? ld32s(B + k0 + 16 + q * 4) : 0u;
          mma_e4m3(tmp, a0, a1, a2, a3, b0, b1);
        }
        const float sw = s2[((size_t)e * (K_HID / 128) + (rb * 16) / 128) * (N_INT / 128) + kc];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int n = q * 2 + (i & 1);
          const float sa = n < nt ? sas[R.pr[es][n] * 4 + kc] : 0.f;
          acc[i] = promote<(VAR & 1) != 0>(acc[i], tmp[i], sa, sw);
        }
      }
      if (kg == SPU - 1) {
        for (int i = 0; i < 4; ++i) {
          const int n = q * 2 + (i & 1), r = rb * 16 + g + 8 * (i >> 1);
          if (n < nt) {
            const int p = R.pr[es][n];
            cache3[(size_t)p * K_HID + r] = __float2bfloat16(__fmul_rn(acc[i], stw[p]));
          }
          acc[i] = 0.f;
        }
        __threadfence();
        __syncwarp();
        int last = 0;
        if (lane == 0) last = (atomicAdd(&cnt[rb], 1) == R.ne - 1);
        last = __shfl_sync(0xffffffffu, last, 0);
        if (last) {   // every expert has written its pairs for these 16 rows: fixed-order sum over the 8 slots of each token
          __threadfence();
          for (int i = lane; i < M * 16; i += 32) {
            const int tok = i / 16, r = rb * 16 + i % 16;
            float sum = 0.f;
            for (int k = 0; k < TOPK; ++k) sum = __fadd_rn(sum, __bfloat162float(__ldcg(cache3 + (size_t)(tok * TOPK + k) * K_HID + r)));
            out[(size_t)tok * K_HID + r] = __float2bfloat16(sum);
          }
          if (lane == 0) cnt[rb] = 0;
        }
      }
    }
  }
  wait_n<0>();
}

template <int NB> static size_t smem_up() { return ((sizeof(Route) + E_ALL * 4 + 15) / 16) * 16 + 8 * XPAD + 8 * 16 * 4 + Up<NB>::ST * 32 * Up<NB>::RP + 16 * 8 * 4; }
template <int NB> static size_t smem_down() { return ((sizeof(Route) + E_ALL * 4 + 15) / 16) * 16 + MAXP * APAD + MAXP * 4 * 4 + MAXP * 4 + Dn<NB>::ST * 16 * Dn<NB>::RP; }

template <int VAR, int NB> static int launch_up(int grid, cudaStream_t s, const uint8_t* xq, const float* xs, int rs, int cs, const uint8_t* w13,
                                                 const float* s13, const void* ids, int ids64, bf* c2, int M) {
  static bool set = false;
  if (!set) { if (cudaFuncSetAttribute(s5_moe_up_kernel<VAR, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_up<NB>())) return 1001; set = true; }
  s5_moe_up_kernel<VAR, NB><<<grid, 128, smem_up<NB>(), s>>>(xq, xs, rs, cs, w13, s13, ids, ids64, c2, M);
  return (int)cudaGetLastError();
}
template <int VAR, int NB> static int launch_down(int grid, cudaStream_t s, const uint8_t* aq, const float* as, int rs, int cs, const uint8_t* w2,
                                                   const float* s2, const void* ids, int ids64, const float* tw, bf* c3, bf* out, int* cnt, int M) {
  static bool set = false;
  if (!set) { if (cudaFuncSetAttribute(s5_moe_down_kernel<VAR, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_down<NB>())) return 1002; set = true; }
  s5_moe_down_kernel<VAR, NB><<<grid, 128, smem_down<NB>(), s>>>(aq, as, rs, cs, w2, s2, ids, ids64, tw, c3, out, cnt, M);
  return (int)cudaGetLastError();
}

// var: bits 0-2 rounding variant (only 1 and 3 are built: the two closest to stock in Session 6), bits 4-5 log2(NB).
template <int V> struct Dummy {};
#define S5_DISPATCH(FN, ...) { const int v = var & 7, nb = (var >> 4) & 3; \
  if (v == 1 && nb == 0) return FN<1, 1>(__VA_ARGS__); if (v == 1 && nb == 1) return FN<1, 2>(__VA_ARGS__); if (v == 1 && nb == 2) return FN<1, 4>(__VA_ARGS__); \
  if (v == 3 && nb == 0) return FN<3, 1>(__VA_ARGS__); if (v == 3 && nb == 1) return FN<3, 2>(__VA_ARGS__); if (v == 3 && nb == 2) return FN<3, 4>(__VA_ARGS__); \
  return 902; }

extern "C" int s5_moe_up(int var, int grid, void* stream, const void* xq, const float* xs, int xs_rs, int xs_cs, const void* w13,
                         const float* s13, const void* ids, int ids64, void* cache2, int M) {
  if (M < 1 || M > 8) return 900;
  S5_DISPATCH(launch_up, grid, (cudaStream_t)stream, (const uint8_t*)xq, xs, xs_rs, xs_cs, (const uint8_t*)w13, s13, ids, ids64, (bf*)cache2, M)
}
extern "C" int s5_moe_down(int var, int grid, void* stream, const void* aq, const float* as, int as_rs, int as_cs, const void* w2,
                           const float* s2, const void* ids, int ids64, const float* tw, void* cache3, void* out, int* cnt, int M) {
  if (M < 1 || M > 8) return 900;
  S5_DISPATCH(launch_down, grid, (cudaStream_t)stream, (const uint8_t*)aq, as, as_rs, as_cs, (const uint8_t*)w2, s2, ids, ids64, tw,
              (bf*)cache3, (bf*)out, cnt, M)
}
extern "C" int s5_moe_smem(int which) { return (int)(which ? smem_down<4>() : smem_up<4>()); }
