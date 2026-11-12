// Session 8, routed-MoE kernels v3 for decode verify (M <= 8, top-8 of 256, FP8 block-128), SM120. Redesigned from the Session 7 NCU
// profile of v2 (s5_moe.cu): v2 up ran at 72% DRAM with one 72 KB CTA per SM (4 warps, 0.07 eligible), v2 down at 40% because every
// CTA serialized 24 fence + atomic + sum epilogues. Idea credit as before: weight-major / SwapAB scheduling (MonoMoE, arXiv 2609.04244).
//
// Both kernels: 4 warps per CTA, each warp an independent worker with its own cp.async ring (warp-level sync only), ~40 KB shared
// memory per CTA so 2 CTAs fit per SM (8 independent streams per SM). Activations and scales are read through L1/L2, not staged.
// Arithmetic per output element is the v2 variant 1 (FMA in the block promotion, __expf, SiLU not pre-rounded); mma.sync m16n8k32 e4m3.
//
//   up:   one warp = one unit (16 gate + 16 up rows of one expert, K 2048); gate and up tiles in the same warp, so SiLU-mul is done in
//         registers. Units are dealt round-robin over all warps of the grid.
//   down: stream-K. The flattened list of (16-row block, expert) items, row-block major, is cut into one equal contiguous range per
//         warp. A warp sums its experts' contributions for a row block in shared memory (each (row, token) has exactly one owner lane
//         per item: a token appears at most once per expert), writes rows it covered completely, and leaves a partial only for the row
//         blocks at its range ends. Those are combined by the last contributor (atomic counter, reset by it) in fixed warp order.
//         At most two fences per warp, no per-pair round trip. Per-pair values are still rounded to bf16 before summing, as in the
//         stock path; the summation order is by expert, not by slot (S track, not bit-exact).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
using bf = __nv_bfloat16;

constexpr int K_HID = 2048, N_INT = 512, E_ALL = 256, TOPK = 8, MAXP = 64;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void wait_n() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ uint32_t lds32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t ldg32(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float promote(float acc, float t, float sa, float sw) { return fmaf(__fmul_rn(t, sa), sw, acc); }

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
constexpr int ROUTE_BYTES = ((sizeof(Route) + E_ALL * 4 + 15) / 16) * 16;

// ---------------------------------------------------------------- up: gate/up + SiLU-mul
constexpr int U_RP = 144;  // 128-byte K chunk + 16 B pad (rows 4 banks apart)
template <int ST>
__global__ void __launch_bounds__(128) s7_up_kernel(const uint8_t* __restrict__ xq, const float* __restrict__ xs, int xs_rs, int xs_cs,
                                                     const uint8_t* __restrict__ w13, const float* __restrict__ s13, const void* ids,
                                                     int ids64, bf* __restrict__ cache2, int M) {
  extern __shared__ __align__(16) uint8_t sm[];
  Route& R = *reinterpret_cast<Route*>(sm);
  int* map = reinterpret_cast<int*>(sm + sizeof(Route));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  uint8_t* ring = sm + ROUTE_BYTES + warp * ST * 32 * U_RP;
  build_route(R, map, ids, ids64, M * TOPK);
  const int units = R.ne * (N_INT / 16), W = gridDim.x * 4;
  for (int u = blockIdx.x * 4 + warp; u < units; u += W) {
    const int es = u >> 5, j = u & 31, e = R.ex[es], nt = R.cnt[es];
    const int tg = g < nt ? R.pr[es][g] / TOPK : -1;
    const uint8_t* wbase = w13 + (size_t)e * (2 * N_INT) * K_HID;
    auto load = [&](int kc) {
      uint8_t* dst = ring + (kc % ST) * 32 * U_RP;
#pragma unroll
      for (int i = lane; i < 32 * 8; i += 32) {
        const int r = i >> 3, c = (i & 7) * 16;
        const int row = (r < 16) ? j * 16 + r : N_INT + j * 16 + (r - 16);
        cp16(dst + r * U_RP + c, wbase + (size_t)row * K_HID + kc * 128 + c);
      }
      commit();
    };
#pragma unroll
    for (int kc = 0; kc < ST - 1; ++kc) load(kc);
    float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
    float sa[2];
    for (int kc = 0; kc < K_HID / 128; ++kc) {
      wait_n<ST - 2>();
      __syncwarp();
      if (kc + ST - 1 < K_HID / 128) load(kc + ST - 1); else commit();
      const uint8_t* A = ring + (kc % ST) * 32 * U_RP;
      uint32_t b[8];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        b[2 * kk] = tg >= 0 ? ldg32(xq + (size_t)tg * K_HID + kc * 128 + kk * 32 + q * 4) : 0u;
        b[2 * kk + 1] = tg >= 0 ? ldg32(xq + (size_t)tg * K_HID + kc * 128 + kk * 32 + 16 + q * 4) : 0u;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) { const int n = q * 2 + h; sa[h] = n < nt ? __ldg(xs + (R.pr[es][n] / TOPK) * xs_rs + kc * xs_cs) : 0.f; }
      float tg4[4] = {0.f, 0.f, 0.f, 0.f}, tu4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int k0 = kk * 32 + q * 4;
        mma_e4m3(tg4, lds32(A + g * U_RP + k0), lds32(A + (g + 8) * U_RP + k0), lds32(A + g * U_RP + k0 + 16), lds32(A + (g + 8) * U_RP + k0 + 16), b[2 * kk], b[2 * kk + 1]);
        const uint8_t* Au = A + 16 * U_RP;
        mma_e4m3(tu4, lds32(Au + g * U_RP + k0), lds32(Au + (g + 8) * U_RP + k0), lds32(Au + g * U_RP + k0 + 16), lds32(Au + (g + 8) * U_RP + k0 + 16), b[2 * kk], b[2 * kk + 1]);
      }
      const float swg = __ldg(s13 + ((size_t)e * (2 * N_INT / 128) + (j * 16) / 128) * (K_HID / 128) + kc);
      const float swu = __ldg(s13 + ((size_t)e * (2 * N_INT / 128) + N_INT / 128 + (j * 16) / 128) * (K_HID / 128) + kc);
#pragma unroll
      for (int i = 0; i < 4; ++i) { ag[i] = promote(ag[i], tg4[i], sa[i & 1], swg); au[i] = promote(au[i], tu4[i], sa[i & 1], swu); }
    }
    wait_n<0>();
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = q * 2 + (i & 1), r = g + 8 * (i >> 1);
      if (n < nt) {
        const float gv = __bfloat162float(__float2bfloat16(ag[i])), uv = __bfloat162float(__float2bfloat16(au[i]));
        const float sl = gv / (1.0f + __expf(-gv));
        cache2[(size_t)R.pr[es][n] * N_INT + j * 16 + r] = __float2bfloat16(__fmul_rn(sl, uv));
      }
    }
  }
}

// ---------------------------------------------------------------- down: stream-K over (row block, expert)
constexpr int D_RP = 272;  // 256-byte K half + 16 B pad
__device__ __forceinline__ long long rstart(long long T, int W, int w) { return T * w / W; }
__device__ __forceinline__ int warp_of(long long item, long long T, int W) {
  int w = (int)(item * W / T);
  while (w + 1 <= W && rstart(T, W, w + 1) <= item) ++w;
  while (w > 0 && rstart(T, W, w) > item) --w;
  return w;
}
template <int ST>
__global__ void __launch_bounds__(128) s7_down_kernel(const uint8_t* __restrict__ aq, const float* __restrict__ as, int as_rs, int as_cs,
                                                       const uint8_t* __restrict__ w2, const float* __restrict__ s2, const void* ids,
                                                       int ids64, const float* __restrict__ tw, float* __restrict__ part, bf* __restrict__ out,
                                                       int* __restrict__ cnt, int M) {
  extern __shared__ __align__(16) uint8_t sm[];
  Route& R = *reinterpret_cast<Route*>(sm);
  int* map = reinterpret_cast<int*>(sm + sizeof(Route));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  uint8_t* ring = sm + ROUTE_BYTES + warp * ST * 16 * D_RP;
  float* sacc = reinterpret_cast<float*>(sm + ROUTE_BYTES + 4 * ST * 16 * D_RP) + warp * 16 * 8;   // [16 rows][8 tokens]
  build_route(R, map, ids, ids64, M * TOPK);
  const int ne = R.ne, W = gridDim.x * 4, w = blockIdx.x * 4 + warp;
  const long long T = (long long)(K_HID / 16) * ne;
  const long long i0 = rstart(T, W, w), i1 = rstart(T, W, w + 1);
  if (ne == 0 || i0 >= i1) return;   // no CTA-wide barrier follows
  for (int i = lane; i < 128; i += 32) sacc[i] = 0.f;
  __syncwarp();
  const int first_rb = (int)(i0 / ne);
  const int S = (int)(i1 - i0) * 2;   // two K halves (256 each) per item
  auto load = [&](int s) {
    const long long it = i0 + s / 2;
    const int half = s & 1, rb = (int)(it / ne), es = (int)(it % ne), e = R.ex[es];
    uint8_t* dst = ring + (s % ST) * 16 * D_RP;
#pragma unroll
    for (int i = lane; i < 16 * 16; i += 32) {
      const int r = i >> 4, c = (i & 15) * 16;
      cp16(dst + r * D_RP + c, w2 + ((size_t)e * K_HID + rb * 16 + r) * N_INT + half * 256 + c);
    }
    commit();
  };
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) { if (s < S) load(s); else commit(); }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < S; ++s) {
    wait_n<ST - 2>();
    __syncwarp();
    if (s + ST - 1 < S) load(s + ST - 1); else commit();
    const long long it = i0 + s / 2;
    const int half = s & 1, rb = (int)(it / ne), es = (int)(it % ne), e = R.ex[es], nt = R.cnt[es];
    const int pg = g < nt ? R.pr[es][g] : -1;
    const uint8_t* A0 = ring + (s % ST) * 16 * D_RP;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      const int kc = half * 2 + nb;
      const uint8_t* A = A0 + nb * 128;
      float t4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int k0 = kk * 32 + q * 4;
        const uint32_t b0 = pg >= 0 ? ldg32(aq + (size_t)pg * N_INT + kc * 128 + kk * 32 + q * 4) : 0u;
        const uint32_t b1 = pg >= 0 ? ldg32(aq + (size_t)pg * N_INT + kc * 128 + kk * 32 + 16 + q * 4) : 0u;
        mma_e4m3(t4, lds32(A + g * D_RP + k0), lds32(A + (g + 8) * D_RP + k0), lds32(A + g * D_RP + k0 + 16), lds32(A + (g + 8) * D_RP + k0 + 16), b0, b1);
      }
      const float sw = __ldg(s2 + ((size_t)e * (K_HID / 128) + (rb * 16) / 128) * (N_INT / 128) + kc);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = q * 2 + (i & 1);
        const float sa = n < nt ? __ldg(as + R.pr[es][n] * as_rs + kc * as_cs) : 0.f;
        acc[i] = promote(acc[i], t4[i], sa, sw);
      }
    }
    if (half == 1) {   // item (rb, expert) complete: add its routed-weighted, bf16-rounded pairs into the row block's sums
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = q * 2 + (i & 1), r = g + 8 * (i >> 1);
        if (n < nt) {
          const int p = R.pr[es][n];
          sacc[r * 8 + p / TOPK] += __bfloat162float(__float2bfloat16(__fmul_rn(acc[i], __ldg(tw + p))));
        }
        acc[i] = 0.f;
      }
      __syncwarp();
      const bool last_item = (it + 1 >= i1), next_rb_differs = last_item || ((it + 1) / ne != rb);
      if (next_rb_differs) {
        const bool full = (i0 <= (long long)rb * ne) && (i1 >= (long long)(rb + 1) * ne);
        if (full) {
          for (int x = lane; x < 16 * M; x += 32) {
            const int r = x & 15, t = x >> 4;
            out[(size_t)t * K_HID + rb * 16 + r] = __float2bfloat16(sacc[r * 8 + t]);
          }
        } else {
          const int side = (rb == first_rb) ? 0 : 1;
          float* mine = part + ((size_t)w * 2 + side) * 128;
          for (int x = lane; x < 128; x += 32) mine[x] = sacc[x];
          __threadfence();
          __syncwarp();
          const int wlo = warp_of((long long)rb * ne, T, W), whi = warp_of((long long)(rb + 1) * ne - 1, T, W);
          int last = 0;
          if (lane == 0) last = (atomicAdd(&cnt[rb], 1) == whi - wlo);
          last = __shfl_sync(0xffffffffu, last, 0);
          if (last) {
            __threadfence();
            for (int x = lane; x < 16 * M; x += 32) {
              const int r = x & 15, t = x >> 4;
              float sum = 0.f;
              for (int ww = wlo; ww <= whi; ++ww) {
                const int sd = (rb == (int)(rstart(T, W, ww) / ne)) ? 0 : 1;
                sum = __fadd_rn(sum, __ldcg(part + ((size_t)ww * 2 + sd) * 128 + r * 8 + t));
              }
              out[(size_t)t * K_HID + rb * 16 + r] = __float2bfloat16(sum);
            }
            if (lane == 0) cnt[rb] = 0;
          }
        }
        __syncwarp();
        for (int x = lane; x < 128; x += 32) sacc[x] = 0.f;
        __syncwarp();
      }
    }
  }
  wait_n<0>();
}

template <int ST> static size_t smem_up() { return ROUTE_BYTES + 4 * ST * 32 * U_RP; }
template <int ST> static size_t smem_down() { return ROUTE_BYTES + 4 * ST * 16 * D_RP + 4 * 16 * 8 * 4; }

template <int ST> static int up_t(int grid, cudaStream_t s, const uint8_t* xq, const float* xs, int rs, int cs, const uint8_t* w13,
                                  const float* s13, const void* ids, int ids64, bf* c2, int M) {
  static bool set = false;
  if (!set) { if (cudaFuncSetAttribute(s7_up_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_up<ST>())) return 1001; set = true; }
  s7_up_kernel<ST><<<grid, 128, smem_up<ST>(), s>>>(xq, xs, rs, cs, w13, s13, ids, ids64, c2, M);
  return (int)cudaGetLastError();
}
template <int ST> static int down_t(int grid, cudaStream_t s, const uint8_t* aq, const float* as, int rs, int cs, const uint8_t* w2,
                                    const float* s2, const void* ids, int ids64, const float* tw, float* part, bf* out, int* cnt, int M) {
  static bool set = false;
  if (!set) { if (cudaFuncSetAttribute(s7_down_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_down<ST>())) return 1002; set = true; }
  s7_down_kernel<ST><<<grid, 128, smem_down<ST>(), s>>>(aq, as, rs, cs, w2, s2, ids, ids64, tw, part, out, cnt, M);
  return (int)cudaGetLastError();
}

// st: ring stages per warp (2 keeps both kernels near 40 KB of shared memory: 2 CTAs per SM; 3 trades that for depth).
// part: float scratch of at least grid * 4 warps * 2 * 128 floats; cnt: 128 ints, zero at first use (the kernel leaves them zero).
extern "C" int s7_moe_up(int st, int grid, void* stream, const void* xq, const float* xs, int xs_rs, int xs_cs, const void* w13,
                         const float* s13, const void* ids, int ids64, void* cache2, int M) {
  if (M < 1 || M > 8) return 900;
  if (st == 3) return up_t<3>(grid, (cudaStream_t)stream, (const uint8_t*)xq, xs, xs_rs, xs_cs, (const uint8_t*)w13, s13, ids, ids64, (bf*)cache2, M);
  return up_t<2>(grid, (cudaStream_t)stream, (const uint8_t*)xq, xs, xs_rs, xs_cs, (const uint8_t*)w13, s13, ids, ids64, (bf*)cache2, M);
}
extern "C" int s7_moe_down(int st, int grid, void* stream, const void* aq, const float* as, int as_rs, int as_cs, const void* w2,
                           const float* s2, const void* ids, int ids64, const float* tw, float* part, void* out, int* cnt, int M) {
  if (M < 1 || M > 8) return 900;
  if (st == 3) return down_t<3>(grid, (cudaStream_t)stream, (const uint8_t*)aq, as, as_rs, as_cs, (const uint8_t*)w2, s2, ids, ids64, tw, part, (bf*)out, cnt, M);
  return down_t<2>(grid, (cudaStream_t)stream, (const uint8_t*)aq, as, as_rs, as_cs, (const uint8_t*)w2, s2, ids, ids64, tw, part, (bf*)out, cnt, M);
}
extern "C" int s7_moe_smem(int which, int st) { return (int)(which ? (st == 3 ? smem_down<3>() : smem_down<2>()) : (st == 3 ? smem_up<3>() : smem_up<2>())); }
