// Session 5, our kernel for in_proj_ba (N=64, K=2048, M<=16, bf16): y[m, n] = x[m, k] w[n, k]^T with fp32 accumulators.
// Same per-element arithmetic as s4_ba (WMMA 16x16x16 executes as two mma.m16n8k16, one per 8-column half): here each CTA owns one
// 8-column half, so N=64 runs on 8 CTAs instead of 4, each streaming half the weight bytes. K is walked in ascending 16-steps as before.
// Loads: all of the CTA's weight rows and the activation are issued at once as 8 cp.async groups (one per K span) and the single
// computing warp waits group by group, so the MMA chain overlaps the remaining loads. Rows are padded by 16 bytes (no bank conflicts).
// Target: bitwise equal to s4_ba and to the cuBLAS kernel; checked in s5_dense_exact.py.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
using bf = __nv_bfloat16;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}
template <int N> __device__ __forceinline__ void wait_n() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void wait_dyn(int n) {
  switch (n) {
    case 0: wait_n<0>(); break; case 1: wait_n<1>(); break; case 2: wait_n<2>(); break; case 3: wait_n<3>(); break;
    case 4: wait_n<4>(); break; case 5: wait_n<5>(); break; case 6: wait_n<6>(); break; default: wait_n<7>(); break;
  }
}
__device__ __forceinline__ uint32_t ld32(const bf* p) { return *reinterpret_cast<const uint32_t*>(p); }

constexpr int SPANS = 8;

__global__ void s5_ba_kernel(const bf* __restrict__ x, const bf* __restrict__ w, bf* __restrict__ out, int m, int K, int n) {
  extern __shared__ uint4 smem4[];
  const int P = K + 8;                        // padded row stride (elements)
  bf* sw = reinterpret_cast<bf*>(smem4);      // [8][P]  weight rows j0..j0+7
  bf* sx = sw + 8 * P;                        // [16][P] activation rows (rows >= m are never read as data)
  const int j0 = blockIdx.x * 8, t = threadIdx.x, T = blockDim.x, L = K / SPANS;
  for (int r = m; r < 8; ++r)                 // rows m..7 feed the A fragment's low half: must be zero-valued
    for (int i = t; i < K / 8; i += T) reinterpret_cast<uint4*>(sx + r * P)[i] = make_uint4(0, 0, 0, 0);
  for (int s = 0; s < SPANS; ++s) {
    const int k1 = s * L;
    for (int i = t; i < 8 * (L / 8); i += T) {
      const int r = i / (L / 8), c = (i % (L / 8)) * 8;
      if (j0 + r < n) cp16(sw + r * P + k1 + c, w + (size_t)(j0 + r) * K + k1 + c);
    }
    for (int i = t; i < m * (L / 8); i += T) {
      const int r = i / (L / 8), c = (i % (L / 8)) * 8;
      cp16(sx + r * P + k1 + c, x + (size_t)r * K + k1 + c);
    }
    asm volatile("cp.async.commit_group;");
  }
  for (int r = 0; r < 8; ++r)                 // weight rows past n (not the case for N=64) read as zero
    if (j0 + r >= n) for (int i = t; i < K / 8; i += T) reinterpret_cast<uint4*>(sw + r * P)[i] = make_uint4(0, 0, 0, 0);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  const int lane = t & 31, g = lane >> 2, q = lane & 3;
  const bool hi = (g + 8) < m;
  for (int s = 0; s < SPANS; ++s) {
    wait_dyn(SPANS - 1 - s);
    __syncthreads();
    if (t < 32) {
      const int kend = (s + 1) * L;
      for (int k0 = s * L; k0 < kend; k0 += 16) {
        const uint32_t a0 = ld32(sx + g * P + k0 + q * 2);
        const uint32_t a2 = ld32(sx + g * P + k0 + q * 2 + 8);
        const uint32_t a1 = hi ? ld32(sx + (g + 8) * P + k0 + q * 2) : 0u;
        const uint32_t a3 = hi ? ld32(sx + (g + 8) * P + k0 + q * 2 + 8) : 0u;
        const uint32_t b0 = ld32(sw + g * P + k0 + q * 2);
        const uint32_t b1 = ld32(sw + g * P + k0 + q * 2 + 8);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  if (t < 32) {
    const int col = j0 + q * 2;
    if (g < m) {
      if (col < n) out[g * n + col] = __float2bfloat16(c0);
      if (col + 1 < n) out[g * n + col + 1] = __float2bfloat16(c1);
    }
    if (g + 8 < m) {
      if (col < n) out[(g + 8) * n + col] = __float2bfloat16(c2);
      if (col + 1 < n) out[(g + 8) * n + col + 1] = __float2bfloat16(c3);
    }
  }
}

extern "C" int s5_ba(bf* out, const bf* x, const bf* w, int m, int K, int n, int threads, void* stream) {
  if (m > 16 || K % (SPANS * 8) || (K + 8) * 24 * 2 > 101376) return 999;
  const size_t bytes = (size_t)24 * (K + 8) * sizeof(bf);
  static int set = 0;
  if (!set) {
    int e = (int)cudaFuncSetAttribute(s5_ba_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    if (e != 0) return 1000 + e;
    set = 1;
  }
  s5_ba_kernel<<<(n + 7) / 8, threads, bytes, (cudaStream_t)stream>>>(x, w, out, m, K, n);
  return (int)cudaGetLastError();
}
