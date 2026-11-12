/* Session 5 (our variant of SGLang's Apache-2.0 SM120 blockwise FP8 GEMM, see fp8_blockwise_scaled_mm_sm120.cuh).
Decode verify (M <= 16) only: the upstream swapAB kernel with a narrower weight tile (64 rows instead of 128, on the pingpong schedule: the cooperative one needs >= 128 rows), so a 2048-row
projection runs on 32 or 64 CTAs instead of 16. The K walk (128 per stage, mma order inside), the FP32 promotion per K block and the
epilogue are the same template, so each output element sees the same arithmetic. The weight scales come in per 64- or 32-row block:
the caller repeats each 128-row scale, so every row multiplies by exactly the value it had. Checked bitwise in s5_dense_exact.py. */
#include "fp8_blockwise_scaled_mm_sm120.cuh"

namespace sglang {

// scales_b_rep: [K/128, N/T] view of the weight scale repeated 128/T times along the weight rows (K-major, like the original).
template <int T>
void s5_narrow(tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView sa, tvm::ffi::TensorView sb) {
  RuntimeCheck(a.size(0) <= 32, "s5 narrow tile is for decode verify only");
  RuntimeCheck(b.size(1) % T == 0 && sb.size(1) == b.size(1) / T, "repeated weight scale shape");
  RuntimeCheck(host::is_type<bf16_t>(out.dtype()), "bf16 out only");
  const cudaStream_t stream = LaunchKernel::resolve_device(a.device());
  launch_sm120_fp8_blockwise_scaled_mm_swapab<
      cutlass::bfloat16_t,
      Shape<Int<T>, _32, _128>,
      Shape<Int<T>, _32, _128>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Shape<_1, _32, _1>,
      Shape<_1, _1, _1>,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>(out, a, b, sa, sb, stream);
}

void fp8_blockwise_scaled_mm_n64(tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView sa, tvm::ffi::TensorView sb) {
  s5_narrow<64>(out, a, b, sa, sb);
}

}  // namespace sglang
