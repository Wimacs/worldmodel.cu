#include "world_cuda_ops.cuh"

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#if __has_include("kernel_forward.h")
#define WORLD_HAS_CUTLASS_FMHA 1
#include "kernel_forward.h"
#include "world_sparse_fmha.cuh"
#else
#define WORLD_HAS_CUTLASS_FMHA 0
#endif

#include <stdio.h>

#define WORLD_ATTN_D64_Q_BLOCK 4
#define WORLD_ATTN_D64_K_BLOCK 64
#define WORLD_ATTN_D64_FLASH_WARPS 16

#define CUDA_OK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

#define CUTLASS_OK(expr) do { \
    cutlass::Status _s = (expr); \
    if (_s != cutlass::Status::kSuccess) { \
        fprintf(stderr, "CUTLASS error %s:%d: %s\n", __FILE__, __LINE__, cutlassGetStatusString(_s)); \
        return 1; \
    } \
} while (0)

static int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__device__ __forceinline__ float wm_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float wm_warp_sum(float x) {
    x += __shfl_down_sync(0xffffffffu, x, 16);
    x += __shfl_down_sync(0xffffffffu, x, 8);
    x += __shfl_down_sync(0xffffffffu, x, 4);
    x += __shfl_down_sync(0xffffffffu, x, 2);
    x += __shfl_down_sync(0xffffffffu, x, 1);
    return x;
}

__device__ __forceinline__ float wm_rope_phase(
        int pair_id,
        int64_t x_pos,
        int64_t y_pos,
        int64_t t_pos,
        const float *xy,
        const float *inv_t,
        int width,
        int height,
        int d_xy) {
    if (pair_id < d_xy) {
        float x = (2.0f * (float)x_pos + 1.0f) / (float)width - 1.0f;
        return x * xy[pair_id];
    }
    if (pair_id < 2 * d_xy) {
        float y = (2.0f * (float)y_pos + 1.0f) / (float)height - 1.0f;
        return y * xy[pair_id - d_xy];
    }
    return (float)t_pos * inv_t[pair_id - 2 * d_xy];
}

__global__ static void silu_f32_kernel(const float *x, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = wm_silu(x[i]);
}

__global__ static void f32_to_f16_kernel(const float *x, __half *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

// Dynamic symmetric per-row activation quantization.  Keeping one scale per
// token avoids a single outlier token reducing the precision of every row.
__global__ static void quantize_rows_f32_i8_kernel(
        const float *__restrict__ x,
        int8_t *__restrict__ q,
        float *__restrict__ scales,
        int rows,
        int cols) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (int64_t)row * cols;
    float amax = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        amax = fmaxf(amax, fabsf(row_x[col]));
    }
    red[tid] = amax;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }

    float scale = red[0] > 0.0f ? red[0] * (1.0f / 127.0f) : 1.0f;
    float inv_scale = red[0] > 0.0f ? 1.0f / scale : 0.0f;
    if (tid == 0) scales[row] = scale;
    for (int col = tid; col < cols; col += blockDim.x) {
        int v = __float2int_rn(row_x[col] * inv_scale);
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        q[(int64_t)row * cols + col] = (int8_t)v;
    }
}

// Fuses RMSNorm/AdaRMSNorm with dynamic A8 quantization.  A null mod_scale
// and bias implements plain RMSNorm (used by the controller branch).
__global__ static void rms_norm_quantize_rows_i8_kernel(
        const float *__restrict__ x,
        const float *__restrict__ mod_scale,
        const float *__restrict__ bias,
        int8_t *__restrict__ q,
        float *__restrict__ q_scales,
        int rows,
        int cols,
        float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (int64_t)row * cols;
    float sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = row_x[col];
        sum += v * v;
    }
    red[tid] = sum;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }

    float inv_rms = rsqrtf(red[0] / (float)cols + eps);
    float amax = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = row_x[col] * inv_rms;
        if (mod_scale) v *= 1.0f + mod_scale[col];
        if (bias) v += bias[col];
        amax = fmaxf(amax, fabsf(v));
    }
    red[tid] = amax;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }

    float q_scale = red[0] > 0.0f ? red[0] * (1.0f / 127.0f) : 1.0f;
    float inv_q_scale = red[0] > 0.0f ? 1.0f / q_scale : 0.0f;
    if (tid == 0) q_scales[row] = q_scale;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = row_x[col] * inv_rms;
        if (mod_scale) v *= 1.0f + mod_scale[col];
        if (bias) v += bias[col];
        int qi = __float2int_rn(v * inv_q_scale);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        q[(int64_t)row * cols + col] = (int8_t)qi;
    }
}

// Converts an INT32 GEMM result through dequantization and SiLU directly to
// the A8 input of the next GEMM.  This removes the large FP32 MLP hidden
// activation and its standalone SiLU and quantization passes.
__global__ static void dequant_silu_quantize_rows_i8_kernel(
        const int32_t *__restrict__ acc,
        const float *input_row_scales,
        const float *__restrict__ weight_scales,
        const float *__restrict__ bias,
        int8_t *__restrict__ q,
        float *output_row_scales,
        int rows,
        int cols) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    float input_scale = input_row_scales[row];
    const int32_t *row_acc = acc + (int64_t)row * cols;

    float amax = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = (float)row_acc[col] * input_scale * weight_scales[col];
        if (bias) v += bias[col];
        v = wm_silu(v);
        amax = fmaxf(amax, fabsf(v));
    }
    red[tid] = amax;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }

    float q_scale = red[0] > 0.0f ? red[0] * (1.0f / 127.0f) : 1.0f;
    float inv_q_scale = red[0] > 0.0f ? 1.0f / q_scale : 0.0f;
    if (tid == 0) output_row_scales[row] = q_scale;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = (float)row_acc[col] * input_scale * weight_scales[col];
        if (bias) v += bias[col];
        int qi = __float2int_rn(wm_silu(v) * inv_q_scale);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        q[(int64_t)row * cols + col] = (int8_t)qi;
    }
}

__global__ static void dequant_gated_residual_f32_kernel(
        const int32_t *__restrict__ acc,
        const float *__restrict__ row_scales,
        const float *__restrict__ col_scales,
        const float *__restrict__ residual,
        const float *__restrict__ gate,
        float *__restrict__ out,
        int rows,
        int cols) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t count = (int64_t)rows * cols;
    if (i >= count) return;
    int row = (int)(i / cols);
    int col = (int)(i - (int64_t)row * cols);
    float update = (float)acc[i] * row_scales[row] * col_scales[col];
    out[i] = residual[i] + update * gate[col];
}

__global__ static void dequant_add_residual_f32_kernel(
        const int32_t *__restrict__ acc,
        const float *__restrict__ row_scales,
        const float *__restrict__ col_scales,
        const float *__restrict__ residual,
        float *__restrict__ out,
        int rows,
        int cols) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t count = (int64_t)rows * cols;
    if (i >= count) return;
    int row = (int)(i / cols);
    int col = (int)(i - (int64_t)row * cols);
    float update = (float)acc[i] * row_scales[row] * col_scales[col];
    out[i] = residual[i] + update;
}

__global__ static void silu_f32_to_f16_kernel(const float *x, __half *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(wm_silu(x[i]));
}

__global__ static void add_bias_silu_f32_kernel(const float *x, const float *bias, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = wm_silu(x[i] + bias[i]);
}

__global__ static void add_channel_silu_inplace_f32_kernel(float *x, const float *bias, int rows, int D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)rows * D;
    if (i < n) x[i] = wm_silu(x[i] + bias[i % D]);
}

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = float,
    typename ElementCompute_ = float>
class LinearCombinationSilu {
public:
    using ElementOutput = ElementOutput_;
    using ElementSource = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementScalar = ElementCompute;
    using ElementC = ElementSource;
    using ElementD = ElementOutput;

    static int const kCount = Count;
    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentSource = cutlass::Array<ElementSource, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

    struct Params {
        ElementCompute alpha;

        CUTLASS_HOST_DEVICE
        Params(): alpha(ElementCompute(1)) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha_, ElementCompute = ElementCompute(0)): alpha(alpha_) {}
    };

private:
    ElementCompute alpha_;

public:
    CUTLASS_HOST_DEVICE
    LinearCombinationSilu(Params const &params, int = 0): alpha_(params.alpha) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return false; }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int) {}

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentSource const &) const {
        return (*this)(accumulator);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> accumulator_converter;
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> output_converter;
        FragmentCompute tmp = accumulator_converter(accumulator);
        for (int i = 0; i < kCount; ++i) {
            ElementCompute v = alpha_ * tmp[i];
            tmp[i] = v / (ElementCompute(1) + expf(-v));
        }
        return output_converter(tmp);
    }

    CUTLASS_HOST_DEVICE
    ElementD operator()(ElementAccumulator const accumulator, ElementC const) const {
        return (*this)(accumulator);
    }

    CUTLASS_HOST_DEVICE
    ElementD operator()(ElementAccumulator const accumulator) const {
        cutlass::NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
        cutlass::NumericConverter<ElementD, ElementCompute> output_converter;
        ElementCompute v = alpha_ * accumulator_converter(accumulator);
        return output_converter(v / (ElementCompute(1) + expf(-v)));
    }
};

__global__ static void ada_rms_norm_single_f32_kernel(
        const float *x,
        const float *scale,
        const float *bias,
        float *y,
        int rows,
        int D,
        float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (int64_t)row * D;
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = row_x[d];
        sum += v * v;
    }
    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)D + eps);
    float *row_y = y + (int64_t)row * D;
    for (int d = tid; d < D; d += blockDim.x) {
        row_y[d] = row_x[d] * inv * (1.0f + scale[d]) + bias[d];
    }
}

__global__ static void ada_rms_norm_single_f16_kernel(
        const float *x,
        const float *scale,
        const float *bias,
        __half *y,
        int rows,
        int D,
        float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (int64_t)row * D;
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = row_x[d];
        sum += v * v;
    }
    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)D + eps);
    __half *row_y = y + (int64_t)row * D;
    for (int d = tid; d < D; d += blockDim.x) {
        row_y[d] = __float2half_rn(row_x[d] * inv * (1.0f + scale[d]) + bias[d]);
    }
}

__global__ static void rms_norm_rows_f32_kernel(
        const float *x,
        float *y,
        int rows,
        int D,
        float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *row_x = x + (int64_t)row * D;
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = row_x[d];
        sum += v * v;
    }
    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)D + eps);
    float *row_y = y + (int64_t)row * D;
    for (int d = tid; d < D; d += blockDim.x) {
        row_y[d] = row_x[d] * inv;
    }
}

__global__ static void out_norm_silu_f32_kernel(
        const float *tokens,
        const float *mod,
        float *out,
        int rows,
        int D,
        float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float *row_x = tokens + (int64_t)row * D;
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = row_x[d];
        sum += v * v;
    }
    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)D + eps);
    float *row_y = out + (int64_t)row * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float a = mod[d];
        float b = mod[D + d];
        row_y[d] = wm_silu(row_x[d] * inv * (1.0f + a) + b);
    }
}

int wm_cuda_linear_f32(
        const float *x_rm,
        const float *w_rm,
        float *y_rm,
        int m,
        int k,
        int n) {
    using Gemm = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 64, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        1,
        1>;

    typename Gemm::Arguments args(
        {m, n, k},
        {x_rm, k},
        {w_rm, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

using WorldW8A8Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        int32_t,
        128 / cutlass::sizeof_bits<int32_t>::value,
        int32_t,
        int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

int wm_cuda_gemm_i8_i32_can_implement(
        const int8_t *x_i8,
        const int8_t *w_rm_i8,
        int32_t *acc_i32,
        int m,
        int k,
        int n) {
    if (!x_i8 || !w_rm_i8 || !acc_i32 || m <= 0 || k <= 0 || n <= 0) {
        fprintf(stderr, "invalid INT8 GEMM arguments M=%d K=%d N=%d\n", m, k, n);
        return 1;
    }
    if ((k & 31) != 0) {
        fprintf(stderr, "INT8 tensor-op requires K divisible by 32, got K=%d\n", k);
        return 1;
    }

    typename WorldW8A8Gemm::Arguments args(
        {m, n, k},
        {x_i8, k},
        {w_rm_i8, k},
        {acc_i32, n},
        {acc_i32, n},
        {1, 0});
    WorldW8A8Gemm gemm;
    CUTLASS_OK(gemm.can_implement(args));
    return 0;
}

int wm_cuda_gemm_i8_i32(
        const int8_t *x_i8,
        const int8_t *w_rm_i8,
        int32_t *acc_i32,
        int m,
        int k,
        int n) {
    if (wm_cuda_gemm_i8_i32_can_implement(x_i8, w_rm_i8, acc_i32, m, k, n)) {
        return 1;
    }
    typename WorldW8A8Gemm::Arguments args(
        {m, n, k},
        {x_i8, k},
        {w_rm_i8, k},
        {acc_i32, n},
        {acc_i32, n},
        {1, 0});
    WorldW8A8Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_linear_fp16_weight_simt(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n) {
    int64_t x_elems = (int64_t)m * k;
    f32_to_f16_kernel<<<div_up_i64(x_elems, 256), 256>>>(x_rm, x_half_tmp, x_elems);
    CUDA_OK(cudaGetLastError());

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 64, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        1,
        1>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_half_tmp);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_linear_fp16_weight_tensorop(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n) {
    int64_t x_elems = (int64_t)m * k;
    f32_to_f16_kernel<<<div_up_i64(x_elems, 256), 256>>>(x_rm, x_half_tmp, x_elems);
    CUDA_OK(cudaGetLastError());

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_half_tmp);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_linear_fp16_weight_tensorop_m64n64(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n) {
    int64_t x_elems = (int64_t)m * k;
    f32_to_f16_kernel<<<div_up_i64(x_elems, 256), 256>>>(x_rm, x_half_tmp, x_elems);
    CUDA_OK(cudaGetLastError());

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_half_tmp);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_linear_fp16_input_weight_tensorop_m64n64(
        const __half *x_rm_h,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_rm_h);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm.can_implement(args));
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_linear_fp16_weight_tensorop_m64n64_silu_half(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        __half *y_rm_h,
        int m,
        int k,
        int n) {
    int64_t x_elems = (int64_t)m * k;
    f32_to_f16_kernel<<<div_up_i64(x_elems, 256), 256>>>(x_rm, x_half_tmp, x_elems);
    CUDA_OK(cudaGetLastError());

    using Epilogue = LinearCombinationSilu<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float,
        float>;
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_half_tmp);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    cutlass::half_t *y_h = reinterpret_cast<cutlass::half_t *>(y_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_h, n},
        {y_h, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm.can_implement(args));
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_linear_fp16_input_weight_tensorop_m64n64_silu_half(
        const __half *x_rm_h,
        const __half *w_rm_h,
        __half *y_rm_h,
        int m,
        int k,
        int n) {
    using Epilogue = LinearCombinationSilu<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float,
        float>;
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_rm_h);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    cutlass::half_t *y_h = reinterpret_cast<cutlass::half_t *>(y_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_h, n},
        {y_h, n},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm.can_implement(args));
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_should_use_m64n64_tensorop(int enabled, int m, int k, int n) {
    return enabled && m > 1 && m <= 256 && (m % 64) == 0 &&
           k >= 1024 && n >= 1024 && (k % 32) == 0 && (n % 64) == 0;
}

size_t wm_cuda_linear_fp16_weight_tensorop_splitk_workspace_size(
        int m,
        int k,
        int n,
        int split_k_slices) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,
        8,
        8,
        true>;

    typename Gemm::Arguments args(
        {m, n, k},
        {static_cast<const cutlass::half_t *>(nullptr), k},
        {static_cast<const cutlass::half_t *>(nullptr), k},
        {static_cast<const float *>(nullptr), n},
        {static_cast<float *>(nullptr), n},
        {1.0f, 0.0f},
        split_k_slices);
    return Gemm::get_workspace_size(args);
}

int wm_cuda_linear_fp16_input_weight_tensorop_splitk(
        const __half *x_rm_h,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n,
        int split_k_slices,
        void *workspace,
        size_t workspace_bytes) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,
        8,
        8,
        true>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_rm_h);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f},
        split_k_slices);
    size_t needed_workspace = Gemm::get_workspace_size(args);
    if (needed_workspace > workspace_bytes || (needed_workspace > 0 && !workspace)) {
        fprintf(stderr,
                "split-K workspace too small: need %zu bytes have %zu bytes\n",
                needed_workspace,
                workspace_bytes);
        return 1;
    }
    Gemm gemm;
    CUTLASS_OK(gemm.can_implement(args));
    CUTLASS_OK(gemm(args, workspace));
    CUDA_OK(cudaGetLastError());
    return 0;
}

size_t wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel_workspace_size(
        int m,
        int k,
        int n,
        int split_k_slices) {
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
        float,
        128 / cutlass::sizeof_bits<float>::value,
        float,
        float>;
    using Gemm = cutlass::gemm::device::GemmSplitKParallel<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue>;

    typename Gemm::Arguments args(
        {m, n, k},
        {static_cast<const cutlass::half_t *>(nullptr), k},
        {static_cast<const cutlass::half_t *>(nullptr), k},
        {static_cast<const float *>(nullptr), n},
        {static_cast<float *>(nullptr), n},
        {1.0f, 0.0f},
        split_k_slices);
    return Gemm::get_workspace_size(args);
}

int wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel(
        const __half *x_rm_h,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n,
        int split_k_slices,
        void *workspace,
        size_t workspace_bytes) {
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
        float,
        128 / cutlass::sizeof_bits<float>::value,
        float,
        float>;
    using Gemm = cutlass::gemm::device::GemmSplitKParallel<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue>;

    const cutlass::half_t *x_h = reinterpret_cast<const cutlass::half_t *>(x_rm_h);
    const cutlass::half_t *w_h = reinterpret_cast<const cutlass::half_t *>(w_rm_h);
    typename Gemm::Arguments args(
        {m, n, k},
        {x_h, k},
        {w_h, k},
        {y_rm, n},
        {y_rm, n},
        {1.0f, 0.0f},
        split_k_slices);
    size_t needed_workspace = Gemm::get_workspace_size(args);
    if (needed_workspace > workspace_bytes || (needed_workspace > 0 && !workspace)) {
        fprintf(stderr,
                "parallel split-K workspace too small: need %zu bytes have %zu bytes\n",
                needed_workspace,
                workspace_bytes);
        return 1;
    }
    Gemm gemm;
    CUTLASS_OK(gemm.can_implement(args));
    CUTLASS_OK(gemm(args, workspace));
    CUDA_OK(cudaGetLastError());
    return 0;
}


__global__ static void qkv_separate_rms_rope_f32_kernel(
        const float *q_raw,
        const float *k_raw,
        const float *v_raw,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int T,
        int n_heads,
        int n_kv_heads,
        int d_head,
        int width,
        int height,
        float eps) {
    extern __shared__ float sh[];
    float *vals = sh;
    float *red = sh + d_head;

    int t = blockIdx.x;
    int role = blockIdx.y;
    int tid = threadIdx.x;
    int kv_dim = n_kv_heads * d_head;
    int half = d_head / 2;
    int d_xy = d_head / 8;

    if (role < n_heads + n_kv_heads) {
        int is_k = role >= n_heads;
        int h = is_k ? role - n_heads : role;
        const float *src = is_k
            ? k_raw + (int64_t)t * kv_dim + h * d_head
            : q_raw + (int64_t)t * (n_heads * d_head) + h * d_head;

        float sum = 0.0f;
        for (int d = tid; d < d_head; d += blockDim.x) {
            float z = src[d];
            vals[d] = z;
            sum += z * z;
        }
        red[tid] = sum;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }

        float inv = rsqrtf(red[0] / (float)d_head + eps);
        float *dst = is_k
            ? k + (((int64_t)h * T + t) * d_head)
            : q + (((int64_t)h * T + t) * d_head);

        for (int p = tid; p < half; p += blockDim.x) {
            float phase = wm_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, width, height, d_xy);
            float c = cosf(phase);
            float s = sinf(phase);
            float a = vals[2 * p] * inv;
            float b = vals[2 * p + 1] * inv;
            dst[p] = a * c - b * s;
            dst[half + p] = b * c + a * s;
        }
    } else {
        int h = role - n_heads - n_kv_heads;
        const float *src = v_raw + (int64_t)t * kv_dim + h * d_head;
        float *dst = v + (((int64_t)h * T + t) * d_head);
        for (int d = tid; d < d_head; d += blockDim.x) {
            dst[d] = src[d];
        }
    }
}

__global__ static void qkv_fused_rms_rope_f32_kernel(
        const float *qkv_raw,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int T,
        int n_heads,
        int n_kv_heads,
        int d_head,
        int width,
        int height,
        float eps) {
    extern __shared__ float sh[];
    float *vals = sh;
    float *red = sh + d_head;

    int t = blockIdx.x;
    int role = blockIdx.y;
    int tid = threadIdx.x;
    int q_dim = n_heads * d_head;
    int kv_dim = n_kv_heads * d_head;
    int qkv_dim = q_dim + 2 * kv_dim;
    int half = d_head / 2;
    int d_xy = d_head / 8;
    const float *row = qkv_raw + (int64_t)t * qkv_dim;

    if (role < n_heads + n_kv_heads) {
        int is_k = role >= n_heads;
        int h = is_k ? role - n_heads : role;
        const float *src = is_k ? row + q_dim + h * d_head : row + h * d_head;

        float sum = 0.0f;
        for (int d = tid; d < d_head; d += blockDim.x) {
            float z = src[d];
            vals[d] = z;
            sum += z * z;
        }
        red[tid] = sum;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }

        float inv = rsqrtf(red[0] / (float)d_head + eps);
        float *dst = is_k
            ? k + (((int64_t)h * T + t) * d_head)
            : q + (((int64_t)h * T + t) * d_head);

        for (int p = tid; p < half; p += blockDim.x) {
            float phase = wm_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, width, height, d_xy);
            float c = cosf(phase);
            float s = sinf(phase);
            float a = vals[2 * p] * inv;
            float b = vals[2 * p + 1] * inv;
            dst[p] = a * c - b * s;
            dst[half + p] = b * c + a * s;
        }
    } else {
        int h = role - n_heads - n_kv_heads;
        const float *src = row + q_dim + kv_dim + h * d_head;
        float *dst = v + (((int64_t)h * T + t) * d_head);
        for (int d = tid; d < d_head; d += blockDim.x) {
            dst[d] = src[d];
        }
    }
}

__global__ static void qkv_fused_rms_rope_i32_dequant_kernel(
        const int32_t *__restrict__ qkv_acc,
        const float *__restrict__ row_scales,
        const float *__restrict__ weight_scales,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int T,
        int n_heads,
        int n_kv_heads,
        int d_head,
        int width,
        int height,
        float eps) {
    extern __shared__ float sh[];
    float *vals = sh;
    float *red = sh + d_head;

    int t = blockIdx.x;
    int role = blockIdx.y;
    int tid = threadIdx.x;
    int q_dim = n_heads * d_head;
    int kv_dim = n_kv_heads * d_head;
    int qkv_dim = q_dim + 2 * kv_dim;
    int half = d_head / 2;
    int d_xy = d_head / 8;
    const int32_t *row = qkv_acc + (int64_t)t * qkv_dim;
    float row_scale = row_scales[t];

    if (role < n_heads + n_kv_heads) {
        int is_k = role >= n_heads;
        int h = is_k ? role - n_heads : role;
        int base = is_k ? q_dim + h * d_head : h * d_head;

        float sum = 0.0f;
        for (int d = tid; d < d_head; d += blockDim.x) {
            int col = base + d;
            float z = (float)row[col] * row_scale * weight_scales[col];
            vals[d] = z;
            sum += z * z;
        }
        red[tid] = sum;
        __syncthreads();
        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }

        float inv = rsqrtf(red[0] / (float)d_head + eps);
        float *dst = is_k
            ? k + (((int64_t)h * T + t) * d_head)
            : q + (((int64_t)h * T + t) * d_head);
        for (int p = tid; p < half; p += blockDim.x) {
            float phase = wm_rope_phase(
                p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, width, height, d_xy);
            float c = cosf(phase);
            float s = sinf(phase);
            float a = vals[2 * p] * inv;
            float b = vals[2 * p + 1] * inv;
            dst[p] = a * c - b * s;
            dst[half + p] = b * c + a * s;
        }
    } else {
        int h = role - n_heads - n_kv_heads;
        int base = q_dim + kv_dim + h * d_head;
        float *dst = v + (((int64_t)h * T + t) * d_head);
        for (int d = tid; d < d_head; d += blockDim.x) {
            int col = base + d;
            dst[d] = (float)row[col] * row_scale * weight_scales[col];
        }
    }
}

__global__ static void current_frame_attention_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        float *out_tokens,
        int Hq,
        int Hkv,
        int T,
        int D,
        float scale) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int tq = row % T;
    int hq = row / T;
    int group = Hq / Hkv;
    int hk = hq / group;

    const float *qrow = q + (((int64_t)hq * T + tq) * D);
    const float *kbase = k + ((int64_t)hk * T * D);
    const float *vbase = v + ((int64_t)hk * T * D);
    float *orow = out_tokens + (int64_t)tq * (Hq * D) + hq * D;

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int tk = 0; tk < T; ++tk) {
        float partial = 0.0f;
        const float *krow = kbase + (int64_t)tk * D;
        for (int d = tid; d < D; d += blockDim.x) {
            partial += qrow[d] * krow[d];
        }

        red[tid] = partial;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }

        float score = red[0] * scale;
        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);

        if (tid < D) {
            acc = acc * alpha + beta * vbase[(int64_t)tk * D + tid];
        }
        l = l * alpha + beta;
        m = new_m;
        __syncthreads();
    }

    if (tid < D) {
        orow[tid] = acc / l;
    }
}

__global__ static void init_cache_written_kernel(bool *written, int ring_length, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int capacity = ring_length + T;
    if (i < capacity) written[i] = i >= ring_length;
}

__global__ static void kv_cache_upsert_copy_f32_kernel(
        float *cache_k,
        float *cache_v,
        const float *k,
        const float *v,
        bool *written,
        int H,
        int T,
        int D,
        int ring_length,
        int base,
        bool write_step,
        bool frozen) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * T * D;
    if (i >= total) return;

    int d = (int)(i % D);
    int64_t q = i / D;
    int t = (int)(q % T);
    int h = (int)(q / T);

    int tail_idx = ring_length + t;
    int ring_idx = base + t;
    int dst_idx = (!frozen && write_step) ? ring_idx : tail_idx;

    int capacity = ring_length + T;
    int64_t src = ((int64_t)h * T + t) * D + d;
    int64_t tail = ((int64_t)h * capacity + tail_idx) * D + d;
    int64_t dst = ((int64_t)h * capacity + dst_idx) * D + d;

    float kv = k[src];
    float vv = v[src];
    cache_k[tail] = kv;
    cache_v[tail] = vv;
    if (!frozen) {
        cache_k[dst] = kv;
        cache_v[dst] = vv;
    }

    if (h == 0 && d == 0) {
        written[tail_idx] = true;
        if (!frozen) written[dst_idx] = true;
    }
}

__global__ static void kv_cache_upsert_copy_f16_kernel(
        __half *cache_k,
        __half *cache_v,
        const float *k,
        const float *v,
        bool *written,
        int H,
        int T,
        int D,
        int ring_length,
        int base,
        bool write_step,
        bool frozen) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * T * D;
    if (i >= total) return;

    int d = (int)(i % D);
    int64_t q = i / D;
    int t = (int)(q % T);
    int h = (int)(q / T);

    int tail_idx = ring_length + t;
    int ring_idx = base + t;
    int dst_idx = (!frozen && write_step) ? ring_idx : tail_idx;

    int capacity = ring_length + T;
    int64_t src = ((int64_t)h * T + t) * D + d;
    int64_t tail = ((int64_t)h * capacity + tail_idx) * D + d;
    int64_t dst = ((int64_t)h * capacity + dst_idx) * D + d;

    __half kv = __float2half_rn(k[src]);
    __half vv = __float2half_rn(v[src]);
    cache_k[tail] = kv;
    cache_v[tail] = vv;
    if (!frozen) {
        cache_k[dst] = kv;
        cache_v[dst] = vv;
    }

    if (h == 0 && d == 0) {
        written[tail_idx] = true;
        if (!frozen) written[dst_idx] = true;
    }
}

__global__ static void collect_cache_frame_indices_kernel(
        const bool *written,
        int64_t *indices,
        int32_t *block_ids,
        int *count,
        int capacity,
        int T,
        int base,
        bool write_step) {
    __shared__ int out_base;
    int slot = blockIdx.x;
    int tid = threadIdx.x;
    int slots = capacity / T;
    if (slot >= slots) return;

    int slot_base = slot * T;
    bool slot_written = written[slot_base] && !(write_step && slot_base == base);
    if (tid == 0) {
        int prefix_slots = 0;
        for (int s = 0; s < slot; ++s) {
            int prev_base = s * T;
            bool prev_written = written[prev_base] && !(write_step && prev_base == base);
            if (prev_written) ++prefix_slots;
        }
        out_base = prefix_slots * T;
        if (slot == slots - 1) {
            *count = (prefix_slots + (slot_written ? 1 : 0)) * T;
        }
    }
    __syncthreads();

    if (slot_written) {
        for (int t = tid; t < T; t += blockDim.x) {
            indices[out_base + t] = (int64_t)(slot_base + t);
        }
        constexpr int sparse_block_size = 128;
        int blocks_per_frame = T / sparse_block_size;
        for (int block = tid; block < blocks_per_frame; block += blockDim.x) {
            block_ids[out_base / sparse_block_size + block] =
                slot_base / sparse_block_size + block;
        }
    }
}

__global__ static void gather_indexed_kv_d64_hq_f16_kernel(
        const __half *__restrict__ cache_k,
        const __half *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        int Nkv,
        __half *__restrict__ k_compact,
        __half *__restrict__ v_compact,
        int Hq,
        int Hkv,
        int Tk) {
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)Hq * Nkv * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t q = i >> 6;
    int n = (int)(q % Nkv);
    int hq = (int)(q / Nkv);
    int group = Hq / Hkv;
    int hk = hq / group;
    int tk = (int)indices[n];
    if (tk < 0) tk = 0;
    if (tk >= Tk) tk = Tk - 1;
    int64_t src = ((int64_t)hk * Tk + tk) * 64 + d;
    k_compact[i] = cache_k[src];
    v_compact[i] = cache_v[src];
}

__global__ static void q_d64_htd_f32_to_gqa_bmhd_f16_kernel(
        const float *__restrict__ q,
        __half *__restrict__ q_bmhd,
        int Hq,
        int Hkv,
        int Tq) {
    int group = Hq / Hkv;
    int M = group * Tq;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)M * Hkv * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t x = i >> 6;
    int hk = (int)(x % Hkv);
    int m = (int)(x / Hkv);
    int g = m / Tq;
    int t = m - g * Tq;
    int hq = hk * group + g;
    q_bmhd[i] = __float2half_rn(q[((int64_t)hq * Tq + t) * 64 + d]);
}

__global__ static void gather_indexed_kv_d64_bnhd_hkv_f16_kernel(
        const __half *__restrict__ cache_k,
        const __half *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        int Nkv,
        __half *__restrict__ k_bnhd,
        __half *__restrict__ v_bnhd,
        int Hkv,
        int Tk) {
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)Nkv * Hkv * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t x = i >> 6;
    int hk = (int)(x % Hkv);
    int n = (int)(x / Hkv);
    int tk = (int)indices[n];
    if (tk < 0) tk = 0;
    if (tk >= Tk) tk = Tk - 1;
    int64_t src = ((int64_t)hk * Tk + tk) * 64 + d;
    k_bnhd[i] = cache_k[src];
    v_bnhd[i] = cache_v[src];
}

__global__ static void scatter_gqa_bmhd_f16_to_tokens_f32_kernel(
        const __half *__restrict__ group_out,
        float *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq) {
    int group = Hq / Hkv;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)Tq * Hq * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t x = i >> 6;
    int hq = (int)(x % Hq);
    int t = (int)(x / Hq);
    int hk = hq / group;
    int g = hq - hk * group;
    int m = g * Tq + t;
    int64_t src = ((int64_t)m * Hkv + hk) * 64 + d;
    out_tokens[i] = __half2float(group_out[src]);
}

__global__ static void scatter_gqa_bmhd_f16_to_tokens_f16_kernel(
        const __half *__restrict__ group_out,
        __half *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq) {
    int group = Hq / Hkv;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)Tq * Hq * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t x = i >> 6;
    int hq = (int)(x % Hq);
    int t = (int)(x / Hq);
    int hk = hq / group;
    int g = hq - hk * group;
    int m = g * Tq + t;
    int64_t src = ((int64_t)m * Hkv + hk) * 64 + d;
    out_tokens[i] = group_out[src];
}

__global__ static void gather_indexed_kv_d64_hkv_f16_kernel(
        const __half *__restrict__ cache_k,
        const __half *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        int Nkv,
        __half *__restrict__ k_compact,
        __half *__restrict__ v_compact,
        int Hkv,
        int Tk) {
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)Hkv * Nkv * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t q = i >> 6;
    int n = (int)(q % Nkv);
    int hk = (int)(q / Nkv);
    int tk = (int)indices[n];
    if (tk < 0) tk = 0;
    if (tk >= Tk) tk = Tk - 1;
    int64_t src = ((int64_t)hk * Tk + tk) * 64 + d;
    k_compact[i] = cache_k[src];
    v_compact[i] = cache_v[src];
}

__global__ static void scatter_grouped_attn_d64_tokens_f32_kernel(
        const float *__restrict__ group_out,
        float *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)Hq * Tq * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t q = i >> 6;
    int t = (int)(q % Tq);
    int hq = (int)(q / Tq);
    int group = Hq / Hkv;
    int hk = hq / group;
    int g = hq - hk * group;
    int64_t src = ((int64_t)hk * (group * Tq) + g * Tq + t) * 64 + d;
    int64_t dst = ((int64_t)t * Hq + hq) * 64 + d;
    out_tokens[dst] = group_out[src];
}

__global__ static void softmax_rows_inplace_f32_kernel(float *x, int rows, int cols) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    float *row_x = x + (int64_t)row * cols;

    float mx = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        mx = fmaxf(mx, row_x[c]);
    }
    red[tid] = mx;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    mx = red[0];

    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = expf(row_x[c] - mx);
        row_x[c] = v;
        sum += v;
    }
    red[tid] = sum;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        row_x[c] *= inv;
    }
}

__global__ static void indexed_attention_cache_f32_kernel(
        const float *q,
        const float *cache_k,
        const float *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        int D,
        float scale) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int tq = row % Tq;
    int hq = row / Tq;
    int group = Hq / Hkv;
    int hk = hq / group;
    int Nkv = *index_count;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const float *qrow = q + (((int64_t)hq * Tq + tq) * D);
    const float *kbase = cache_k + (int64_t)hk * Tk * D;
    const float *vbase = cache_v + (int64_t)hk * Tk * D;
    float *orow = out_tokens + (int64_t)tq * (Hq * D) + hq * D;

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n = 0; n < Nkv; ++n) {
        int tk = (int)indices[n];
        float partial = 0.0f;
        const float *krow = kbase + (int64_t)tk * D;
        for (int d = tid; d < D; d += blockDim.x) {
            partial += qrow[d] * krow[d];
        }

        red[tid] = partial;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }

        float score = red[0] * scale;
        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);

        if (tid < D) {
            acc = acc * alpha + beta * vbase[(int64_t)tk * D + tid];
        }
        l = l * alpha + beta;
        m = new_m;
        __syncthreads();
    }

    if (tid < D) {
        orow[tid] = Nkv > 0 ? acc / l : 0.0f;
    }
}

__global__ static void indexed_attention_cache_d64_warp_f32_kernel(
        const float *q,
        const float *cache_k,
        const float *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    int warp_row = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_rows = Hq * Tq;
    if (warp_row >= total_rows) return;

    int tq = warp_row % Tq;
    int hq = warp_row / Tq;
    int group = Hq / Hkv;
    int hk = hq / group;
    int Nkv = *index_count;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const float *qrow = q + (((int64_t)hq * Tq + tq) * 64);
    const float *kbase = cache_k + (int64_t)hk * Tk * 64;
    const float *vbase = cache_v + (int64_t)hk * Tk * 64;
    float *orow = out_tokens + (int64_t)tq * (Hq * 64) + hq * 64;

    float q0 = qrow[lane];
    float q1 = qrow[lane + 32];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n = 0; n < Nkv; ++n) {
        int tk = (int)indices[n];
        const float *krow = kbase + (int64_t)tk * 64;
        float dot = wm_warp_sum(q0 * krow[lane] + q1 * krow[lane + 32]);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        float score = dot * scale;
        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        const float *vrow = vbase + (int64_t)tk * 64;
        acc0 = acc0 * alpha + beta * vrow[lane];
        acc1 = acc1 * alpha + beta * vrow[lane + 32];
        l = l * alpha + beta;
        m = new_m;
    }

    if (Nkv > 0) {
        float inv_l = 1.0f / l;
        orow[lane] = acc0 * inv_l;
        orow[lane + 32] = acc1 * inv_l;
    } else {
        orow[lane] = 0.0f;
        orow[lane + 32] = 0.0f;
    }
}

__global__ static void indexed_attention_cache_d64_warp_f16_kv_kernel(
        const float *__restrict__ q,
        const __half *__restrict__ cache_k,
        const __half *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        const int *__restrict__ index_count,
        float *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    int warp_row = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_rows = Hq * Tq;
    if (warp_row >= total_rows) return;

    int tq = warp_row % Tq;
    int hq = warp_row / Tq;
    int group = Hq / Hkv;
    int hk = hq / group;
    int Nkv = *index_count;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const float *qrow = q + (((int64_t)hq * Tq + tq) * 64);
    const __half *kbase = cache_k + (int64_t)hk * Tk * 64;
    const __half *vbase = cache_v + (int64_t)hk * Tk * 64;
    float *orow = out_tokens + (int64_t)tq * (Hq * 64) + hq * 64;

    int d0 = lane << 1;
    float q0 = qrow[d0];
    float q1 = qrow[d0 + 1];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n = 0; n < Nkv; ++n) {
        int tk = (int)indices[n];
        const __half2 *krow = (const __half2 *)(kbase + (int64_t)tk * 64);
        float2 k2 = __half22float2(krow[lane]);
        float dot = wm_warp_sum(q0 * k2.x + q1 * k2.y);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        float score = dot * scale;
        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        const __half2 *vrow = (const __half2 *)(vbase + (int64_t)tk * 64);
        float2 v2 = __half22float2(vrow[lane]);
        acc0 = acc0 * alpha + beta * v2.x;
        acc1 = acc1 * alpha + beta * v2.y;
        l = l * alpha + beta;
        m = new_m;
    }

    if (Nkv > 0) {
        float inv_l = 1.0f / l;
        orow[d0] = acc0 * inv_l;
        orow[d0 + 1] = acc1 * inv_l;
    } else {
        orow[d0] = 0.0f;
        orow[d0 + 1] = 0.0f;
    }
}

__global__ static void indexed_attention_cache_d64_flash_f16_kv_kernel(
        const float *__restrict__ q,
        const __half *__restrict__ cache_k,
        const __half *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        const int *__restrict__ index_count,
        float *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    extern __shared__ __half smem_h[];
    __half *sh_k = smem_h;
    __half *sh_v = sh_k + WORLD_ATTN_D64_K_BLOCK * 64;

    int group = Hq / Hkv;
    int q_per_h = WORLD_ATTN_D64_FLASH_WARPS / group;
    if (q_per_h < 1) q_per_h = 1;
    int q_blocks = (Tq + q_per_h - 1) / q_per_h;
    int q_block = blockIdx.x % q_blocks;
    int hk = blockIdx.x / q_blocks;
    if (hk >= Hkv) return;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int local_h = warp / q_per_h;
    int tq = q_block * q_per_h + (warp - local_h * q_per_h);
    bool valid_q = local_h < group && tq < Tq;
    int hq = hk * group + local_h;

    int Nkv = *index_count;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const __half *kbase = cache_k + (int64_t)hk * Tk * 64;
    const __half *vbase = cache_v + (int64_t)hk * Tk * 64;
    const float *qrow = valid_q ? q + ((int64_t)hq * Tq + tq) * 64 : q;
    float *orow = valid_q ? out_tokens + (int64_t)tq * (Hq * 64) + hq * 64 : out_tokens;

    int d0 = lane << 1;
    float q0 = valid_q ? qrow[d0] : 0.0f;
    float q1 = valid_q ? qrow[d0 + 1] : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n0 = 0; n0 < Nkv; n0 += WORLD_ATTN_D64_K_BLOCK) {
        int active = Nkv - n0;
        if (active > WORLD_ATTN_D64_K_BLOCK) active = WORLD_ATTN_D64_K_BLOCK;
        int tile_elems = active * 64;
        for (int i = tid; i < tile_elems; i += blockDim.x) {
            int n = i >> 6;
            int d = i & 63;
            int tk = (int)indices[n0 + n];
            sh_k[i] = kbase[(int64_t)tk * 64 + d];
            sh_v[i] = vbase[(int64_t)tk * 64 + d];
        }
        __syncthreads();

        if (valid_q) {
            for (int n = 0; n < active; ++n) {
                const __half2 *krow = (const __half2 *)(sh_k + n * 64);
                float2 k2 = __half22float2(krow[lane]);
                float dot = wm_warp_sum(q0 * k2.x + q1 * k2.y);
                dot = __shfl_sync(0xffffffffu, dot, 0);
                float score = dot * scale;
                float new_m = fmaxf(m, score);
                float alpha = expf(m - new_m);
                float beta = expf(score - new_m);
                const __half2 *vrow = (const __half2 *)(sh_v + n * 64);
                float2 v2 = __half22float2(vrow[lane]);
                acc0 = acc0 * alpha + beta * v2.x;
                acc1 = acc1 * alpha + beta * v2.y;
                l = l * alpha + beta;
                m = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        if (Nkv > 0) {
            float inv_l = 1.0f / l;
            orow[d0] = acc0 * inv_l;
            orow[d0 + 1] = acc1 * inv_l;
        } else {
            orow[d0] = 0.0f;
            orow[d0 + 1] = 0.0f;
        }
    }
}

__global__ static void indexed_attention_cache_d64_flash_f32_kernel(
        const float *__restrict__ q,
        const float *__restrict__ cache_k,
        const float *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        const int *__restrict__ index_count,
        float *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    extern __shared__ float smem[];
    float *sh_k = smem;
    float *sh_v = sh_k + WORLD_ATTN_D64_K_BLOCK * 64;

    int group = Hq / Hkv;
    int q_per_h = WORLD_ATTN_D64_FLASH_WARPS / group;
    if (q_per_h < 1) q_per_h = 1;
    int q_blocks = (Tq + q_per_h - 1) / q_per_h;
    int q_block = blockIdx.x % q_blocks;
    int hk = blockIdx.x / q_blocks;
    if (hk >= Hkv) return;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int local_h = warp / q_per_h;
    int tq = q_block * q_per_h + (warp - local_h * q_per_h);
    bool valid_q = local_h < group && tq < Tq;
    int hq = hk * group + local_h;

    int Nkv = *index_count;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const float *kbase = cache_k + (int64_t)hk * Tk * 64;
    const float *vbase = cache_v + (int64_t)hk * Tk * 64;
    const float *qrow = valid_q ? q + ((int64_t)hq * Tq + tq) * 64 : q;
    float *orow = valid_q ? out_tokens + (int64_t)tq * (Hq * 64) + hq * 64 : out_tokens;

    float q0 = valid_q ? qrow[lane] : 0.0f;
    float q1 = valid_q ? qrow[lane + 32] : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n0 = 0; n0 < Nkv; n0 += WORLD_ATTN_D64_K_BLOCK) {
        int active = Nkv - n0;
        if (active > WORLD_ATTN_D64_K_BLOCK) active = WORLD_ATTN_D64_K_BLOCK;
        int tile_elems = active * 64;
        for (int i = tid; i < tile_elems; i += blockDim.x) {
            int n = i >> 6;
            int d = i & 63;
            int tk = (int)indices[n0 + n];
            sh_k[i] = kbase[(int64_t)tk * 64 + d];
            sh_v[i] = vbase[(int64_t)tk * 64 + d];
        }
        __syncthreads();

        if (valid_q) {
            for (int n = 0; n < active; ++n) {
                const float *krow = sh_k + n * 64;
                float dot = wm_warp_sum(q0 * krow[lane] + q1 * krow[lane + 32]);
                dot = __shfl_sync(0xffffffffu, dot, 0);
                float score = dot * scale;
                float new_m = fmaxf(m, score);
                float alpha = expf(m - new_m);
                float beta = expf(score - new_m);
                const float *vrow = sh_v + n * 64;
                acc0 = acc0 * alpha + beta * vrow[lane];
                acc1 = acc1 * alpha + beta * vrow[lane + 32];
                l = l * alpha + beta;
                m = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        if (Nkv > 0) {
            float inv_l = 1.0f / l;
            orow[lane] = acc0 * inv_l;
            orow[lane + 32] = acc1 * inv_l;
        } else {
            orow[lane] = 0.0f;
            orow[lane + 32] = 0.0f;
        }
    }
}

__global__ static void indexed_attention_cache_d64_q4_shared_f32_kernel(
        const float *__restrict__ q,
        const float *__restrict__ cache_k,
        const float *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        const int *__restrict__ index_count,
        float *__restrict__ out_tokens,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    extern __shared__ float smem[];
    float *sh_k = smem;
    float *sh_v = sh_k + WORLD_ATTN_D64_K_BLOCK * 64;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int q_blocks = (Tq + WORLD_ATTN_D64_Q_BLOCK - 1) / WORLD_ATTN_D64_Q_BLOCK;
    int q_block = blockIdx.x % q_blocks;
    int hq = blockIdx.x / q_blocks;
    if (hq >= Hq) return;

    int tq0 = q_block * WORLD_ATTN_D64_Q_BLOCK;
    int tq = tq0 + warp;
    bool valid_q = tq < Tq;

    int group = Hq / Hkv;
    int hk = hq / group;
    int Nkv = *index_count;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const float *kbase = cache_k + (int64_t)hk * Tk * 64;
    const float *vbase = cache_v + (int64_t)hk * Tk * 64;
    const float *qrow = valid_q ? q + ((int64_t)hq * Tq + tq) * 64 : q;
    float *orow = valid_q ? out_tokens + (int64_t)tq * (Hq * 64) + hq * 64 : out_tokens;

    float q0 = valid_q ? qrow[lane] : 0.0f;
    float q1 = valid_q ? qrow[lane + 32] : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n0 = 0; n0 < Nkv; n0 += WORLD_ATTN_D64_K_BLOCK) {
        int active = Nkv - n0;
        if (active > WORLD_ATTN_D64_K_BLOCK) active = WORLD_ATTN_D64_K_BLOCK;
        int tile_elems = active * 64;
        for (int i = tid; i < tile_elems; i += blockDim.x) {
            int n = i >> 6;
            int d = i & 63;
            int tk = (int)indices[n0 + n];
            sh_k[i] = kbase[(int64_t)tk * 64 + d];
            sh_v[i] = vbase[(int64_t)tk * 64 + d];
        }
        __syncthreads();

        if (valid_q) {
            for (int n = 0; n < active; ++n) {
                const float *krow = sh_k + n * 64;
                float dot = wm_warp_sum(q0 * krow[lane] + q1 * krow[lane + 32]);
                dot = __shfl_sync(0xffffffffu, dot, 0);
                float score = dot * scale;
                float new_m = fmaxf(m, score);
                float alpha = expf(m - new_m);
                float beta = expf(score - new_m);
                const float *vrow = sh_v + n * 64;
                acc0 = acc0 * alpha + beta * vrow[lane];
                acc1 = acc1 * alpha + beta * vrow[lane + 32];
                l = l * alpha + beta;
                m = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        if (Nkv > 0) {
            float inv_l = 1.0f / l;
            orow[lane] = acc0 * inv_l;
            orow[lane + 32] = acc1 * inv_l;
        } else {
            orow[lane] = 0.0f;
            orow[lane + 32] = 0.0f;
        }
    }
}

__global__ static void gated_residual_add_f32_kernel(
        const float *residual,
        const float *update,
        const float *gate,
        float *out,
        int T,
        int D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)T * D;
    if (i >= total) return;
    int d = (int)(i % D);
    out[i] = residual[i] + update[i] * gate[d];
}

__global__ static void add_f32_kernel(const float *a, const float *b, float *out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ static void latent_update_f32_kernel(float *latent, const float *velocity, float dsigma, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) latent[i] += dsigma * velocity[i];
}

__global__ static void lerp_inplace_f32_kernel(float *x, const float *end, float weight, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + weight * (end[i] - x[i]);
}

__global__ static void patchify_f32_kernel(
        const float *x,
        const float *weight,
        float *tokens,
        int C,
        int H,
        int W,
        int D,
        int ph,
        int pw,
        int Hp,
        int Wp) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int d = row % D;
    int token = (row / D) % (Hp * Wp);
    int oy = token / Wp;
    int ox = token - oy * Wp;
    int patch_elems = C * ph * pw;

    float sum = 0.0f;
    for (int i = tid; i < patch_elems; i += blockDim.x) {
        int p = i;
        int dx = p % pw;
        p /= pw;
        int dy = p % ph;
        int c = p / ph;
        int iy = oy * ph + dy;
        int ix = ox * pw + dx;
        float xv = x[(c * H + iy) * W + ix];
        float wv = weight[(((int64_t)d * C + c) * ph + dy) * pw + dx];
        sum += xv * wv;
    }

    red[tid] = sum;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    if (tid == 0) tokens[token * D + d] = red[0];
}

__global__ static void patchify_im2row_f32_kernel(
        const float *x,
        float *rows,
        int C,
        int H,
        int W,
        int ph,
        int pw,
        int Hp,
        int Wp) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int patch_elems = C * ph * pw;
    int64_t total = (int64_t)Hp * Wp * patch_elems;
    if (i >= total) return;
    int p = (int)(i % patch_elems);
    int token = (int)(i / patch_elems);
    int ox = token % Wp;
    int oy = token / Wp;
    int q = p;
    int dx = q % pw;
    q /= pw;
    int dy = q % ph;
    int c = q / ph;
    int iy = oy * ph + dy;
    int ix = ox * pw + dx;
    rows[i] = x[(c * H + iy) * W + ix];
}

__global__ static void unpatchify_orig_f32_kernel(
        const float *tokens,
        const float *weight,
        const float *bias,
        float *x,
        int T,
        int D,
        int C,
        int H,
        int W,
        int ph,
        int pw,
        int Wp,
        int out_dim) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int o = row % out_dim;
    int token = row / out_dim;

    int p = o;
    int dx = p % pw;
    p /= pw;
    int dy = p % ph;
    p /= ph;
    int c = p;

    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float wv = weight[(((int64_t)d * C + c) * ph + dy) * pw + dx];
        sum += tokens[(int64_t)token * D + d] * wv;
    }

    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }

    if (tid == 0) {
        int oy = token / Wp;
        int ox = token - oy * Wp;
        int iy = oy * ph + dy;
        int ix = ox * pw + dx;
        x[(c * H + iy) * W + ix] = red[0] + bias[c];
    }
}

int wm_cuda_attention_d64_cutlass_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        int Nkv,
        float *out_tokens,
        __half *q_h,
        __half *k_compact,
        __half *v_compact,
        float *scores,
        __half *probs_h,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;
    if (Nkv <= 0) {
        CUDA_OK(cudaMemset(out_tokens, 0, (size_t)Tq * Hq * 64 * sizeof(float)));
        return 0;
    }

    int64_t q_elems = (int64_t)Hq * Tq * 64;
    f32_to_f16_kernel<<<div_up_i64(q_elems, 256), 256>>>(q, q_h, q_elems);
    CUDA_OK(cudaGetLastError());

    int64_t kv_compact_elems = (int64_t)Hq * Nkv * 64;
    gather_indexed_kv_d64_hq_f16_kernel<<<div_up_i64(kv_compact_elems, 256), 256>>>(
        cache_k, cache_v, indices, Nkv, k_compact, v_compact, Hq, Hkv, Tk);
    CUDA_OK(cudaGetLastError());

    using GemmQK = cutlass::gemm::device::GemmBatched<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4,
        8,
        8>;

    const cutlass::half_t *q_ptr = reinterpret_cast<const cutlass::half_t *>(q_h);
    const cutlass::half_t *k_ptr = reinterpret_cast<const cutlass::half_t *>(k_compact);
    typename GemmQK::Arguments qk_args(
        {Tq, Nkv, 64},
        {q_ptr, 64},
        (int64_t)Tq * 64,
        {k_ptr, 64},
        (int64_t)Nkv * 64,
        {scores, Nkv},
        (int64_t)Tq * Nkv,
        {scores, Nkv},
        (int64_t)Tq * Nkv,
        {scale, 0.0f},
        Hq);
    GemmQK qk_gemm;
    CUTLASS_OK(qk_gemm.can_implement(qk_args));
    CUTLASS_OK(qk_gemm(qk_args));
    CUDA_OK(cudaGetLastError());

    softmax_rows_inplace_f32_kernel<<<Hq * Tq, 256>>>(scores, Hq * Tq, Nkv);
    CUDA_OK(cudaGetLastError());

    int64_t prob_elems = (int64_t)Hq * Tq * Nkv;
    f32_to_f16_kernel<<<div_up_i64(prob_elems, 256), 256>>>(scores, probs_h, prob_elems);
    CUDA_OK(cudaGetLastError());

    using GemmAV = cutlass::gemm::device::GemmBatched<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4,
        8,
        8>;

    const cutlass::half_t *probs_ptr = reinterpret_cast<const cutlass::half_t *>(probs_h);
    const cutlass::half_t *v_ptr = reinterpret_cast<const cutlass::half_t *>(v_compact);
    typename GemmAV::Arguments av_args(
        {Tq, 64, Nkv},
        {probs_ptr, Nkv},
        (int64_t)Tq * Nkv,
        {v_ptr, 64},
        (int64_t)Nkv * 64,
        {out_tokens, Hq * 64},
        64,
        {out_tokens, Hq * 64},
        64,
        {1.0f, 0.0f},
        Hq);
    GemmAV av_gemm;
    CUTLASS_OK(av_gemm.can_implement(av_args));
    CUTLASS_OK(av_gemm(av_args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_attention_d64_fmha_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        int Nkv,
        float *out_tokens,
        __half *out_tokens_h,
        int output_half,
        __half *q_bmhd,
        __half *k_bnhd,
        __half *v_bnhd,
        __half *out_bmhd,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
#if WORLD_HAS_CUTLASS_FMHA
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;
    int64_t out_elems = (int64_t)Tq * Hq * 64;
    if (Nkv <= 0) {
        if (output_half) {
            CUDA_OK(cudaMemset(out_tokens_h, 0, (size_t)out_elems * sizeof(__half)));
        } else {
            CUDA_OK(cudaMemset(out_tokens, 0, (size_t)out_elems * sizeof(float)));
        }
        return 0;
    }
    if (Hkv <= 0 || Hq <= 0 || (Hq % Hkv) != 0) {
        fprintf(stderr, "CUTLASS FMHA GQA bridge requires Hq divisible by Hkv\n");
        return 1;
    }
    int group = Hq / Hkv;
    int M = group * Tq;

    q_d64_htd_f32_to_gqa_bmhd_f16_kernel<<<div_up_i64(out_elems, 256), 256>>>(q, q_bmhd, Hq, Hkv, Tq);
    CUDA_OK(cudaGetLastError());

    int64_t kv_compact_elems = (int64_t)Nkv * Hkv * 64;
    gather_indexed_kv_d64_bnhd_hkv_f16_kernel<<<div_up_i64(kv_compact_elems, 256), 256>>>(
        cache_k, cache_v, indices, Nkv, k_bnhd, v_bnhd, Hkv, Tk);
    CUDA_OK(cudaGetLastError());

    using Attention = AttentionKernel<
        cutlass::half_t,
        cutlass::arch::Sm80,
        true,
        64,
        64,
        64,
        false,
        false>;

    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t *>(q_bmhd);
    p.key_ptr = reinterpret_cast<cutlass::half_t *>(k_bnhd);
    p.value_ptr = reinterpret_cast<cutlass::half_t *>(v_bnhd);
    p.output_ptr = reinterpret_cast<cutlass::half_t *>(out_bmhd);
    p.output_accum_ptr = NULL;
    p.logsumexp_ptr = NULL;
    p.scale = scale;
    p.num_heads = Hkv;
    p.num_batches = 1;
    p.head_dim = 64;
    p.head_dim_value = 64;
    p.num_queries = M;
    p.num_keys = Nkv;
    p.custom_mask_type = Attention::NoCustomMask;
    p.q_strideH = 64;
    p.k_strideH = 64;
    p.v_strideH = 64;
    p.q_strideM = Hkv * 64;
    p.k_strideM = Hkv * 64;
    p.v_strideM = Hkv * 64;
    p.q_strideB = (int64_t)M * Hkv * 64;
    p.k_strideB = (int64_t)Nkv * Hkv * 64;
    p.v_strideB = (int64_t)Nkv * Hkv * 64;
    p.o_strideM = Hkv * 64;

    if (!Attention::check_supported(p)) {
        fprintf(stderr, "CUTLASS FMHA does not support this runtime attention shape\n");
        return 1;
    }
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        CUDA_OK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    CUDA_OK(cudaGetLastError());

    if (output_half) {
        scatter_gqa_bmhd_f16_to_tokens_f16_kernel<<<div_up_i64(out_elems, 256), 256>>>(
            out_bmhd, out_tokens_h, Hq, Hkv, Tq);
    } else {
        scatter_gqa_bmhd_f16_to_tokens_f32_kernel<<<div_up_i64(out_elems, 256), 256>>>(
            out_bmhd, out_tokens, Hq, Hkv, Tq);
    }
    CUDA_OK(cudaGetLastError());
    return 0;
#else
    (void)q;
    (void)cache_k;
    (void)cache_v;
    (void)indices;
    (void)Nkv;
    (void)out_tokens;
    (void)out_tokens_h;
    (void)output_half;
    (void)q_bmhd;
    (void)k_bnhd;
    (void)v_bnhd;
    (void)out_bmhd;
    (void)Hq;
    (void)Hkv;
    (void)Tq;
    (void)Tk;
    (void)scale;
    fprintf(stderr, "CUTLASS FMHA headers are not available in this build\n");
    return 1;
#endif
}

int wm_cuda_attention_d64_sparse_fmha_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int32_t *block_ids,
        int block_count,
        float *out_tokens,
        __half *out_tokens_h,
        int output_half,
        __half *q_bmhd,
        __half *out_bmhd,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
#if WORLD_HAS_CUTLASS_FMHA
    constexpr int kSparseBlockSize = 128;
    int64_t out_elems = (int64_t)Tq * Hq * 64;
    if (block_count <= 0) {
        if (output_half) {
            CUDA_OK(cudaMemset(out_tokens_h, 0, (size_t)out_elems * sizeof(__half)));
        } else {
            CUDA_OK(cudaMemset(out_tokens, 0, (size_t)out_elems * sizeof(float)));
        }
        return 0;
    }
    if (Tk % kSparseBlockSize != 0 || block_count > Tk / kSparseBlockSize) {
        fprintf(stderr, "Sparse CUTLASS FMHA requires valid 128-token KV blocks\n");
        return 1;
    }
    if (Hkv <= 0 || Hq <= 0 || (Hq % Hkv) != 0) {
        fprintf(stderr, "Sparse CUTLASS FMHA requires Hq divisible by Hkv\n");
        return 1;
    }

    int group = Hq / Hkv;
    int M = group * Tq;
    int Nkv = block_count * kSparseBlockSize;
    q_d64_htd_f32_to_gqa_bmhd_f16_kernel<<<div_up_i64(out_elems, 256), 256>>>(
        q, q_bmhd, Hq, Hkv, Tq);
    CUDA_OK(cudaGetLastError());

    using Attention = world_sparse_fmha::SparseAttentionKernel<
        cutlass::half_t,
        cutlass::arch::Sm80,
        true,
        64,
        64,
        64,
        false,
        false>;

    typename Attention::Params p;
    p.query_ptr = reinterpret_cast<cutlass::half_t *>(q_bmhd);
    p.key_ptr = reinterpret_cast<cutlass::half_t *>(const_cast<__half *>(cache_k));
    p.value_ptr = reinterpret_cast<cutlass::half_t *>(const_cast<__half *>(cache_v));
    p.sparse_block_ids = block_ids;
    p.sparse_block_count = block_count;
    p.sparse_block_size = kSparseBlockSize;
    p.sparse_block_strideB = 0;
    p.output_ptr = reinterpret_cast<cutlass::half_t *>(out_bmhd);
    p.output_accum_ptr = NULL;
    p.logsumexp_ptr = NULL;
    p.scale = scale;
    p.num_heads = Hkv;
    p.num_batches = 1;
    p.head_dim = 64;
    p.head_dim_value = 64;
    p.num_queries = M;
    p.num_keys = Nkv;
    p.custom_mask_type = Attention::NoCustomMask;
    p.q_strideH = 64;
    p.k_strideH = Tk * 64;
    p.v_strideH = Tk * 64;
    p.q_strideM = Hkv * 64;
    p.k_strideM = 64;
    p.v_strideM = 64;
    p.q_strideB = (int64_t)M * Hkv * 64;
    p.k_strideB = (int64_t)Hkv * Tk * 64;
    p.v_strideB = (int64_t)Hkv * Tk * 64;
    p.o_strideM = Hkv * 64;

    if (!Attention::check_supported(p)) {
        fprintf(stderr, "Sparse CUTLASS FMHA does not support this runtime attention shape\n");
        return 1;
    }
    constexpr auto kernel_fn = world_sparse_fmha::sparse_attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        CUDA_OK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    CUDA_OK(cudaGetLastError());

    if (output_half) {
        scatter_gqa_bmhd_f16_to_tokens_f16_kernel<<<div_up_i64(out_elems, 256), 256>>>(
            out_bmhd, out_tokens_h, Hq, Hkv, Tq);
    } else {
        scatter_gqa_bmhd_f16_to_tokens_f32_kernel<<<div_up_i64(out_elems, 256), 256>>>(
            out_bmhd, out_tokens, Hq, Hkv, Tq);
    }
    CUDA_OK(cudaGetLastError());
    return 0;
#else
    (void)q;
    (void)cache_k;
    (void)cache_v;
    (void)block_ids;
    (void)block_count;
    (void)out_tokens;
    (void)out_tokens_h;
    (void)output_half;
    (void)q_bmhd;
    (void)out_bmhd;
    (void)Hq;
    (void)Hkv;
    (void)Tq;
    (void)Tk;
    (void)scale;
    fprintf(stderr, "CUTLASS FMHA headers are not available in this build\n");
    return 1;
#endif
}

int wm_cuda_attention_d64_cutlass_grouped_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        int Nkv,
        float *out_tokens,
        __half *q_h,
        __half *k_compact,
        __half *v_compact,
        float *scores,
        __half *probs_h,
        int Hq,
        int Hkv,
        int Tq,
        int Tk,
        float scale) {
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;
    if (Nkv <= 0) {
        CUDA_OK(cudaMemset(out_tokens, 0, (size_t)Tq * Hq * 64 * sizeof(float)));
        return 0;
    }
    if (Hkv <= 0 || (Hq % Hkv) != 0 || (Hq / Hkv) <= 1 || Nkv < 64) {
        return wm_cuda_attention_d64_cutlass_f16_kv(
            q, cache_k, cache_v, indices, Nkv, out_tokens,
            q_h, k_compact, v_compact, scores, probs_h,
            Hq, Hkv, Tq, Tk, scale);
    }

    int group = Hq / Hkv;
    int grouped_rows = group * Tq;

    int64_t q_elems = (int64_t)Hq * Tq * 64;
    f32_to_f16_kernel<<<div_up_i64(q_elems, 256), 256>>>(q, q_h, q_elems);
    CUDA_OK(cudaGetLastError());

    int64_t kv_compact_elems = (int64_t)Hkv * Nkv * 64;
    gather_indexed_kv_d64_hkv_f16_kernel<<<div_up_i64(kv_compact_elems, 256), 256>>>(
        cache_k, cache_v, indices, Nkv, k_compact, v_compact, Hkv, Tk);
    CUDA_OK(cudaGetLastError());

    using GemmQK = cutlass::gemm::device::GemmBatched<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4,
        8,
        8>;

    const cutlass::half_t *q_ptr = reinterpret_cast<const cutlass::half_t *>(q_h);
    const cutlass::half_t *k_ptr = reinterpret_cast<const cutlass::half_t *>(k_compact);
    typename GemmQK::Arguments qk_args(
        {grouped_rows, Nkv, 64},
        {q_ptr, 64},
        (int64_t)grouped_rows * 64,
        {k_ptr, 64},
        (int64_t)Nkv * 64,
        {scores, Nkv},
        (int64_t)grouped_rows * Nkv,
        {scores, Nkv},
        (int64_t)grouped_rows * Nkv,
        {scale, 0.0f},
        Hkv);
    GemmQK qk_gemm;
    CUTLASS_OK(qk_gemm.can_implement(qk_args));
    CUTLASS_OK(qk_gemm(qk_args));
    CUDA_OK(cudaGetLastError());

    softmax_rows_inplace_f32_kernel<<<Hq * Tq, 256>>>(scores, Hq * Tq, Nkv);
    CUDA_OK(cudaGetLastError());

    int64_t prob_elems = (int64_t)Hq * Tq * Nkv;
    f32_to_f16_kernel<<<div_up_i64(prob_elems, 256), 256>>>(scores, probs_h, prob_elems);
    CUDA_OK(cudaGetLastError());

    using GemmAV = cutlass::gemm::device::GemmBatched<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            float,
            128 / cutlass::sizeof_bits<float>::value,
            float,
            float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4,
        8,
        8>;

    const cutlass::half_t *probs_ptr = reinterpret_cast<const cutlass::half_t *>(probs_h);
    const cutlass::half_t *v_ptr = reinterpret_cast<const cutlass::half_t *>(v_compact);
    float *group_out = scores;
    typename GemmAV::Arguments av_args(
        {grouped_rows, 64, Nkv},
        {probs_ptr, Nkv},
        (int64_t)grouped_rows * Nkv,
        {v_ptr, 64},
        (int64_t)Nkv * 64,
        {group_out, 64},
        (int64_t)grouped_rows * 64,
        {group_out, 64},
        (int64_t)grouped_rows * 64,
        {1.0f, 0.0f},
        Hkv);
    GemmAV av_gemm;
    CUTLASS_OK(av_gemm.can_implement(av_args));
    CUTLASS_OK(av_gemm(av_args));
    CUDA_OK(cudaGetLastError());

    scatter_grouped_attn_d64_tokens_f32_kernel<<<div_up_i64(q_elems, 256), 256>>>(
        group_out, out_tokens, Hq, Hkv, Tq);
    CUDA_OK(cudaGetLastError());
    return 0;
}


static int wm_cuda_check_launch(void) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int wm_cuda_silu_f32(const float *x, float *y, int64_t n) {
    silu_f32_kernel<<<div_up_i64(n, 256), 256>>>(x, y, n);
    return wm_cuda_check_launch();
}

int wm_cuda_f32_to_f16(const float *x, __half *y, int64_t n) {
    f32_to_f16_kernel<<<div_up_i64(n, 256), 256>>>(x, y, n);
    return wm_cuda_check_launch();
}

int wm_cuda_silu_f32_to_f16(const float *x, __half *y, int64_t n) {
    silu_f32_to_f16_kernel<<<div_up_i64(n, 256), 256>>>(x, y, n);
    return wm_cuda_check_launch();
}

int wm_cuda_add_bias_silu_f32(const float *x, const float *bias, float *y, int64_t n) {
    add_bias_silu_f32_kernel<<<div_up_i64(n, 256), 256>>>(x, bias, y, n);
    return wm_cuda_check_launch();
}

int wm_cuda_add_channel_silu_inplace_f32(float *x, const float *bias, int rows, int d) {
    add_channel_silu_inplace_f32_kernel<<<div_up_i64((int64_t)rows * d, 256), 256>>>(x, bias, rows, d);
    return wm_cuda_check_launch();
}

int wm_cuda_quantize_rows_f32_i8(const float *x, int8_t *q, float *scales, int rows, int cols) {
    quantize_rows_f32_i8_kernel<<<rows, 256>>>(x, q, scales, rows, cols);
    return wm_cuda_check_launch();
}

int wm_cuda_rms_norm_quantize_rows_i8(
        const float *x, const float *mod_scale, const float *bias,
        int8_t *q, float *q_scales, int rows, int cols, float eps) {
    rms_norm_quantize_rows_i8_kernel<<<rows, 256>>>(
        x, mod_scale, bias, q, q_scales, rows, cols, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_dequant_silu_quantize_rows_i8(
        const int32_t *acc, const float *input_row_scales, const float *weight_scales,
        const float *bias, int8_t *q, float *output_row_scales, int rows, int cols) {
    dequant_silu_quantize_rows_i8_kernel<<<rows, 256>>>(
        acc, input_row_scales, weight_scales, bias, q, output_row_scales, rows, cols);
    return wm_cuda_check_launch();
}

int wm_cuda_dequant_gated_residual_f32(
        const int32_t *acc, const float *row_scales, const float *col_scales,
        const float *residual, const float *gate, float *out, int rows, int cols) {
    dequant_gated_residual_f32_kernel<<<div_up_i64((int64_t)rows * cols, 256), 256>>>(
        acc, row_scales, col_scales, residual, gate, out, rows, cols);
    return wm_cuda_check_launch();
}

int wm_cuda_dequant_add_residual_f32(
        const int32_t *acc, const float *row_scales, const float *col_scales,
        const float *residual, float *out, int rows, int cols) {
    dequant_add_residual_f32_kernel<<<div_up_i64((int64_t)rows * cols, 256), 256>>>(
        acc, row_scales, col_scales, residual, out, rows, cols);
    return wm_cuda_check_launch();
}

int wm_cuda_ada_rms_norm_f32(
        const float *x, const float *scale, const float *bias, float *y,
        int rows, int d, float eps) {
    ada_rms_norm_single_f32_kernel<<<rows, 256>>>(x, scale, bias, y, rows, d, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_ada_rms_norm_f16(
        const float *x, const float *scale, const float *bias, __half *y,
        int rows, int d, float eps) {
    ada_rms_norm_single_f16_kernel<<<rows, 256>>>(x, scale, bias, y, rows, d, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_rms_norm_rows_f32(const float *x, float *y, int rows, int d, float eps) {
    rms_norm_rows_f32_kernel<<<rows, 256>>>(x, y, rows, d, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_out_norm_silu_f32(
        const float *tokens, const float *mod, float *out, int rows, int d, float eps) {
    out_norm_silu_f32_kernel<<<rows, 256>>>(tokens, mod, out, rows, d, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_qkv_separate_rms_rope_f32(
        const float *q_raw, const float *k_raw, const float *v_raw,
        float *q, float *k, float *v,
        const int64_t *x_pos, const int64_t *y_pos, const int64_t *t_pos,
        const float *xy, const float *inv_t,
        int tokens, int n_heads, int n_kv_heads, int d_head,
        int width, int height, float eps) {
    dim3 grid(tokens, n_heads + 2 * n_kv_heads);
    size_t shared_bytes = (size_t)(d_head + 256) * sizeof(float);
    qkv_separate_rms_rope_f32_kernel<<<grid, 256, shared_bytes>>>(
        q_raw, k_raw, v_raw, q, k, v,
        x_pos, y_pos, t_pos, xy, inv_t,
        tokens, n_heads, n_kv_heads, d_head, width, height, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_qkv_fused_rms_rope_f32(
        const float *qkv_raw, float *q, float *k, float *v,
        const int64_t *x_pos, const int64_t *y_pos, const int64_t *t_pos,
        const float *xy, const float *inv_t,
        int tokens, int n_heads, int n_kv_heads, int d_head,
        int width, int height, float eps) {
    dim3 grid(tokens, n_heads + 2 * n_kv_heads);
    size_t shared_bytes = (size_t)(d_head + 256) * sizeof(float);
    qkv_fused_rms_rope_f32_kernel<<<grid, 256, shared_bytes>>>(
        qkv_raw, q, k, v, x_pos, y_pos, t_pos, xy, inv_t,
        tokens, n_heads, n_kv_heads, d_head, width, height, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_qkv_fused_rms_rope_i32_dequant(
        const int32_t *qkv_acc, const float *row_scales, const float *weight_scales,
        float *q, float *k, float *v,
        const int64_t *x_pos, const int64_t *y_pos, const int64_t *t_pos,
        const float *xy, const float *inv_t,
        int tokens, int n_heads, int n_kv_heads, int d_head,
        int width, int height, float eps) {
    dim3 grid(tokens, n_heads + 2 * n_kv_heads);
    size_t shared_bytes = (size_t)(d_head + 256) * sizeof(float);
    qkv_fused_rms_rope_i32_dequant_kernel<<<grid, 256, shared_bytes>>>(
        qkv_acc, row_scales, weight_scales, q, k, v,
        x_pos, y_pos, t_pos, xy, inv_t,
        tokens, n_heads, n_kv_heads, d_head, width, height, eps);
    return wm_cuda_check_launch();
}

int wm_cuda_current_frame_attention_f32(
        const float *q, const float *k, const float *v, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int d_head, float scale) {
    current_frame_attention_f32_kernel<<<n_heads * tokens, 256>>>(
        q, k, v, out_tokens, n_heads, n_kv_heads, tokens, d_head, scale);
    return wm_cuda_check_launch();
}

int wm_cuda_init_cache_written(bool *written, int ring_length, int tokens) {
    init_cache_written_kernel<<<div_up_i64((int64_t)ring_length + tokens, 256), 256>>>(
        written, ring_length, tokens);
    return wm_cuda_check_launch();
}

int wm_cuda_kv_cache_upsert_copy_f32(
        float *cache_k, float *cache_v, const float *k, const float *v,
        bool *written, int n_kv_heads, int tokens, int d_head,
        int ring_length, int base, bool write_step, bool frozen) {
    int64_t total = (int64_t)n_kv_heads * tokens * d_head;
    kv_cache_upsert_copy_f32_kernel<<<div_up_i64(total, 256), 256>>>(
        cache_k, cache_v, k, v, written,
        n_kv_heads, tokens, d_head, ring_length, base, write_step, frozen);
    return wm_cuda_check_launch();
}

int wm_cuda_kv_cache_upsert_copy_f16(
        __half *cache_k, __half *cache_v, const float *k, const float *v,
        bool *written, int n_kv_heads, int tokens, int d_head,
        int ring_length, int base, bool write_step, bool frozen) {
    int64_t total = (int64_t)n_kv_heads * tokens * d_head;
    kv_cache_upsert_copy_f16_kernel<<<div_up_i64(total, 256), 256>>>(
        cache_k, cache_v, k, v, written,
        n_kv_heads, tokens, d_head, ring_length, base, write_step, frozen);
    return wm_cuda_check_launch();
}

int wm_cuda_collect_cache_frame_indices(
        const bool *written, int64_t *indices, int32_t *block_ids, int *count,
        int capacity, int tokens, int base, bool write_step) {
    collect_cache_frame_indices_kernel<<<capacity / tokens, 256>>>(
        written, indices, block_ids, count, capacity, tokens, base, write_step);
    return wm_cuda_check_launch();
}

int wm_cuda_indexed_attention_f32(
        const float *q, const float *cache_k, const float *cache_v,
        const int64_t *indices, const int *index_count, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int capacity,
        int d_head, float scale) {
    indexed_attention_cache_f32_kernel<<<n_heads * tokens, 256>>>(
        q, cache_k, cache_v, indices, index_count, out_tokens,
        n_heads, n_kv_heads, tokens, capacity, d_head, scale);
    return wm_cuda_check_launch();
}

int wm_cuda_indexed_attention_d64_warp_f32(
        const float *q, const float *cache_k, const float *cache_v,
        const int64_t *indices, const int *index_count, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int capacity, float scale) {
    indexed_attention_cache_d64_warp_f32_kernel<<<
        div_up_i64((int64_t)n_heads * tokens, 4), 128>>>(
        q, cache_k, cache_v, indices, index_count, out_tokens,
        n_heads, n_kv_heads, tokens, capacity, scale);
    return wm_cuda_check_launch();
}

int wm_cuda_indexed_attention_d64_warp_f16_kv(
        const float *q, const __half *cache_k, const __half *cache_v,
        const int64_t *indices, const int *index_count, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int capacity, float scale) {
    indexed_attention_cache_d64_warp_f16_kv_kernel<<<
        div_up_i64((int64_t)n_heads * tokens, 4), 128>>>(
        q, cache_k, cache_v, indices, index_count, out_tokens,
        n_heads, n_kv_heads, tokens, capacity, scale);
    return wm_cuda_check_launch();
}

static int wm_cuda_validate_flash_shape(int n_heads, int n_kv_heads) {
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) return 1;
    int group = n_heads / n_kv_heads;
    return group <= 0 || group > WORLD_ATTN_D64_FLASH_WARPS;
}

int wm_cuda_indexed_attention_d64_flash_f16_kv(
        const float *q, const __half *cache_k, const __half *cache_v,
        const int64_t *indices, const int *index_count, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int capacity, float scale) {
    if (wm_cuda_validate_flash_shape(n_heads, n_kv_heads)) return 1;
    int group = n_heads / n_kv_heads;
    int q_per_h = WORLD_ATTN_D64_FLASH_WARPS / group;
    int q_blocks = div_up_i64(tokens, q_per_h);
    size_t shared_bytes =
        (size_t)2 * WORLD_ATTN_D64_K_BLOCK * 64 * sizeof(__half);
    indexed_attention_cache_d64_flash_f16_kv_kernel<<<
        n_kv_heads * q_blocks, 32 * WORLD_ATTN_D64_FLASH_WARPS, shared_bytes>>>(
        q, cache_k, cache_v, indices, index_count, out_tokens,
        n_heads, n_kv_heads, tokens, capacity, scale);
    return wm_cuda_check_launch();
}

int wm_cuda_indexed_attention_d64_flash_f32(
        const float *q, const float *cache_k, const float *cache_v,
        const int64_t *indices, const int *index_count, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int capacity, float scale) {
    if (wm_cuda_validate_flash_shape(n_heads, n_kv_heads)) return 1;
    int group = n_heads / n_kv_heads;
    int q_per_h = WORLD_ATTN_D64_FLASH_WARPS / group;
    int q_blocks = div_up_i64(tokens, q_per_h);
    size_t shared_bytes =
        (size_t)2 * WORLD_ATTN_D64_K_BLOCK * 64 * sizeof(float);
    indexed_attention_cache_d64_flash_f32_kernel<<<
        n_kv_heads * q_blocks, 32 * WORLD_ATTN_D64_FLASH_WARPS, shared_bytes>>>(
        q, cache_k, cache_v, indices, index_count, out_tokens,
        n_heads, n_kv_heads, tokens, capacity, scale);
    return wm_cuda_check_launch();
}

int wm_cuda_indexed_attention_d64_q4_shared_f32(
        const float *q, const float *cache_k, const float *cache_v,
        const int64_t *indices, const int *index_count, float *out_tokens,
        int n_heads, int n_kv_heads, int tokens, int capacity, float scale) {
    int q_blocks = div_up_i64(tokens, WORLD_ATTN_D64_Q_BLOCK);
    size_t shared_bytes =
        (size_t)2 * WORLD_ATTN_D64_K_BLOCK * 64 * sizeof(float);
    indexed_attention_cache_d64_q4_shared_f32_kernel<<<
        n_heads * q_blocks, 128, shared_bytes>>>(
        q, cache_k, cache_v, indices, index_count, out_tokens,
        n_heads, n_kv_heads, tokens, capacity, scale);
    return wm_cuda_check_launch();
}

int wm_cuda_gated_residual_add_f32(
        const float *residual, const float *update, const float *gate,
        float *out, int tokens, int d) {
    gated_residual_add_f32_kernel<<<
        div_up_i64((int64_t)tokens * d, 256), 256>>>(
        residual, update, gate, out, tokens, d);
    return wm_cuda_check_launch();
}

int wm_cuda_add_f32(const float *a, const float *b, float *out, int64_t n) {
    add_f32_kernel<<<div_up_i64(n, 256), 256>>>(a, b, out, n);
    return wm_cuda_check_launch();
}

int wm_cuda_latent_update_f32(
        float *latent, const float *velocity, float dsigma, int64_t n) {
    latent_update_f32_kernel<<<div_up_i64(n, 256), 256>>>(latent, velocity, dsigma, n);
    return wm_cuda_check_launch();
}

int wm_cuda_lerp_inplace_f32(
        float *x, const float *end, float weight, int64_t n) {
    lerp_inplace_f32_kernel<<<div_up_i64(n, 256), 256>>>(x, end, weight, n);
    return wm_cuda_check_launch();
}

int wm_cuda_patchify_f32(
        const float *x, const float *weight, float *tokens,
        int channels, int height, int width, int d,
        int patch_h, int patch_w, int token_h, int token_w) {
    patchify_f32_kernel<<<token_h * token_w * d, 256>>>(
        x, weight, tokens, channels, height, width, d,
        patch_h, patch_w, token_h, token_w);
    return wm_cuda_check_launch();
}

int wm_cuda_patchify_im2row_f32(
        const float *x, float *rows,
        int channels, int height, int width,
        int patch_h, int patch_w, int token_h, int token_w) {
    int64_t total = (int64_t)token_h * token_w * channels * patch_h * patch_w;
    patchify_im2row_f32_kernel<<<div_up_i64(total, 256), 256>>>(
        x, rows, channels, height, width,
        patch_h, patch_w, token_h, token_w);
    return wm_cuda_check_launch();
}

int wm_cuda_unpatchify_f32(
        const float *tokens, const float *weight, const float *bias, float *x,
        int token_count, int d, int channels, int height, int width,
        int patch_h, int patch_w, int token_w, int out_dim) {
    unpatchify_orig_f32_kernel<<<token_count * out_dim, 256>>>(
        tokens, weight, bias, x, token_count, d, channels, height, width,
        patch_h, patch_w, token_w, out_dim);
    return wm_cuda_check_launch();
}

int wm_cuda_has_cutlass_fmha(void) {
    return WORLD_HAS_CUTLASS_FMHA;
}
