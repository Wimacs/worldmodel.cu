#include "world_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#if __has_include("kernel_forward.h")
#define WORLD_HAS_CUTLASS_FMHA 1
#include "kernel_forward.h"
#include "world_sparse_fmha.cuh"
#else
#define WORLD_HAS_CUTLASS_FMHA 0
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <math.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WORLD_ATTN_D64_Q_BLOCK 4
#define WORLD_ATTN_D64_K_BLOCK 64
#define WORLD_ATTN_D64_FLASH_WARPS 16

// Experimental row-wise activation / output-channel-wise weight PTQ.  This
// is deliberately distinct from TurboDiffusion's 128x128 blockwise W8A8.
#define WORLD_W8A8_QKV  (1 << 0)
#define WORLD_W8A8_OUT  (1 << 1)
#define WORLD_W8A8_CTRL (1 << 2)
#define WORLD_W8A8_MLP  (1 << 3)
#define WORLD_W8A8_ALL  (WORLD_W8A8_QKV | WORLD_W8A8_OUT | WORLD_W8A8_CTRL | WORLD_W8A8_MLP)

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

static int parse_w8a8_ops(const char *ops) {
    if (!ops || !ops[0] || strcmp(ops, "all") == 0 || strcmp(ops, "1") == 0) {
        return WORLD_W8A8_ALL;
    }
    if (strcmp(ops, "none") == 0 || strcmp(ops, "off") == 0 || strcmp(ops, "0") == 0) {
        return 0;
    }

    int mask = 0;
    const char *p = ops;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        const char *begin = p;
        while (*p && *p != ',' && *p != ' ' && *p != '\t') ++p;
        size_t len = (size_t)(p - begin);
        if (len == 3 && strncmp(begin, "qkv", len) == 0) mask |= WORLD_W8A8_QKV;
        else if (len == 3 && strncmp(begin, "out", len) == 0) mask |= WORLD_W8A8_OUT;
        else if (len == 4 && strncmp(begin, "ctrl", len) == 0) mask |= WORLD_W8A8_CTRL;
        else if (len == 10 && strncmp(begin, "controller", len) == 0) mask |= WORLD_W8A8_CTRL;
        else if (len == 3 && strncmp(begin, "mlp", len) == 0) mask |= WORLD_W8A8_MLP;
        else if (len != 0) {
            fprintf(stderr, "unknown WORLD_W8A8 op '%.*s'\n", (int)len, begin);
            return -1;
        }
    }
    return mask;
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

__global__ static void taehv_copy_latent_clamp_kernel(
        const float *latent,
        float *out,
        int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(latent[i] / 3.0f) * 3.0f;
}

__global__ static void taehv_relu_kernel(float *x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}

__global__ static void taehv_conv2d_nchw_kernel(
        const float *in,
        const float *weight,
        const float *bias,
        float *out,
        int N,
        int C_in,
        int C_out,
        int H,
        int W,
        int K,
        int has_bias) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C_out * H * W;
    if (i >= total) return;

    int x = (int)(i % W);
    int64_t q = i / W;
    int y = (int)(q % H);
    q /= H;
    int co = (int)(q % C_out);
    int n = (int)(q / C_out);
    int pad = K / 2;

    float sum = has_bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = y + ky - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kx = 0; kx < K; ++kx) {
                int ix = x + kx - pad;
                if (ix < 0 || ix >= W) continue;
                float xv = in[((int64_t)n * C_in * H + ci * H + iy) * W + ix];
                float wv = weight[(((int64_t)co * C_in + ci) * K + ky) * K + kx];
                sum += xv * wv;
            }
        }
    }
    out[i] = sum;
}

__global__ static void taehv_conv2d_stride2_nchw_kernel(
        const float *in,
        const float *weight,
        const float *bias,
        float *out,
        int N,
        int C_in,
        int C_out,
        int H,
        int W,
        int K,
        int has_bias) {
    int H_out = (H + 1) / 2;
    int W_out = (W + 1) / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C_out * H_out * W_out;
    if (i >= total) return;

    int x = (int)(i % W_out);
    int64_t q = i / W_out;
    int y = (int)(q % H_out);
    q /= H_out;
    int co = (int)(q % C_out);
    int n = (int)(q / C_out);
    int pad = K / 2;

    float sum = has_bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = y * 2 + ky - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kx = 0; kx < K; ++kx) {
                int ix = x * 2 + kx - pad;
                if (ix < 0 || ix >= W) continue;
                float xv = in[((int64_t)n * C_in * H + ci * H + iy) * W + ix];
                float wv = weight[(((int64_t)co * C_in + ci) * K + ky) * K + kx];
                sum += xv * wv;
            }
        }
    }
    out[i] = sum;
}

__global__ static void taehv_add_bias_nchw_kernel(
        float *out,
        const float *bias,
        int N,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H * W;
    if (i >= total) return;
    int c = (int)((i / ((int64_t)H * W)) % C);
    out[i] += bias[c];
}

__global__ static void taehv_im2col3x3_nchw_tile_kernel(
        const float *in,
        float *cols,
        int C,
        int H,
        int W,
        int frame,
        int tile_start,
        int tile_cols) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int k_elems = C * 9;
    int64_t total = (int64_t)k_elems * tile_cols;
    if (i >= total) return;

    int col = (int)(i % tile_cols);
    int k = (int)(i / tile_cols);
    int spatial = H * W;
    int pos = tile_start + col;
    int x = pos % W;
    int y = pos / W;
    int q = k;
    int kx = q % 3;
    q /= 3;
    int ky = q % 3;
    int c = q / 3;
    int iy = y + ky - 1;
    int ix = x + kx - 1;
    float v = 0.0f;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
        v = in[((int64_t)frame * C * spatial + c * spatial + iy * W + ix)];
    }
    cols[i] = v;
}

__global__ static void taehv_im2col3x3_nchw_batch_tile_kernel(
        const float *in,
        float *cols,
        int N,
        int C,
        int H,
        int W,
        int tile_start,
        int tile_cols) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int k_elems = C * 9;
    int64_t total = (int64_t)k_elems * tile_cols;
    if (i >= total) return;

    int col = (int)(i % tile_cols);
    int k = (int)(i / tile_cols);
    int spatial = H * W;
    int global_col = tile_start + col;
    int frame = global_col / spatial;
    int pos = global_col - frame * spatial;
    if (frame >= N) {
        cols[i] = 0.0f;
        return;
    }
    int x = pos % W;
    int y = pos / W;
    int q = k;
    int kx = q % 3;
    q /= 3;
    int ky = q % 3;
    int c = q / 3;
    int iy = y + ky - 1;
    int ix = x + kx - 1;
    float v = 0.0f;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
        v = in[((int64_t)frame * C * spatial + c * spatial + iy * W + ix)];
    }
    cols[i] = v;
}

__global__ static void taehv_scatter_conv_tile_to_nchw_kernel(
        const float *tile,
        float *out,
        int N,
        int C_out,
        int H,
        int W,
        int tile_start,
        int tile_cols) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C_out * tile_cols;
    if (i >= total) return;

    int col = (int)(i % tile_cols);
    int co = (int)(i / tile_cols);
    int spatial = H * W;
    int global_col = tile_start + col;
    int frame = global_col / spatial;
    int pos = global_col - frame * spatial;
    if (frame >= N) return;
    out[(int64_t)frame * C_out * spatial + (int64_t)co * spatial + pos] = tile[i];
}

__global__ static void taehv_concat_memory_nchw_kernel(
        const float *x,
        const float *past,
        float *out,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)2 * C * H * W;
    if (i >= total) return;

    int w = (int)(i % W);
    int64_t q = i / W;
    int h = (int)(q % H);
    q /= H;
    int c2 = (int)q;
    if (c2 < C) {
        out[i] = x[((int64_t)c2 * H + h) * W + w];
    } else {
        int c = c2 - C;
        out[i] = past[((int64_t)c * H + h) * W + w];
    }
}

__global__ static void taehv_concat_past_nchw_kernel(
        const float *x,
        float *out,
        int N,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * 2 * C * H * W;
    if (i >= total) return;

    int w = (int)(i % W);
    int64_t q = i / W;
    int h = (int)(q % H);
    q /= H;
    int c2 = (int)(q % (2 * C));
    int n = (int)(q / (2 * C));
    if (c2 < C) {
        out[i] = x[((int64_t)n * C * H + c2 * H + h) * W + w];
    } else if (n == 0) {
        out[i] = 0.0f;
    } else {
        int c = c2 - C;
        out[i] = x[((int64_t)(n - 1) * C * H + c * H + h) * W + w];
    }
}

__global__ static void taehv_add_relu_kernel(
        const float *a,
        const float *b,
        float *out,
        int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i] + b[i];
    out[i] = v > 0.0f ? v : 0.0f;
}

__global__ static void taehv_upsample2_nchw_kernel(
        const float *in,
        float *out,
        int N,
        int C,
        int H,
        int W) {
    int H2 = H * 2;
    int W2 = W * 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H2 * W2;
    if (i >= total) return;

    int ox = (int)(i % W2);
    int64_t q = i / W2;
    int oy = (int)(q % H2);
    q /= H2;
    int c = (int)(q % C);
    int n = (int)(q / C);
    out[i] = in[((int64_t)n * C * H + c * H + oy / 2) * W + ox / 2];
}

__global__ static void taehv_tgrow_reshape_kernel(
        const float *in,
        float *out,
        int N,
        int C,
        int H,
        int W,
        int stride) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * stride * C * H * W;
    if (i >= total) return;

    int x = (int)(i % W);
    int64_t q = i / W;
    int y = (int)(q % H);
    q /= H;
    int c = (int)(q % C);
    q /= C;
    int s = (int)(q % stride);
    int n = (int)(q / stride);
    int in_c = s * C + c;
    out[i] = in[((int64_t)n * (C * stride) * H + in_c * H + y) * W + x];
}

__global__ static void taehv_pixel_shuffle_one_u8_kernel(
        const float *in,
        unsigned char *rgb,
        int H,
        int W) {
    int H2 = H * 2;
    int W2 = W * 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H2 * W2 * 3;
    if (i >= total) return;

    int c = (int)(i % 3);
    int64_t q = i / 3;
    int ox = (int)(q % W2);
    q /= W2;
    int oy = (int)(q % H2);
    int ic = c * 4 + (oy & 1) * 2 + (ox & 1);
    float v = in[((int64_t)ic * H + oy / 2) * W + ox / 2];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    int u = (int)floorf(v * 255.0f + 0.5f);
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    rgb[i] = (unsigned char)u;
}

__global__ static void taehv_copy_latent_clamp_nhwc_h_kernel(
        const float *latent,
        __half *out,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * W * C;
    if (i >= total) return;
    int c = (int)(i % C);
    int64_t q = i / C;
    int x = (int)(q % W);
    q /= W;
    int y = (int)(q % H);
    float v = latent[((int64_t)c * H + y) * W + x];
    out[i] = __float2half_rn(tanhf(v / 3.0f) * 3.0f);
}

__global__ static void taehv_relu_nhwc_h_kernel(__half *x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        x[i] = __float2half_rn(v > 0.0f ? v : 0.0f);
    }
}

__global__ static void taehv_add_bias_nhwc_h_kernel(
        __half *out,
        const __half *bias,
        int64_t total,
        int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = __float2half_rn(__half2float(out[i]) + __half2float(bias[i % C]));
}

__global__ static void taehv_concat_memory_nhwc_h_kernel(
        const __half *x,
        const __half *past,
        __half *out,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * W * (2 * C);
    if (i >= total) return;
    int c2 = (int)(i % (2 * C));
    int64_t q = i / (2 * C);
    int w = (int)(q % W);
    q /= W;
    int h = (int)q;
    if (c2 < C) {
        out[i] = x[((int64_t)h * W + w) * C + c2];
    } else {
        int c = c2 - C;
        out[i] = past[((int64_t)h * W + w) * C + c];
    }
}

__global__ static void taehv_add_relu_nhwc_h_kernel(
        const __half *a,
        const __half *b,
        __half *out,
        int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(a[i]) + __half2float(b[i]);
    out[i] = __float2half_rn(v > 0.0f ? v : 0.0f);
}

__global__ static void taehv_upsample2_nhwc_h_kernel(
        const __half *in,
        __half *out,
        int N,
        int C,
        int H,
        int W) {
    int H2 = H * 2;
    int W2 = W * 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H2 * W2 * C;
    if (i >= total) return;
    int c = (int)(i % C);
    int64_t q = i / C;
    int ox = (int)(q % W2);
    q /= W2;
    int oy = (int)(q % H2);
    int n = (int)(q / H2);
    out[i] = in[(((int64_t)n * H + oy / 2) * W + ox / 2) * C + c];
}

__global__ static void taehv_tgrow_reshape_nhwc_h_kernel(
        const __half *in,
        __half *out,
        int N,
        int C,
        int H,
        int W,
        int stride) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * stride * H * W * C;
    if (i >= total) return;
    int c = (int)(i % C);
    int64_t q = i / C;
    int x = (int)(q % W);
    q /= W;
    int y = (int)(q % H);
    q /= H;
    int s = (int)(q % stride);
    int n = (int)(q / stride);
    int in_c = s * C + c;
    out[i] = in[(((int64_t)n * H + y) * W + x) * (C * stride) + in_c];
}

__global__ static void taehv_pixel_shuffle_one_u8_nhwc_h_kernel(
        const __half *in,
        unsigned char *rgb,
        int H,
        int W) {
    int H2 = H * 2;
    int W2 = W * 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H2 * W2 * 3;
    if (i >= total) return;

    int c = (int)(i % 3);
    int64_t q = i / 3;
    int ox = (int)(q % W2);
    q /= W2;
    int oy = (int)(q % H2);
    int ic = c * 4 + (oy & 1) * 2 + (ox & 1);
    float v = __half2float(in[((int64_t)(oy / 2) * W + ox / 2) * 12 + ic]);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    int u = (int)floorf(v * 255.0f + 0.5f);
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    rgb[i] = (unsigned char)u;
}

static uint32_t lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float lcg_uniform01(uint32_t *state) {
    uint32_t r = lcg_next(state);
    return (float)((r >> 8) & 0x00FFFFFFu) / 16777216.0f;
}

static void fill_latent_uniform(float *x, int n, uint32_t *state) {
    for (int i = 0; i < n; ++i) {
        float u = lcg_uniform01(state);
        x[i] = 2.0f * u - 1.0f;
    }
}

static void fill_latent_normal(float *x, int n, uint32_t *state) {
    for (int i = 0; i < n; i += 2) {
        float u1 = lcg_uniform01(state);
        float u2 = lcg_uniform01(state);
        if (u1 < 1.0e-7f) u1 = 1.0e-7f;
        float mag = sqrtf(-2.0f * logf(u1));
        float phase = 2.0f * (float)M_PI * u2;
        x[i] = mag * cosf(phase);
        if (i + 1 < n) x[i + 1] = mag * sinf(phase);
    }
}

static void fill_latent(float *x, int n, unsigned int seed, int noise_mode) {
    uint32_t s = seed ? seed : 1u;
    if (noise_mode == WORLD_NOISE_NORMAL) {
        fill_latent_normal(x, n, &s);
    } else {
        fill_latent_uniform(x, n, &s);
    }
}

static void print_stats(const char *name, const float *x, int n) {
    double sum = 0.0;
    double sq = 0.0;
    float mn = x[0];
    float mx = x[0];
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        sum += v;
        sq += (double)v * (double)v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double mean = sum / (double)n;
    double var = sq / (double)n - mean * mean;
    if (var < 0.0) var = 0.0;
    fprintf(stderr, "%s stats: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
            name, mean, sqrt(var), mn, mx);
}

static int row_major_linear(
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

static int row_major_gemm_i8_i32_can_implement(
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

static int row_major_gemm_i8_i32(
        const int8_t *x_i8,
        const int8_t *w_rm_i8,
        int32_t *acc_i32,
        int m,
        int k,
        int n) {
    if (row_major_gemm_i8_i32_can_implement(x_i8, w_rm_i8, acc_i32, m, k, n)) {
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

static int row_major_linear_fp16_weight_simt(
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

static int row_major_linear_fp16_weight_tensorop(
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

static int row_major_linear_fp16_weight_tensorop_m64n64(
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

static int row_major_linear_fp16_input_weight_tensorop_m64n64(
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

static int row_major_linear_fp16_weight_tensorop_m64n64_silu_half(
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

static int row_major_linear_fp16_input_weight_tensorop_m64n64_silu_half(
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

static int should_use_m64n64_tensorop(int enabled, int m, int k, int n) {
    return enabled && m > 1 && m <= 256 && (m % 64) == 0 &&
           k >= 1024 && n >= 1024 && (k % 32) == 0 && (n % 64) == 0;
}

static size_t row_major_linear_fp16_weight_tensorop_splitk_workspace_size(
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

static int row_major_linear_fp16_input_weight_tensorop_splitk(
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

static size_t row_major_linear_fp16_input_weight_tensorop_splitk_parallel_workspace_size(
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

static int row_major_linear_fp16_input_weight_tensorop_splitk_parallel(
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

static int indexed_attention_cache_d64_cutlass_f16_kv(
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

static int indexed_attention_cache_d64_fmha_f16_kv(
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

static int sparse_attention_cache_d64_fmha_f16_kv(
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

static int indexed_attention_cache_d64_cutlass_grouped_f16_kv(
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
        return indexed_attention_cache_d64_cutlass_f16_kv(
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

static int copy_f32_to_device(float **dst, const float *src, size_t n) {
    *dst = NULL;
    CUDA_OK(cudaMalloc((void **)dst, n * sizeof(float)));
    CUDA_OK(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

static int copy_f32_to_half_device(__half **dst, const float *src, size_t n) {
    *dst = NULL;
    __half *host = (__half *)malloc(n * sizeof(__half));
    if (!host) {
        fprintf(stderr, "failed to allocate half conversion buffer\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) host[i] = __float2half(src[i]);
    cudaError_t err = cudaMalloc((void **)dst, n * sizeof(__half));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        free(host);
        return 1;
    }
    err = cudaMemcpy(*dst, host, n * sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        cudaFree(*dst);
        *dst = NULL;
        free(host);
        return 1;
    }
    free(host);
    return 0;
}

static int copy_f32_to_i8_per_output_device(
        int8_t **dst,
        float **dst_scales,
        const float *src,
        int out_features,
        int in_features) {
    *dst = NULL;
    *dst_scales = NULL;
    size_t count = (size_t)out_features * in_features;
    int8_t *host_q = (int8_t *)malloc(count * sizeof(int8_t));
    float *host_scales = (float *)malloc((size_t)out_features * sizeof(float));
    if (!host_q || !host_scales) {
        fprintf(stderr, "failed to allocate W8 quantization buffers [%d,%d]\n",
                out_features, in_features);
        free(host_q);
        free(host_scales);
        return 1;
    }

    for (int row = 0; row < out_features; ++row) {
        const float *src_row = src + (int64_t)row * in_features;
        int8_t *dst_row = host_q + (int64_t)row * in_features;
        float amax = 0.0f;
        for (int col = 0; col < in_features; ++col) {
            if (!isfinite(src_row[col])) {
                fprintf(stderr,
                        "non-finite W8 weight at output=%d input=%d for [%d,%d]\n",
                        row, col, out_features, in_features);
                free(host_q);
                free(host_scales);
                return 1;
            }
            amax = fmaxf(amax, fabsf(src_row[col]));
        }
        float scale = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
        float inv_scale = amax > 0.0f ? 1.0f / scale : 0.0f;
        host_scales[row] = scale;
        for (int col = 0; col < in_features; ++col) {
            // Match CUDA __float2int_rn and the PyTorch reference: round to
            // nearest with ties to even under the default floating mode.
            int v = (int)nearbyintf(src_row[col] * inv_scale);
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            dst_row[col] = (int8_t)v;
        }
    }

    cudaError_t err = cudaMalloc((void **)dst, count * sizeof(int8_t));
    if (err == cudaSuccess) {
        err = cudaMalloc((void **)dst_scales, (size_t)out_features * sizeof(float));
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(*dst, host_q, count * sizeof(int8_t), cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(*dst_scales, host_scales,
                         (size_t)out_features * sizeof(float), cudaMemcpyHostToDevice);
    }
    free(host_q);
    free(host_scales);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error while copying W8 weights [%d,%d]: %s\n",
                out_features, in_features, cudaGetErrorString(err));
        cudaFree(*dst);
        cudaFree(*dst_scales);
        *dst = NULL;
        *dst_scales = NULL;
        return 1;
    }
    return 0;
}

static int copy_oihw_f32_to_krsc_half_device(
        __half **dst,
        const float *src,
        int out_c,
        int in_c,
        int kernel) {
    *dst = NULL;
    size_t n = (size_t)out_c * in_c * kernel * kernel;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) {
        fprintf(stderr, "failed to allocate KRSC weight packing buffer\n");
        return 1;
    }
    for (int co = 0; co < out_c; ++co) {
        for (int ky = 0; ky < kernel; ++ky) {
            for (int kx = 0; kx < kernel; ++kx) {
                for (int ci = 0; ci < in_c; ++ci) {
                    size_t dst_idx = (((size_t)co * kernel + ky) * kernel + kx) * in_c + ci;
                    size_t src_idx = (((size_t)co * in_c + ci) * kernel + ky) * kernel + kx;
                    host[dst_idx] = src[src_idx];
                }
            }
        }
    }
    int rc = copy_f32_to_half_device(dst, host, n);
    free(host);
    return rc;
}

typedef struct {
    float *cond_bias;
    float *cond_proj_weight;
    float *qkv_proj_weight;
    __half *qkv_proj_weight_h;
    int8_t *qkv_proj_weight_i8;
    float *qkv_proj_weight_i8_scales;
    float *out_proj_weight;
    __half *out_proj_weight_h;
    int8_t *out_proj_weight_i8;
    float *out_proj_weight_i8_scales;
    float v_lamb;
    float *ctrl_fc1_x_weight;
    __half *ctrl_fc1_x_weight_h;
    int8_t *ctrl_fc1_x_weight_i8;
    float *ctrl_fc1_x_weight_i8_scales;
    float *ctrl_fc1_c_weight;
    float *ctrl_fc2_weight;
    __half *ctrl_fc2_weight_h;
    int8_t *ctrl_fc2_weight_i8;
    float *ctrl_fc2_weight_i8_scales;
    float *dit_mlp_fc1_weight;
    __half *dit_mlp_fc1_weight_h;
    int8_t *dit_mlp_fc1_weight_i8;
    float *dit_mlp_fc1_weight_i8_scales;
    float *dit_mlp_fc2_weight;
    __half *dit_mlp_fc2_weight_h;
    int8_t *dit_mlp_fc2_weight_i8;
    float *dit_mlp_fc2_weight_i8_scales;
    int has_ctrl;
} DeviceWorldLayerWeights;

typedef struct {
    float *k;
    float *v;
    __half *k_h;
    __half *v_h;
    bool *written;
    unsigned char *h_slot_written;
    int64_t *indices;
    int32_t *block_ids;
    int *index_count;
    int ring_length;
    int capacity;
    int slot_count;
    int pinned_dilation;
    int is_global;
} DeviceWorldLayerCache;

static int cache_host_index_count(const DeviceWorldLayerCache *cache, int T, int base, bool write_step) {
    if (!cache || !cache->h_slot_written || T <= 0 || cache->slot_count <= 0) return 0;
    int base_slot = base / T;
    int written_slots = 0;
    for (int slot = 0; slot < cache->slot_count; ++slot) {
        int slot_written = cache->h_slot_written[slot] != 0;
        if (write_step && slot == base_slot) slot_written = 0;
        if (slot_written) ++written_slots;
    }
    return written_slots * T;
}

static void cache_host_note_upsert(DeviceWorldLayerCache *cache, int T, int base, bool write_step, bool frozen) {
    if (!cache || !cache->h_slot_written || T <= 0 || cache->slot_count <= 0) return;
    int tail_slot = cache->slot_count - 1;
    cache->h_slot_written[tail_slot] = 1;
    if (!frozen && write_step) {
        int base_slot = base / T;
        if (base_slot >= 0 && base_slot < tail_slot) {
            cache->h_slot_written[base_slot] = 1;
        }
    }
}

static int positive_mod_int(int x, int m) {
    int r = x % m;
    return r < 0 ? r + m : r;
}

static void free_device_world_layers(DeviceWorldLayerWeights *layers, int n_layers) {
    if (!layers) return;
    for (int i = 0; i < n_layers; ++i) {
        cudaFree(layers[i].cond_bias);
        cudaFree(layers[i].cond_proj_weight);
        cudaFree(layers[i].qkv_proj_weight);
        cudaFree(layers[i].qkv_proj_weight_h);
        cudaFree(layers[i].qkv_proj_weight_i8);
        cudaFree(layers[i].qkv_proj_weight_i8_scales);
        cudaFree(layers[i].out_proj_weight);
        cudaFree(layers[i].out_proj_weight_h);
        cudaFree(layers[i].out_proj_weight_i8);
        cudaFree(layers[i].out_proj_weight_i8_scales);
        cudaFree(layers[i].ctrl_fc1_x_weight);
        cudaFree(layers[i].ctrl_fc1_x_weight_h);
        cudaFree(layers[i].ctrl_fc1_x_weight_i8);
        cudaFree(layers[i].ctrl_fc1_x_weight_i8_scales);
        cudaFree(layers[i].ctrl_fc1_c_weight);
        cudaFree(layers[i].ctrl_fc2_weight);
        cudaFree(layers[i].ctrl_fc2_weight_h);
        cudaFree(layers[i].ctrl_fc2_weight_i8);
        cudaFree(layers[i].ctrl_fc2_weight_i8_scales);
        cudaFree(layers[i].dit_mlp_fc1_weight);
        cudaFree(layers[i].dit_mlp_fc1_weight_h);
        cudaFree(layers[i].dit_mlp_fc1_weight_i8);
        cudaFree(layers[i].dit_mlp_fc1_weight_i8_scales);
        cudaFree(layers[i].dit_mlp_fc2_weight);
        cudaFree(layers[i].dit_mlp_fc2_weight_h);
        cudaFree(layers[i].dit_mlp_fc2_weight_i8);
        cudaFree(layers[i].dit_mlp_fc2_weight_i8_scales);
    }
    free(layers);
}

static void free_device_world_caches(DeviceWorldLayerCache *caches, int n_layers) {
    if (!caches) return;
    for (int i = 0; i < n_layers; ++i) {
        cudaFree(caches[i].k);
        cudaFree(caches[i].v);
        cudaFree(caches[i].k_h);
        cudaFree(caches[i].v_h);
        cudaFree(caches[i].written);
        cudaFree(caches[i].indices);
        cudaFree(caches[i].block_ids);
        cudaFree(caches[i].index_count);
        free(caches[i].h_slot_written);
    }
    free(caches);
}

static int alloc_device_world_caches(
        DeviceWorldLayerCache **dst_caches,
        const WorldConfig *cfg,
        int n_layers,
        int T,
        int n_kv_heads,
        int d_head,
        int alloc_half_cache) {
    *dst_caches = NULL;
    DeviceWorldLayerCache *caches = (DeviceWorldLayerCache *)calloc((size_t)n_layers, sizeof(*caches));
    if (!caches) return 1;

    int period = cfg->global_attn_period > 0 ? cfg->global_attn_period : 1;
    int offset = positive_mod_int(cfg->global_attn_offset, period);
    for (int layer = 0; layer < n_layers; ++layer) {
        DeviceWorldLayerCache *c = &caches[layer];
        c->is_global = ((layer - offset) % period) == 0;
        int window = c->is_global ? cfg->global_window : cfg->local_window;
        c->pinned_dilation = c->is_global ? cfg->global_pinned_dilation : 1;
        if (window <= 0 || c->pinned_dilation <= 0) goto fail;
        c->ring_length = window * T;
        c->capacity = c->ring_length + T;
        c->slot_count = c->capacity / T;
        if (c->ring_length % T != 0 || ((c->ring_length / T) % c->pinned_dilation) != 0) goto fail;
        if (c->slot_count <= 0 || c->slot_count * T != c->capacity) goto fail;

        size_t kv_elems = (size_t)n_kv_heads * c->capacity * d_head;
        if (cudaMalloc((void **)&c->k, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->v, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (alloc_half_cache) {
            if (cudaMalloc((void **)&c->k_h, kv_elems * sizeof(__half)) != cudaSuccess) goto fail;
            if (cudaMalloc((void **)&c->v_h, kv_elems * sizeof(__half)) != cudaSuccess) goto fail;
        }
        if (cudaMalloc((void **)&c->written, (size_t)c->capacity * sizeof(bool)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->indices, (size_t)c->capacity * sizeof(int64_t)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->block_ids,
                       (size_t)div_up_i64(c->capacity, 128) * sizeof(int32_t)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->index_count, sizeof(int)) != cudaSuccess) goto fail;
        c->h_slot_written = (unsigned char *)calloc((size_t)c->slot_count, sizeof(unsigned char));
        if (!c->h_slot_written) goto fail;
        c->h_slot_written[c->slot_count - 1] = 1;
        if (cudaMemset(c->k, 0, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (cudaMemset(c->v, 0, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (alloc_half_cache) {
            if (cudaMemset(c->k_h, 0, kv_elems * sizeof(__half)) != cudaSuccess) goto fail;
            if (cudaMemset(c->v_h, 0, kv_elems * sizeof(__half)) != cudaSuccess) goto fail;
        }
        init_cache_written_kernel<<<div_up_i64(c->capacity, 256), 256>>>(c->written, c->ring_length, T);
        if (cudaGetLastError() != cudaSuccess) goto fail;
    }

    *dst_caches = caches;
    return 0;

fail:
    free_device_world_caches(caches, n_layers);
    return 1;
}

static int copy_cond_proj_to_device(float **dst, const WorldLayerWeights *src, int D) {
    *dst = NULL;
    size_t block = (size_t)D * D;
    size_t total = 6 * block;
    float *host = (float *)malloc(total * sizeof(float));
    if (!host) {
        fprintf(stderr, "failed to allocate fused cond projection host weight\n");
        return 1;
    }
    memcpy(host + 0 * block, src->attn_cond_s_weight, block * sizeof(float));
    memcpy(host + 1 * block, src->attn_cond_b_weight, block * sizeof(float));
    memcpy(host + 2 * block, src->attn_cond_g_weight, block * sizeof(float));
    memcpy(host + 3 * block, src->mlp_cond_s_weight, block * sizeof(float));
    memcpy(host + 4 * block, src->mlp_cond_b_weight, block * sizeof(float));
    memcpy(host + 5 * block, src->mlp_cond_g_weight, block * sizeof(float));
    int rc = copy_f32_to_device(dst, host, total);
    free(host);
    return rc;
}

static int copy_qkv_proj_to_device(float **dst, const WorldLayerWeights *src, int D, int kv_dim) {
    *dst = NULL;
    size_t q_elems = (size_t)D * D;
    size_t kv_elems = (size_t)kv_dim * D;
    size_t total = q_elems + 2 * kv_elems;
    float *host = (float *)malloc(total * sizeof(float));
    if (!host) {
        fprintf(stderr, "failed to allocate fused QKV host weight\n");
        return 1;
    }
    memcpy(host, src->q_proj_weight, q_elems * sizeof(float));
    memcpy(host + q_elems, src->k_proj_weight, kv_elems * sizeof(float));
    memcpy(host + q_elems + kv_elems, src->v_proj_weight, kv_elems * sizeof(float));
    int rc = copy_f32_to_device(dst, host, total);
    free(host);
    return rc;
}

static int copy_qkv_proj_to_half_device(__half **dst, const WorldLayerWeights *src, int D, int kv_dim) {
    *dst = NULL;
    size_t q_elems = (size_t)D * D;
    size_t kv_elems = (size_t)kv_dim * D;
    size_t total = q_elems + 2 * kv_elems;
    float *host = (float *)malloc(total * sizeof(float));
    if (!host) {
        fprintf(stderr, "failed to allocate fused QKV host weight\n");
        return 1;
    }
    memcpy(host, src->q_proj_weight, q_elems * sizeof(float));
    memcpy(host + q_elems, src->k_proj_weight, kv_elems * sizeof(float));
    memcpy(host + q_elems + kv_elems, src->v_proj_weight, kv_elems * sizeof(float));
    int rc = copy_f32_to_half_device(dst, host, total);
    free(host);
    return rc;
}

static int copy_qkv_proj_to_i8_device(
        int8_t **dst,
        float **dst_scales,
        const WorldLayerWeights *src,
        int D,
        int kv_dim) {
    *dst = NULL;
    *dst_scales = NULL;
    size_t q_elems = (size_t)D * D;
    size_t kv_elems = (size_t)kv_dim * D;
    size_t total = q_elems + 2 * kv_elems;
    float *host = (float *)malloc(total * sizeof(float));
    if (!host) {
        fprintf(stderr, "failed to allocate fused QKV W8 host weight\n");
        return 1;
    }
    memcpy(host, src->q_proj_weight, q_elems * sizeof(float));
    memcpy(host + q_elems, src->k_proj_weight, kv_elems * sizeof(float));
    memcpy(host + q_elems + kv_elems, src->v_proj_weight, kv_elems * sizeof(float));
    int rc = copy_f32_to_i8_per_output_device(dst, dst_scales, host, D + 2 * kv_dim, D);
    free(host);
    return rc;
}

static int copy_world_layers_to_device(
        DeviceWorldLayerWeights **dst_layers,
        const WorldLayerWeights *src_layers,
        int n_layers,
        int D,
        int kv_dim,
        int mlp_hidden,
        int w8a8_drop_fallback,
        int w8a8_mask,
        int w8a8_layer_begin,
        int w8a8_layer_end) {
    *dst_layers = NULL;
    DeviceWorldLayerWeights *dst = (DeviceWorldLayerWeights *)calloc((size_t)n_layers, sizeof(*dst));
    if (!dst) return 1;

    for (int i = 0; i < n_layers; ++i) {
        const WorldLayerWeights *src = &src_layers[i];
        DeviceWorldLayerWeights *dl = &dst[i];
        int layer_w8a8_mask =
            i >= w8a8_layer_begin && i < w8a8_layer_end ? w8a8_mask : 0;
        dl->has_ctrl = src->has_ctrl;
        dl->v_lamb = src->v_lamb ? src->v_lamb[0] : 0.0f;
        if (copy_f32_to_device(&dl->cond_bias, src->cond_bias, (size_t)D)) goto fail;
        if (copy_cond_proj_to_device(&dl->cond_proj_weight, src, D)) goto fail;
        if ((layer_w8a8_mask & WORLD_W8A8_QKV) &&
                copy_qkv_proj_to_i8_device(
                    &dl->qkv_proj_weight_i8,
                    &dl->qkv_proj_weight_i8_scales,
                    src,
                    D,
                    kv_dim)) goto fail;
        if (!(w8a8_drop_fallback && (layer_w8a8_mask & WORLD_W8A8_QKV))) {
            if (copy_qkv_proj_to_device(&dl->qkv_proj_weight, src, D, kv_dim)) goto fail;
            if (copy_qkv_proj_to_half_device(&dl->qkv_proj_weight_h, src, D, kv_dim)) goto fail;
        }
        if ((layer_w8a8_mask & WORLD_W8A8_OUT) &&
                copy_f32_to_i8_per_output_device(
                    &dl->out_proj_weight_i8,
                    &dl->out_proj_weight_i8_scales,
                    src->out_proj_weight,
                    D,
                    D)) goto fail;
        if (!(w8a8_drop_fallback && (layer_w8a8_mask & WORLD_W8A8_OUT))) {
            if (copy_f32_to_device(&dl->out_proj_weight, src->out_proj_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_half_device(&dl->out_proj_weight_h, src->out_proj_weight, (size_t)D * D)) goto fail;
        }
        if (src->has_ctrl) {
            if (copy_f32_to_device(&dl->ctrl_fc1_c_weight, src->ctrl_fc1_c_weight, (size_t)D * D)) goto fail;
            if ((layer_w8a8_mask & WORLD_W8A8_CTRL) &&
                    copy_f32_to_i8_per_output_device(
                        &dl->ctrl_fc1_x_weight_i8,
                        &dl->ctrl_fc1_x_weight_i8_scales,
                        src->ctrl_fc1_x_weight,
                        D,
                        D)) goto fail;
            if ((layer_w8a8_mask & WORLD_W8A8_CTRL) &&
                    copy_f32_to_i8_per_output_device(
                        &dl->ctrl_fc2_weight_i8,
                        &dl->ctrl_fc2_weight_i8_scales,
                        src->ctrl_fc2_weight,
                        D,
                        D)) goto fail;
            if (!(w8a8_drop_fallback && (layer_w8a8_mask & WORLD_W8A8_CTRL))) {
                if (copy_f32_to_device(&dl->ctrl_fc1_x_weight, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
                if (copy_f32_to_half_device(&dl->ctrl_fc1_x_weight_h, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
                if (copy_f32_to_device(&dl->ctrl_fc2_weight, src->ctrl_fc2_weight, (size_t)D * D)) goto fail;
                if (copy_f32_to_half_device(&dl->ctrl_fc2_weight_h, src->ctrl_fc2_weight, (size_t)D * D)) goto fail;
            }
        }
        if ((layer_w8a8_mask & WORLD_W8A8_MLP) &&
                copy_f32_to_i8_per_output_device(
                    &dl->dit_mlp_fc1_weight_i8,
                    &dl->dit_mlp_fc1_weight_i8_scales,
                    src->dit_mlp_fc1_weight,
                    mlp_hidden,
                    D)) goto fail;
        if ((layer_w8a8_mask & WORLD_W8A8_MLP) &&
                copy_f32_to_i8_per_output_device(
                    &dl->dit_mlp_fc2_weight_i8,
                    &dl->dit_mlp_fc2_weight_i8_scales,
                    src->dit_mlp_fc2_weight,
                    D,
                    mlp_hidden)) goto fail;
        if (!(w8a8_drop_fallback && (layer_w8a8_mask & WORLD_W8A8_MLP))) {
            if (copy_f32_to_device(&dl->dit_mlp_fc1_weight, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
            if (copy_f32_to_half_device(&dl->dit_mlp_fc1_weight_h, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
            if (copy_f32_to_device(&dl->dit_mlp_fc2_weight, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
            if (copy_f32_to_half_device(&dl->dit_mlp_fc2_weight_h, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
        }
    }

    *dst_layers = dst;
    return 0;

fail:
    free_device_world_layers(dst, n_layers);
    return 1;
}

static void fill_noise_embedding(float *emb, float sigma) {
    int half = 256;
    float root2 = sqrtf(2.0f);
    for (int i = 0; i < half; ++i) {
        float a = half == 1 ? 0.0f : (float)i / (float)(half - 1);
        float freq = powf(10000.0f, -a);
        float phase = sigma * 1000.0f * freq;
        emb[i] = sinf(phase) * root2;
        emb[half + i] = cosf(phase) * root2;
    }
}

static void fill_positions(int64_t *x_pos, int64_t *y_pos, int64_t *t_pos, int T, int width, int frame_timestamp) {
    for (int i = 0; i < T; ++i) {
        y_pos[i] = i / width;
        x_pos[i] = i - (int)y_pos[i] * width;
        t_pos[i] = frame_timestamp;
    }
}

static void fill_rope_tables(float *xy, float *inv_t, int d_head, int height, int width) {
    int d_xy = d_head / 8;
    int d_t = d_head / 4;
    int n_xy = (d_xy + 1) / 2;
    float max_freq = (float)(height < width ? height : width) * 0.8f;
    for (int i = 0; i < d_xy; ++i) {
        int src = i / 2;
        float a = n_xy == 1 ? 0.0f : (float)src / (float)(n_xy - 1);
        float v = (1.0f + (max_freq * 0.5f - 1.0f) * a) * (float)M_PI;
        xy[i] = v;
    }
    for (int i = 0; i < d_t; ++i) {
        int src = i / 2;
        float exponent = (float)(2 * src) / (float)d_t;
        inv_t[i] = 1.0f / powf(10000.0f, exponent);
    }
}

static int dump_f32(const char *prefix, const char *name, const float *x, size_t n) {
    if (!prefix || !prefix[0]) return 0;
    char path[4096];
    int len = snprintf(path, sizeof(path), "%s.%s.f32", prefix, name);
    if (len < 0 || (size_t)len >= sizeof(path)) {
        fprintf(stderr, "dump path too long for %s\n", name);
        return 1;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "failed to open dump %s: %s\n", path, strerror(errno));
        return 1;
    }
    int ok = fwrite(x, sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "failed to write dump %s\n", path);
        return 1;
    }
    return 0;
}

static int dump_cache_written_counts(const char *prefix, const DeviceWorldLayerCache *caches, int n_layers) {
    if (!prefix || !prefix[0] || !caches) return 0;
    float *counts = (float *)calloc((size_t)n_layers, sizeof(float));
    if (!counts) {
        fprintf(stderr, "failed to allocate cache count dump buffer\n");
        return 1;
    }

    int rc = 1;
    for (int layer = 0; layer < n_layers; ++layer) {
        const DeviceWorldLayerCache *cache = &caches[layer];
        bool *written = (bool *)malloc((size_t)cache->capacity * sizeof(bool));
        if (!written) {
            fprintf(stderr, "failed to allocate cache written host buffer\n");
            goto cleanup;
        }
        cudaError_t err = cudaMemcpy(written, cache->written, (size_t)cache->capacity * sizeof(bool), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error while dumping cache counts: %s\n", cudaGetErrorString(err));
            free(written);
            goto cleanup;
        }
        int count = 0;
        for (int i = 0; i < cache->capacity; ++i) {
            if (written[i]) count++;
        }
        counts[layer] = (float)count;
        free(written);
    }

    rc = dump_f32(prefix, "cache_written_counts", counts, (size_t)n_layers);

cleanup:
    free(counts);
    return rc;
}

typedef struct {
    float *weight;
    float *bias;
    __half *weight_h;
    __half *weight_krsc_h;
    __half *bias_h;
    int out_c;
    int in_c;
    int kernel;
    int has_bias;
} DeviceVaeConvWeight;

enum {
    WORLD_VAE_STREAM_MEM_MB3_0 = 0,
    WORLD_VAE_STREAM_MEM_MB3_1 = 1,
    WORLD_VAE_STREAM_MEM_MB3_2 = 2,
    WORLD_VAE_STREAM_MEM_MB9_0 = 3,
    WORLD_VAE_STREAM_MEM_MB9_1 = 4,
    WORLD_VAE_STREAM_MEM_MB9_2 = 5,
    WORLD_VAE_STREAM_MEM_MB15_0 = 6,
    WORLD_VAE_STREAM_MEM_MB15_1 = 7,
    WORLD_VAE_STREAM_MEM_MB15_2 = 8,
    WORLD_VAE_STREAM_MEM_COUNT = 9,
};

typedef struct {
    DeviceVaeConvWeight convs[WORLD_VAE_DECODER_CONV_COUNT];
    DeviceVaeConvWeight encoder_convs[WORLD_VAE_ENCODER_CONV_COUNT];
    float *buf0;
    float *buf1;
    float *buf2;
    float *stream_branch0;
    float *stream_branch1;
    float *stream_mem[WORLD_VAE_STREAM_MEM_COUNT];
    float *conv3x3_cols;
    float *conv3x3_out_tile;
    __half *hbuf0;
    __half *hbuf1;
    __half *hbuf2;
    __half *hstream_branch0;
    __half *hstream_branch1;
    __half *hstream_mem[WORLD_VAE_STREAM_MEM_COUNT];
    unsigned char *d_rgb;
    unsigned char *h_rgb;
    size_t max_elems;
    size_t rgb_elems;
    size_t stream_branch0_elems;
    size_t stream_branch1_elems;
    int out_w;
    int out_h;
    int H_pre_shuffle;
    int W_pre_shuffle;
    int stream_started_f32;
    int stream_started_h;
    int encoder_enabled;
    int fp16_nhwc_enabled;
    int cutlass_1x1_enabled;
    int cutlass_3x3_enabled;
    int conv3x3_batch_cols_enabled;
    int conv3x3_tile_cols;
    size_t conv3x3_cols_elems;
    size_t conv3x3_out_tile_elems;
    int profile_enabled;
    cudaEvent_t prof_start;
    cudaEvent_t prof_stop;
    float prof_direct_ms;
    float prof_1x1_gemm_ms;
    float prof_1x1_bias_ms;
    float prof_3x3_im2col_ms;
    float prof_3x3_gemm_ms;
    float prof_3x3_scatter_ms;
    float prof_3x3_bias_ms;
    int prof_direct_calls;
    int prof_1x1_calls;
    int prof_1x1_gemm_launches;
    int prof_3x3_calls;
    int prof_3x3_tiles;
} DeviceVaeDecoder;

static void taehv_pick_scratch(float *cur, float *buf0, float *buf1, float *buf2, float **tmp, float **aux) {
    float *first = buf0 != cur ? buf0 : buf1;
    float *second = (buf0 != cur && buf0 != first) ? buf0 : NULL;
    if (!second && buf1 != cur && buf1 != first) second = buf1;
    if (!second && buf2 != cur && buf2 != first) second = buf2;
    if (!second) second = buf2;
    *tmp = first;
    *aux = second;
}

static void taehv_pick_scratch_h(__half *cur, __half *buf0, __half *buf1, __half *buf2, __half **tmp, __half **aux) {
    __half *first = buf0 != cur ? buf0 : buf1;
    __half *second = (buf0 != cur && buf0 != first) ? buf0 : NULL;
    if (!second && buf1 != cur && buf1 != first) second = buf1;
    if (!second && buf2 != cur && buf2 != first) second = buf2;
    if (!second) second = buf2;
    *tmp = first;
    *aux = second;
}

static int taehv_copy_weights(
        DeviceVaeConvWeight *dev,
        const WorldVaeConvWeight *host,
        int count,
        int pack_krsc_half) {
    memset(dev, 0, (size_t)count * sizeof(dev[0]));
    for (int i = 0; i < count; ++i) {
        const WorldVaeConvWeight *src = &host[i];
        DeviceVaeConvWeight *dst = &dev[i];
        dst->out_c = src->out_c;
        dst->in_c = src->in_c;
        dst->kernel = src->kernel;
        dst->has_bias = src->has_bias;
        size_t w_elems = (size_t)src->out_c * src->in_c * src->kernel * src->kernel;
        CUDA_OK(cudaMalloc((void **)&dst->weight, w_elems * sizeof(float)));
        CUDA_OK(cudaMemcpy(dst->weight, src->weight, w_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (copy_f32_to_half_device(&dst->weight_h, src->weight, w_elems)) return 1;
        if (pack_krsc_half) {
            if (copy_oihw_f32_to_krsc_half_device(
                        &dst->weight_krsc_h,
                        src->weight,
                        src->out_c,
                        src->in_c,
                        src->kernel)) return 1;
        }
        if (src->has_bias) {
            CUDA_OK(cudaMalloc((void **)&dst->bias, (size_t)src->out_c * sizeof(float)));
            CUDA_OK(cudaMemcpy(dst->bias, src->bias, (size_t)src->out_c * sizeof(float), cudaMemcpyHostToDevice));
            if (copy_f32_to_half_device(&dst->bias_h, src->bias, (size_t)src->out_c)) return 1;
        }
    }
    return 0;
}

static void taehv_free_weights(DeviceVaeConvWeight *dev, int count) {
    for (int i = 0; i < count; ++i) {
        cudaFree(dev[i].weight);
        cudaFree(dev[i].bias);
        cudaFree(dev[i].weight_h);
        cudaFree(dev[i].weight_krsc_h);
        cudaFree(dev[i].bias_h);
        dev[i].weight = NULL;
        dev[i].bias = NULL;
        dev[i].weight_h = NULL;
        dev[i].weight_krsc_h = NULL;
        dev[i].bias_h = NULL;
    }
}

static void taehv_decoder_free(DeviceVaeDecoder *dec) {
    if (!dec) return;
    taehv_free_weights(dec->convs, WORLD_VAE_DECODER_CONV_COUNT);
    taehv_free_weights(dec->encoder_convs, WORLD_VAE_ENCODER_CONV_COUNT);
    cudaFree(dec->buf0);
    cudaFree(dec->buf1);
    cudaFree(dec->buf2);
    cudaFree(dec->stream_branch0);
    cudaFree(dec->stream_branch1);
    for (int i = 0; i < WORLD_VAE_STREAM_MEM_COUNT; ++i) {
        cudaFree(dec->stream_mem[i]);
    }
    cudaFree(dec->conv3x3_cols);
    cudaFree(dec->conv3x3_out_tile);
    cudaFree(dec->hbuf0);
    cudaFree(dec->hbuf1);
    cudaFree(dec->hbuf2);
    cudaFree(dec->hstream_branch0);
    cudaFree(dec->hstream_branch1);
    for (int i = 0; i < WORLD_VAE_STREAM_MEM_COUNT; ++i) {
        cudaFree(dec->hstream_mem[i]);
    }
    cudaFree(dec->d_rgb);
    cudaFreeHost(dec->h_rgb);
    if (dec->prof_start) cudaEventDestroy(dec->prof_start);
    if (dec->prof_stop) cudaEventDestroy(dec->prof_stop);
    memset(dec, 0, sizeof(*dec));
}

static size_t taehv_stream_mem_elems(const WorldConfig *cfg, int mem_idx) {
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int C = 0;
    int H = H0;
    int W = W0;
    if (mem_idx < WORLD_VAE_STREAM_MEM_MB9_0) {
        C = 256;
    } else if (mem_idx < WORLD_VAE_STREAM_MEM_MB15_0) {
        C = 128;
        H *= 2;
        W *= 2;
    } else {
        C = 64;
        H *= 4;
        W *= 4;
    }
    return (size_t)C * H * W;
}


static int taehv_decoder_init(DeviceVaeDecoder *dec, const WorldConfig *cfg, const WorldVaeDecoderWeights *host) {
    memset(dec, 0, sizeof(*dec));
    if (!host) return 0;

    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    dec->H_pre_shuffle = H0 * 8;
    dec->W_pre_shuffle = W0 * 8;
    dec->out_h = dec->H_pre_shuffle * 2;
    dec->out_w = dec->W_pre_shuffle * 2;
    dec->max_elems = (size_t)16 * 64 * dec->H_pre_shuffle * dec->W_pre_shuffle;
    dec->rgb_elems = (size_t)4 * dec->out_h * dec->out_w * 3;
    const char *vae_1x1_env = getenv("WORLD_VAE_1X1_GEMM");
    const char *vae_3x3_env = getenv("WORLD_VAE_3X3_GEMM");
    const char *vae_3x3_batch_env = getenv("WORLD_VAE_3X3_BATCH_COLS");
    const char *vae_3x3_tile_env = getenv("WORLD_VAE_3X3_TILE_COLS");
    const char *vae_fp16_nhwc_env = getenv("WORLD_VAE_FP16_NHWC");
    const char *vae_profile_env = getenv("WORLD_VAE_PROFILE");
    dec->cutlass_1x1_enabled = vae_1x1_env ? vae_1x1_env[0] != '0' : 1;
    dec->cutlass_3x3_enabled = vae_3x3_env ? vae_3x3_env[0] != '0' : 1;
    dec->conv3x3_batch_cols_enabled = vae_3x3_batch_env ? vae_3x3_batch_env[0] != '0' : 0;
    dec->fp16_nhwc_enabled = vae_fp16_nhwc_env ? vae_fp16_nhwc_env[0] != '0' : 1;
    dec->profile_enabled = vae_profile_env ? vae_profile_env[0] != '0' : 0;
    dec->conv3x3_tile_cols = 16384;
    if (vae_3x3_tile_env && vae_3x3_tile_env[0]) {
        int requested_tile_cols = atoi(vae_3x3_tile_env);
        if (requested_tile_cols > 0) dec->conv3x3_tile_cols = requested_tile_cols;
    }
    if (dec->conv3x3_tile_cols < 1024) dec->conv3x3_tile_cols = 1024;

#define VAE_INIT_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto fail; \
    } \
} while (0)

    if (taehv_copy_weights(dec->convs, host->convs,
                WORLD_VAE_DECODER_CONV_COUNT, dec->fp16_nhwc_enabled)) goto fail;
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->buf0, dec->max_elems * sizeof(float)));
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->buf1, dec->max_elems * sizeof(float)));
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->buf2, dec->max_elems * sizeof(float)));
    dec->stream_branch0_elems = (size_t)2 * 128 * (H0 * 4) * (W0 * 4);
    dec->stream_branch1_elems = (size_t)2 * 64 * dec->H_pre_shuffle * dec->W_pre_shuffle;
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->stream_branch0, dec->stream_branch0_elems * sizeof(float)));
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->stream_branch1, dec->stream_branch1_elems * sizeof(float)));
    for (int i = 0; i < WORLD_VAE_STREAM_MEM_COUNT; ++i) {
        size_t elems = taehv_stream_mem_elems(cfg, i);
        VAE_INIT_CUDA(cudaMalloc((void **)&dec->stream_mem[i], elems * sizeof(float)));
        VAE_INIT_CUDA(cudaMemset(dec->stream_mem[i], 0, elems * sizeof(float)));
    }
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->d_rgb, dec->rgb_elems));
    VAE_INIT_CUDA(cudaMallocHost((void **)&dec->h_rgb, dec->rgb_elems));
    if (dec->fp16_nhwc_enabled) {
        VAE_INIT_CUDA(cudaMalloc((void **)&dec->hbuf0, dec->max_elems * sizeof(__half)));
        VAE_INIT_CUDA(cudaMalloc((void **)&dec->hbuf1, dec->max_elems * sizeof(__half)));
        VAE_INIT_CUDA(cudaMalloc((void **)&dec->hbuf2, dec->max_elems * sizeof(__half)));
        VAE_INIT_CUDA(cudaMalloc((void **)&dec->hstream_branch0, dec->stream_branch0_elems * sizeof(__half)));
        VAE_INIT_CUDA(cudaMalloc((void **)&dec->hstream_branch1, dec->stream_branch1_elems * sizeof(__half)));
        for (int i = 0; i < WORLD_VAE_STREAM_MEM_COUNT; ++i) {
            size_t elems = taehv_stream_mem_elems(cfg, i);
            VAE_INIT_CUDA(cudaMalloc((void **)&dec->hstream_mem[i], elems * sizeof(__half)));
            VAE_INIT_CUDA(cudaMemset(dec->hstream_mem[i], 0, elems * sizeof(__half)));
        }
    }
    if (dec->profile_enabled) {
        VAE_INIT_CUDA(cudaEventCreate(&dec->prof_start));
        VAE_INIT_CUDA(cudaEventCreate(&dec->prof_stop));
    }
    if (dec->cutlass_3x3_enabled) {
        int max_k_elems = 0;
        int max_out_c = 0;
        for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
            const WorldVaeConvWeight *conv = &host->convs[i];
            if (conv->kernel == 3) {
                int k_elems = conv->in_c * 9;
                if (k_elems > max_k_elems) max_k_elems = k_elems;
                if (conv->out_c > max_out_c) max_out_c = conv->out_c;
            }
        }
        cudaError_t cols_err = cudaErrorMemoryAllocation;
        cudaError_t out_err = dec->conv3x3_batch_cols_enabled ? cudaErrorMemoryAllocation : cudaSuccess;
        while (dec->conv3x3_tile_cols >= 1024 && (cols_err != cudaSuccess || out_err != cudaSuccess)) {
            dec->conv3x3_cols_elems = (size_t)max_k_elems * dec->conv3x3_tile_cols;
            dec->conv3x3_out_tile_elems = (size_t)max_out_c * dec->conv3x3_tile_cols;
            cols_err = cudaMalloc((void **)&dec->conv3x3_cols, dec->conv3x3_cols_elems * sizeof(float));
            out_err = dec->conv3x3_batch_cols_enabled && cols_err == cudaSuccess
                ? cudaMalloc((void **)&dec->conv3x3_out_tile, dec->conv3x3_out_tile_elems * sizeof(float))
                : (cols_err == cudaSuccess ? cudaSuccess : cudaErrorMemoryAllocation);
            if (cols_err != cudaSuccess || out_err != cudaSuccess) {
                fprintf(stderr, "warning: failed to allocate VAE 3x3 CUTLASS workspace tile_cols=%d cols %.2f MiB out %.2f MiB: %s/%s\n",
                        dec->conv3x3_tile_cols,
                        (double)(dec->conv3x3_cols_elems * sizeof(float)) / (1024.0 * 1024.0),
                        (double)(dec->conv3x3_out_tile_elems * sizeof(float)) / (1024.0 * 1024.0),
                        cudaGetErrorString(cols_err),
                        cudaGetErrorString(out_err));
                cudaFree(dec->conv3x3_cols);
                cudaFree(dec->conv3x3_out_tile);
                dec->conv3x3_tile_cols /= 2;
                dec->conv3x3_cols = NULL;
                dec->conv3x3_out_tile = NULL;
                dec->conv3x3_cols_elems = 0;
                dec->conv3x3_out_tile_elems = 0;
            }
        }
        if (cols_err != cudaSuccess || out_err != cudaSuccess) {
            fprintf(stderr, "warning: failed to allocate VAE 3x3 CUTLASS workspace %.2f MiB: %s; falling back to direct 3x3 conv\n",
                    (double)(dec->conv3x3_cols_elems * sizeof(float)) / (1024.0 * 1024.0),
                    cudaGetErrorString(cols_err));
            dec->cutlass_3x3_enabled = 0;
            dec->conv3x3_cols = NULL;
            dec->conv3x3_cols_elems = 0;
        }
    }

    fprintf(stderr, "VAE decoder init: RGB %dx%d, scratch %.2f MiB x3, FP16/NHWC CUTLASS implicit conv %s, F32/NCHW conv, 1x1 CUTLASS GEMM %s, 3x3 CUTLASS GEMM %s tile_cols=%d batch_cols=%s, profiling %s, pinned RGB host buffer\n",
            dec->out_w,
            dec->out_h,
            (double)(dec->max_elems * sizeof(float)) / (1024.0 * 1024.0),
            dec->fp16_nhwc_enabled ? "on" : "off",
            dec->cutlass_1x1_enabled ? "on" : "off",
            dec->cutlass_3x3_enabled ? "on" : "off",
            dec->conv3x3_tile_cols,
            dec->conv3x3_batch_cols_enabled ? "on" : "off",
            dec->profile_enabled ? "on" : "off");
#undef VAE_INIT_CUDA
    return 0;

fail:
#undef VAE_INIT_CUDA
    taehv_decoder_free(dec);
    return 1;
}

static void taehv_profile_reset(DeviceVaeDecoder *dec) {
    if (!dec || !dec->profile_enabled) return;
    dec->prof_direct_ms = 0.0f;
    dec->prof_1x1_gemm_ms = 0.0f;
    dec->prof_1x1_bias_ms = 0.0f;
    dec->prof_3x3_im2col_ms = 0.0f;
    dec->prof_3x3_gemm_ms = 0.0f;
    dec->prof_3x3_scatter_ms = 0.0f;
    dec->prof_3x3_bias_ms = 0.0f;
    dec->prof_direct_calls = 0;
    dec->prof_1x1_calls = 0;
    dec->prof_1x1_gemm_launches = 0;
    dec->prof_3x3_calls = 0;
    dec->prof_3x3_tiles = 0;
}

static int taehv_profile_begin(DeviceVaeDecoder *dec) {
    if (!dec || !dec->profile_enabled) return 0;
    CUDA_OK(cudaEventRecord(dec->prof_start, 0));
    return 0;
}

static int taehv_profile_accum(DeviceVaeDecoder *dec, float *accum) {
    if (!dec || !dec->profile_enabled) return 0;
    float ms = 0.0f;
    CUDA_OK(cudaEventRecord(dec->prof_stop, 0));
    CUDA_OK(cudaEventSynchronize(dec->prof_stop));
    CUDA_OK(cudaEventElapsedTime(&ms, dec->prof_start, dec->prof_stop));
    *accum += ms;
    return 0;
}

static void taehv_profile_print(const DeviceVaeDecoder *dec) {
    if (!dec || !dec->profile_enabled) return;
    fprintf(stderr,
            "VAE profile: direct=%.3fms/%d calls 1x1_gemm=%.3fms/%d launches 1x1_bias=%.3fms 3x3_im2col=%.3fms/%d tiles 3x3_gemm=%.3fms/%d tiles 3x3_scatter=%.3fms 3x3_bias=%.3fms\n",
            dec->prof_direct_ms,
            dec->prof_direct_calls,
            dec->prof_1x1_gemm_ms,
            dec->prof_1x1_gemm_launches,
            dec->prof_1x1_bias_ms,
            dec->prof_3x3_im2col_ms,
            dec->prof_3x3_tiles,
            dec->prof_3x3_gemm_ms,
            dec->prof_3x3_tiles,
            dec->prof_3x3_scatter_ms,
            dec->prof_3x3_bias_ms);
}

static int taehv_run_conv1x1_cutlass_nchw(
        DeviceVaeDecoder *dec,
        const float *in,
        float *out,
        const DeviceVaeConvWeight *conv,
        int N,
        int H,
        int W) {
    int spatial = H * W;
    using Gemm = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        1,
        1>;

    Gemm gemm;
    for (int n = 0; n < N; ++n) {
        const float *in_frame = in + (int64_t)n * conv->in_c * spatial;
        float *out_frame = out + (int64_t)n * conv->out_c * spatial;
        typename Gemm::Arguments args(
            {conv->out_c, spatial, conv->in_c},
            {conv->weight, conv->in_c},
            {in_frame, spatial},
            {out_frame, spatial},
            {out_frame, spatial},
            {1.0f, 0.0f});
        if (taehv_profile_begin(dec)) return 1;
        CUTLASS_OK(gemm(args));
        if (taehv_profile_accum(dec, &dec->prof_1x1_gemm_ms)) return 1;
        if (dec && dec->profile_enabled) dec->prof_1x1_gemm_launches++;
    }
    CUDA_OK(cudaGetLastError());

    if (conv->has_bias) {
        int64_t total = (int64_t)N * conv->out_c * H * W;
        if (taehv_profile_begin(dec)) return 1;
        taehv_add_bias_nchw_kernel<<<div_up_i64(total, 256), 256>>>(out, conv->bias, N, conv->out_c, H, W);
        CUDA_OK(cudaGetLastError());
        if (taehv_profile_accum(dec, &dec->prof_1x1_bias_ms)) return 1;
    }
    if (dec && dec->profile_enabled) dec->prof_1x1_calls++;
    return 0;
}

static int taehv_run_conv3x3_cutlass_nchw(
        DeviceVaeDecoder *dec,
        const float *in,
        float *out,
        const DeviceVaeConvWeight *conv,
        int N,
        int H,
        int W) {
    if (!dec || !dec->conv3x3_cols || dec->conv3x3_tile_cols <= 0) return 1;
    int spatial = H * W;
    int k_elems = conv->in_c * 9;
    if ((size_t)k_elems * dec->conv3x3_tile_cols > dec->conv3x3_cols_elems) return 1;
    if (dec->conv3x3_batch_cols_enabled &&
            (!dec->conv3x3_out_tile ||
             (size_t)conv->out_c * dec->conv3x3_tile_cols > dec->conv3x3_out_tile_elems)) return 1;

    using Gemm = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        1,
        1>;

    Gemm gemm;
    if (dec->conv3x3_batch_cols_enabled) {
        int total_cols = N * spatial;
        for (int tile_start = 0; tile_start < total_cols; tile_start += dec->conv3x3_tile_cols) {
            int tile_cols = total_cols - tile_start;
            if (tile_cols > dec->conv3x3_tile_cols) tile_cols = dec->conv3x3_tile_cols;
            if (taehv_profile_begin(dec)) return 1;
            taehv_im2col3x3_nchw_batch_tile_kernel<<<div_up_i64((int64_t)k_elems * tile_cols, 256), 256>>>(
                in, dec->conv3x3_cols, N, conv->in_c, H, W, tile_start, tile_cols);
            CUDA_OK(cudaGetLastError());
            if (taehv_profile_accum(dec, &dec->prof_3x3_im2col_ms)) return 1;

            typename Gemm::Arguments args(
                {conv->out_c, tile_cols, k_elems},
                {conv->weight, k_elems},
                {dec->conv3x3_cols, tile_cols},
                {dec->conv3x3_out_tile, tile_cols},
                {dec->conv3x3_out_tile, tile_cols},
                {1.0f, 0.0f});
            if (taehv_profile_begin(dec)) return 1;
            CUTLASS_OK(gemm(args));
            CUDA_OK(cudaGetLastError());
            if (taehv_profile_accum(dec, &dec->prof_3x3_gemm_ms)) return 1;

            if (taehv_profile_begin(dec)) return 1;
            taehv_scatter_conv_tile_to_nchw_kernel<<<div_up_i64((int64_t)conv->out_c * tile_cols, 256), 256>>>(
                dec->conv3x3_out_tile, out, N, conv->out_c, H, W, tile_start, tile_cols);
            CUDA_OK(cudaGetLastError());
            if (taehv_profile_accum(dec, &dec->prof_3x3_scatter_ms)) return 1;
            if (dec->profile_enabled) dec->prof_3x3_tiles++;
        }
        if (conv->has_bias) {
            int64_t total = (int64_t)N * conv->out_c * H * W;
            if (taehv_profile_begin(dec)) return 1;
            taehv_add_bias_nchw_kernel<<<div_up_i64(total, 256), 256>>>(out, conv->bias, N, conv->out_c, H, W);
            CUDA_OK(cudaGetLastError());
            if (taehv_profile_accum(dec, &dec->prof_3x3_bias_ms)) return 1;
        }
        if (dec->profile_enabled) dec->prof_3x3_calls++;
        return 0;
    }

    for (int n = 0; n < N; ++n) {
        float *out_frame = out + (int64_t)n * conv->out_c * spatial;
        for (int tile_start = 0; tile_start < spatial; tile_start += dec->conv3x3_tile_cols) {
            int tile_cols = spatial - tile_start;
            if (tile_cols > dec->conv3x3_tile_cols) tile_cols = dec->conv3x3_tile_cols;
            if (taehv_profile_begin(dec)) return 1;
            taehv_im2col3x3_nchw_tile_kernel<<<div_up_i64((int64_t)k_elems * tile_cols, 256), 256>>>(
                in, dec->conv3x3_cols, conv->in_c, H, W, n, tile_start, tile_cols);
            CUDA_OK(cudaGetLastError());
            if (taehv_profile_accum(dec, &dec->prof_3x3_im2col_ms)) return 1;

            typename Gemm::Arguments args(
                {conv->out_c, tile_cols, k_elems},
                {conv->weight, k_elems},
                {dec->conv3x3_cols, tile_cols},
                {out_frame + tile_start, spatial},
                {out_frame + tile_start, spatial},
                {1.0f, 0.0f});
            if (taehv_profile_begin(dec)) return 1;
            CUTLASS_OK(gemm(args));
            CUDA_OK(cudaGetLastError());
            if (taehv_profile_accum(dec, &dec->prof_3x3_gemm_ms)) return 1;
            if (dec->profile_enabled) dec->prof_3x3_tiles++;
        }
    }

    if (conv->has_bias) {
        int64_t total = (int64_t)N * conv->out_c * H * W;
        if (taehv_profile_begin(dec)) return 1;
        taehv_add_bias_nchw_kernel<<<div_up_i64(total, 256), 256>>>(out, conv->bias, N, conv->out_c, H, W);
        CUDA_OK(cudaGetLastError());
        if (taehv_profile_accum(dec, &dec->prof_3x3_bias_ms)) return 1;
    }
    if (dec->profile_enabled) dec->prof_3x3_calls++;
    return 0;
}

static int taehv_run_conv(DeviceVaeDecoder *dec, const float *in, float *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
    if (dec && dec->cutlass_1x1_enabled && conv->kernel == 1) {
        return taehv_run_conv1x1_cutlass_nchw(dec, in, out, conv, N, H, W);
    }
    if (dec && dec->cutlass_3x3_enabled && conv->kernel == 3) {
        return taehv_run_conv3x3_cutlass_nchw(dec, in, out, conv, N, H, W);
    }
    int64_t total = (int64_t)N * conv->out_c * H * W;
    if (taehv_profile_begin(dec)) return 1;
    taehv_conv2d_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, conv->weight, conv->bias, out, N, conv->in_c, conv->out_c, H, W, conv->kernel, conv->has_bias);
    CUDA_OK(cudaGetLastError());
    if (taehv_profile_accum(dec, &dec->prof_direct_ms)) return 1;
    if (dec && dec->profile_enabled) dec->prof_direct_calls++;
    return 0;
}

static int taehv_run_conv_h_nhwc(DeviceVaeDecoder *dec, const __half *in, __half *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
    if (!in || !out || !conv || !conv->weight_krsc_h) return 1;
    if (conv->has_bias && !conv->bias_h) return 1;
    if (conv->kernel != 1 && conv->kernel != 3) return 1;

    using Layout = cutlass::layout::TensorNHWC;
    using Element = cutlass::half_t;
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Element,
        Layout,
        Element,
        Layout,
        Element,
        Layout,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 64, 32>,
        cutlass::gemm::GemmShape<64, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<Element, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kAnalytic,
        cutlass::conv::StrideSupport::kUnity,
        8,
        8>::Kernel;
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    int K = conv->kernel;
    int pad = K / 2;
    cutlass::Tensor4DCoord input_size(N, H, W, conv->in_c);
    cutlass::Tensor4DCoord filter_size(conv->out_c, K, K, conv->in_c);
    cutlass::Tensor4DCoord padding(pad, pad, pad, pad);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(N, H, W, conv->out_c);
    cutlass::conv::Conv2dProblemSize problem(
        input_size,
        filter_size,
        padding,
        stride,
        dilation,
        output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        1);

    Element *in_ptr = const_cast<Element *>(reinterpret_cast<const Element *>(in));
    Element *weight_ptr = const_cast<Element *>(reinterpret_cast<const Element *>(conv->weight_krsc_h));
    Element *out_ptr = reinterpret_cast<Element *>(out);
    typename ImplicitGemm::Arguments args(
        problem,
        cutlass::TensorRef<Element, Layout>(in_ptr, Layout::packed(input_size)),
        cutlass::TensorRef<Element, Layout>(weight_ptr, Layout::packed(filter_size)),
        cutlass::TensorRef<Element, Layout>(out_ptr, Layout::packed(output_size)),
        cutlass::TensorRef<Element, Layout>(out_ptr, Layout::packed(output_size)),
        {1.0f, 0.0f});

    ImplicitGemm implicit_gemm;
    CUTLASS_OK(implicit_gemm.can_implement(args));
    size_t workspace_size = implicit_gemm.get_workspace_size(args);
    void *workspace = NULL;
    if (workspace_size > 0) {
        CUDA_OK(cudaMalloc(&workspace, workspace_size));
    }
    if (taehv_profile_begin(dec)) {
        cudaFree(workspace);
        return 1;
    }
    cutlass::Status status = implicit_gemm(args, workspace, 0);
    cudaFree(workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS error %s:%d: %s\n", __FILE__, __LINE__, cutlassGetStatusString(status));
        return 1;
    }
    CUDA_OK(cudaGetLastError());
    if (conv->kernel == 1) {
        if (taehv_profile_accum(dec, &dec->prof_1x1_gemm_ms)) return 1;
        if (dec && dec->profile_enabled) {
            dec->prof_1x1_calls++;
            dec->prof_1x1_gemm_launches++;
        }
    } else {
        if (taehv_profile_accum(dec, &dec->prof_3x3_gemm_ms)) return 1;
        if (dec && dec->profile_enabled) {
            dec->prof_3x3_calls++;
            dec->prof_3x3_tiles++;
        }
    }

    if (conv->has_bias) {
        int64_t total = (int64_t)N * H * W * conv->out_c;
        if (taehv_profile_begin(dec)) return 1;
        taehv_add_bias_nhwc_h_kernel<<<div_up_i64(total, 256), 256>>>(out, conv->bias_h, total, conv->out_c);
        CUDA_OK(cudaGetLastError());
        if (conv->kernel == 1) {
            if (taehv_profile_accum(dec, &dec->prof_1x1_bias_ms)) return 1;
        } else {
            if (taehv_profile_accum(dec, &dec->prof_3x3_bias_ms)) return 1;
        }
    }
    return 0;
}

static int taehv_run_conv_stride2(
        const float *in,
        float *out,
        const DeviceVaeConvWeight *conv,
        int N,
        int H,
        int W) {
    if (!in || !out || !conv || conv->kernel != 3) return 1;
    int H_out = (H + 1) / 2;
    int W_out = (W + 1) / 2;
    int64_t total = (int64_t)N * conv->out_c * H_out * W_out;
    taehv_conv2d_stride2_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, conv->weight, conv->bias, out,
        N, conv->in_c, conv->out_c, H, W, conv->kernel, conv->has_bias);
    CUDA_OK(cudaGetLastError());
    return 0;
}

static int taehv_run_relu(float *x, int64_t n) {
    taehv_relu_kernel<<<div_up_i64(n, 256), 256>>>(x, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

static int taehv_run_relu_h(__half *x, int64_t n) {
    taehv_relu_nhwc_h_kernel<<<div_up_i64(n, 256), 256>>>(x, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

static int taehv_run_memblock_stream(
        DeviceVaeDecoder *dec,
        float **cur_io,
        float *buf0,
        float *buf1,
        float *buf2,
        float *mem,
        const DeviceVaeConvWeight *conv0,
        const DeviceVaeConvWeight *conv2,
        const DeviceVaeConvWeight *conv4,
        int C,
        int H,
        int W) {
    float *cur = *cur_io;
    float *tmp = NULL;
    float *aux = NULL;
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux);

    int64_t elems = (int64_t)C * H * W;
    taehv_concat_memory_nchw_kernel<<<div_up_i64(elems * 2, 256), 256>>>(cur, mem, aux, C, H, W);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(mem, cur, elems * sizeof(float), cudaMemcpyDeviceToDevice));
    if (taehv_run_conv(dec, aux, tmp, conv0, 1, H, W)) return 1;
    if (taehv_run_relu(tmp, elems)) return 1;
    if (taehv_run_conv(dec, tmp, aux, conv2, 1, H, W)) return 1;
    if (taehv_run_relu(aux, elems)) return 1;
    if (taehv_run_conv(dec, aux, tmp, conv4, 1, H, W)) return 1;
    taehv_add_relu_kernel<<<div_up_i64(elems, 256), 256>>>(cur, tmp, aux, elems);
    CUDA_OK(cudaGetLastError());
    *cur_io = aux;
    return 0;
}

static int taehv_run_memblock_stream_h_nhwc(
        DeviceVaeDecoder *dec,
        __half **cur_io,
        __half *buf0,
        __half *buf1,
        __half *buf2,
        __half *mem,
        const DeviceVaeConvWeight *conv0,
        const DeviceVaeConvWeight *conv2,
        const DeviceVaeConvWeight *conv4,
        int C,
        int H,
        int W) {
    __half *cur = *cur_io;
    __half *tmp = NULL;
    __half *aux = NULL;
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux);

    int64_t elems = (int64_t)H * W * C;
    taehv_concat_memory_nhwc_h_kernel<<<div_up_i64(elems * 2, 256), 256>>>(cur, mem, aux, C, H, W);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(mem, cur, elems * sizeof(__half), cudaMemcpyDeviceToDevice));
    if (taehv_run_conv_h_nhwc(dec, aux, tmp, conv0, 1, H, W)) return 1;
    if (taehv_run_relu_h(tmp, elems)) return 1;
    if (taehv_run_conv_h_nhwc(dec, tmp, aux, conv2, 1, H, W)) return 1;
    if (taehv_run_relu_h(aux, elems)) return 1;
    if (taehv_run_conv_h_nhwc(dec, aux, tmp, conv4, 1, H, W)) return 1;
    taehv_add_relu_nhwc_h_kernel<<<div_up_i64(elems, 256), 256>>>(cur, tmp, aux, elems);
    CUDA_OK(cudaGetLastError());
    *cur_io = aux;
    return 0;
}

static int taehv_run_memblock_batch(
        DeviceVaeDecoder *dec,
        float **cur_io,
        float *buf0,
        float *buf1,
        float *buf2,
        const DeviceVaeConvWeight *conv0,
        const DeviceVaeConvWeight *conv2,
        const DeviceVaeConvWeight *conv4,
        int N,
        int C,
        int H,
        int W) {
    float *cur = *cur_io;
    float *tmp = NULL;
    float *aux = NULL;
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux);

    int64_t elems = (int64_t)N * C * H * W;
    taehv_concat_past_nchw_kernel<<<div_up_i64(elems * 2, 256), 256>>>(
        cur, aux, N, C, H, W);
    CUDA_OK(cudaGetLastError());
    if (taehv_run_conv(dec, aux, tmp, conv0, N, H, W)) return 1;
    if (taehv_run_relu(tmp, elems)) return 1;
    if (taehv_run_conv(dec, tmp, aux, conv2, N, H, W)) return 1;
    if (taehv_run_relu(aux, elems)) return 1;
    if (taehv_run_conv(dec, aux, tmp, conv4, N, H, W)) return 1;
    taehv_add_relu_kernel<<<div_up_i64(elems, 256), 256>>>(cur, tmp, aux, elems);
    CUDA_OK(cudaGetLastError());
    *cur_io = aux;
    return 0;
}

static int taehv_encode_image_rgb(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *rgb,
        int width,
        int height,
        float *latent_out) {
    if (!cfg || !dec || !dec->encoder_enabled || !rgb || !latent_out) return 1;
    int expected_h = cfg->height * cfg->patch_h * 16;
    int expected_w = cfg->width * cfg->patch_w * 16;
    if (width != expected_w || height != expected_h || (width & 1) || (height & 1)) {
        fprintf(stderr, "VAE encoder expected RGB %dx%d, got %dx%d\n",
                expected_w, expected_h, width, height);
        return 1;
    }

    int N = 4;
    int C = 12;
    int H = height / 2;
    int W = width / 2;
    size_t input_elems = (size_t)N * C * H * W;
    if (input_elems > dec->max_elems) return 1;
    float *input = (float *)malloc(input_elems * sizeof(float));
    if (!input) return 1;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < 3; ++c) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int out_c = c * 4 + dy * 2 + dx;
                    float *dst = input + ((size_t)n * C + out_c) * H * W;
                    for (int y = 0; y < H; ++y) {
                        const float *src = rgb + ((size_t)(y * 2 + dy) * width + dx) * 3 + c;
                        for (int x = 0; x < W; ++x) dst[(size_t)y * W + x] = src[(size_t)x * 6];
                    }
                }
            }
        }
    }
    CUDA_OK(cudaMemcpy(dec->buf0, input, input_elems * sizeof(float), cudaMemcpyHostToDevice));
    free(input);

    float *buf0 = dec->buf0;
    float *buf1 = dec->buf1;
    float *buf2 = dec->buf2;
    float *cur = buf0;
    float *tmp = NULL;
    float *aux = NULL;

#define VAE_ENC_CONV_TO(idx, out_c) do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv(dec, cur, tmp, &dec->encoder_convs[(idx)], N, H, W)) return 1; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_ENC_RELU() do { \
    if (taehv_run_relu(cur, (int64_t)N * C * H * W)) return 1; \
} while (0)
#define VAE_ENC_MEMBLOCK(a, b, c) do { \
    if (taehv_run_memblock_batch(dec, &cur, buf0, buf1, buf2, \
                &dec->encoder_convs[(a)], &dec->encoder_convs[(b)], \
                &dec->encoder_convs[(c)], N, C, H, W)) return 1; \
} while (0)
#define VAE_ENC_STRIDE2(idx, out_c) do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv_stride2(cur, tmp, &dec->encoder_convs[(idx)], N, H, W)) return 1; \
    cur = tmp; \
    C = (out_c); \
    H = (H + 1) / 2; \
    W = (W + 1) / 2; \
} while (0)

    VAE_ENC_CONV_TO(WORLD_VAE_ENC_CONV_IN, 64);
    VAE_ENC_RELU();
    N /= 2;
    C *= 2;
    VAE_ENC_CONV_TO(WORLD_VAE_ENC_TPOOL2, 64);
    VAE_ENC_STRIDE2(WORLD_VAE_ENC_CONV3, 64);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB4_0, WORLD_VAE_ENC_MB4_2, WORLD_VAE_ENC_MB4_4);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB5_0, WORLD_VAE_ENC_MB5_2, WORLD_VAE_ENC_MB5_4);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB6_0, WORLD_VAE_ENC_MB6_2, WORLD_VAE_ENC_MB6_4);
    N /= 2;
    C *= 2;
    VAE_ENC_CONV_TO(WORLD_VAE_ENC_TPOOL7, 64);
    VAE_ENC_STRIDE2(WORLD_VAE_ENC_CONV8, 64);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB9_0, WORLD_VAE_ENC_MB9_2, WORLD_VAE_ENC_MB9_4);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB10_0, WORLD_VAE_ENC_MB10_2, WORLD_VAE_ENC_MB10_4);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB11_0, WORLD_VAE_ENC_MB11_2, WORLD_VAE_ENC_MB11_4);
    VAE_ENC_CONV_TO(WORLD_VAE_ENC_TPOOL12, 64);
    VAE_ENC_STRIDE2(WORLD_VAE_ENC_CONV13, 64);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB14_0, WORLD_VAE_ENC_MB14_2, WORLD_VAE_ENC_MB14_4);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB15_0, WORLD_VAE_ENC_MB15_2, WORLD_VAE_ENC_MB15_4);
    VAE_ENC_MEMBLOCK(WORLD_VAE_ENC_MB16_0, WORLD_VAE_ENC_MB16_2, WORLD_VAE_ENC_MB16_4);
    VAE_ENC_CONV_TO(WORLD_VAE_ENC_CONV_OUT, cfg->channels);

#undef VAE_ENC_CONV_TO
#undef VAE_ENC_RELU
#undef VAE_ENC_MEMBLOCK
#undef VAE_ENC_STRIDE2

    if (N != 1 || C != cfg->channels || H != cfg->height * cfg->patch_h ||
            W != cfg->width * cfg->patch_w) {
        fprintf(stderr, "VAE encoder produced unexpected latent [%d,%d,%d,%d]\n", N, C, H, W);
        return 1;
    }
    CUDA_OK(cudaMemcpy(latent_out, cur,
                (size_t)C * H * W * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());
    return 0;
}

static int taehv_stream_decode_one_f32(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *d_latent,
        int emit_rgb) {
    int C_latent = cfg->channels;
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int H = H0;
    int W = W0;
    int C = C_latent;

    float *buf0 = dec->buf0;
    float *buf1 = dec->buf1;
    float *buf2 = dec->buf2;
    float *cur = buf0;
    float *tmp = NULL;
    float *aux = NULL;

    taehv_copy_latent_clamp_kernel<<<div_up_i64((int64_t)C * H * W, 256), 256>>>(
        d_latent, cur, (int64_t)C * H * W);
    CUDA_OK(cudaGetLastError());

#define VAE_CONV_TO_STREAM(idx, out_c) do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv(dec, cur, tmp, &dec->convs[(idx)], 1, H, W)) return 1; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_RELU_STREAM() do { \
    if (taehv_run_relu(cur, (int64_t)C * H * W)) return 1; \
} while (0)
#define VAE_MEMBLOCK_STREAM(mem_idx, a, b, c) do { \
    if (taehv_run_memblock_stream(dec, &cur, buf0, buf1, buf2, dec->stream_mem[(mem_idx)], \
                &dec->convs[(a)], &dec->convs[(b)], &dec->convs[(c)], C, H, W)) return 1; \
} while (0)
#define VAE_UPSAMPLE2_STREAM() do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    taehv_upsample2_nchw_kernel<<<div_up_i64((int64_t)(H * 2) * (W * 2) * C, 256), 256>>>(cur, tmp, 1, C, H, W); \
    CUDA_OK(cudaGetLastError()); \
    cur = tmp; \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_TGROW1_STREAM(idx) do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv(dec, cur, tmp, &dec->convs[(idx)], 1, H, W)) return 1; \
    cur = tmp; \
} while (0)

    VAE_CONV_TO_STREAM(WORLD_VAE_DEC_CONV_IN, 256);
    VAE_RELU_STREAM();
    VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB3_0, WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB3_1, WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB3_2, WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_UPSAMPLE2_STREAM();
    VAE_TGROW1_STREAM(WORLD_VAE_DEC_TGROW7);

    VAE_CONV_TO_STREAM(WORLD_VAE_DEC_CONV8, 128);
    VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB9_0, WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB9_1, WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB9_2, WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_UPSAMPLE2_STREAM();

    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux);
    if (taehv_run_conv(dec, cur, tmp, &dec->convs[WORLD_VAE_DEC_TGROW13], 1, H, W)) return 1;
    taehv_tgrow_reshape_kernel<<<div_up_i64((int64_t)2 * C * H * W, 256), 256>>>(tmp, dec->stream_branch0, 1, C, H, W, 2);
    CUDA_OK(cudaGetLastError());

    int branch0_C = C;
    int branch0_H = H;
    int branch0_W = W;
    for (int b = 0; b < 2; ++b) {
        int64_t branch_elems = (int64_t)branch0_C * branch0_H * branch0_W;
        CUDA_OK(cudaMemcpy(buf0, dec->stream_branch0 + (int64_t)b * branch_elems,
                    branch_elems * sizeof(float), cudaMemcpyDeviceToDevice));
        cur = buf0;
        C = branch0_C;
        H = branch0_H;
        W = branch0_W;

        VAE_CONV_TO_STREAM(WORLD_VAE_DEC_CONV14, 64);
        VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB15_0, WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
        VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB15_1, WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
        VAE_MEMBLOCK_STREAM(WORLD_VAE_STREAM_MEM_MB15_2, WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
        VAE_UPSAMPLE2_STREAM();

        taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux);
        if (taehv_run_conv(dec, cur, tmp, &dec->convs[WORLD_VAE_DEC_TGROW19], 1, H, W)) return 1;
        taehv_tgrow_reshape_kernel<<<div_up_i64((int64_t)2 * C * H * W, 256), 256>>>(tmp, dec->stream_branch1, 1, C, H, W, 2);
        CUDA_OK(cudaGetLastError());

        int branch1_C = C;
        int branch1_H = H;
        int branch1_W = W;
        for (int s = 0; s < 2; ++s) {
            int64_t sub_elems = (int64_t)branch1_C * branch1_H * branch1_W;
            CUDA_OK(cudaMemcpy(buf0, dec->stream_branch1 + (int64_t)s * sub_elems,
                        sub_elems * sizeof(float), cudaMemcpyDeviceToDevice));
            cur = buf0;
            C = branch1_C;
            H = branch1_H;
            W = branch1_W;

            VAE_CONV_TO_STREAM(WORLD_VAE_DEC_CONV20, 64);
            VAE_RELU_STREAM();
            VAE_CONV_TO_STREAM(WORLD_VAE_DEC_CONV_OUT, 12);
            if (emit_rgb) {
                unsigned char *frame_rgb = dec->d_rgb + (size_t)(b * 2 + s) * dec->out_h * dec->out_w * 3;
                taehv_pixel_shuffle_one_u8_kernel<<<div_up_i64((int64_t)dec->out_h * dec->out_w * 3, 256), 256>>>(
                    cur, frame_rgb, H, W);
                CUDA_OK(cudaGetLastError());
            }
        }
    }

#undef VAE_CONV_TO_STREAM
#undef VAE_RELU_STREAM
#undef VAE_MEMBLOCK_STREAM
#undef VAE_UPSAMPLE2_STREAM
#undef VAE_TGROW1_STREAM
    return 0;
}

static int taehv_stream_decode_one_h_nhwc(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *d_latent,
        int emit_rgb) {
    int C_latent = cfg->channels;
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int H = H0;
    int W = W0;
    int C = C_latent;

    __half *buf0 = dec->hbuf0;
    __half *buf1 = dec->hbuf1;
    __half *buf2 = dec->hbuf2;
    __half *cur = buf0;
    __half *tmp = NULL;
    __half *aux = NULL;

    taehv_copy_latent_clamp_nhwc_h_kernel<<<div_up_i64((int64_t)H * W * C, 256), 256>>>(
        d_latent, cur, C, H, W);
    CUDA_OK(cudaGetLastError());

#define VAE_CONV_TO_STREAM_H(idx, out_c) do { \
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv_h_nhwc(dec, cur, tmp, &dec->convs[(idx)], 1, H, W)) return 1; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_RELU_STREAM_H() do { \
    if (taehv_run_relu_h(cur, (int64_t)H * W * C)) return 1; \
} while (0)
#define VAE_MEMBLOCK_STREAM_H(mem_idx, a, b, c) do { \
    if (taehv_run_memblock_stream_h_nhwc(dec, &cur, buf0, buf1, buf2, dec->hstream_mem[(mem_idx)], \
                &dec->convs[(a)], &dec->convs[(b)], &dec->convs[(c)], C, H, W)) return 1; \
} while (0)
#define VAE_UPSAMPLE2_STREAM_H() do { \
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux); \
    taehv_upsample2_nhwc_h_kernel<<<div_up_i64((int64_t)(H * 2) * (W * 2) * C, 256), 256>>>(cur, tmp, 1, C, H, W); \
    CUDA_OK(cudaGetLastError()); \
    cur = tmp; \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_TGROW1_STREAM_H(idx) do { \
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv_h_nhwc(dec, cur, tmp, &dec->convs[(idx)], 1, H, W)) return 1; \
    cur = tmp; \
} while (0)

    VAE_CONV_TO_STREAM_H(WORLD_VAE_DEC_CONV_IN, 256);
    VAE_RELU_STREAM_H();
    VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB3_0, WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB3_1, WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB3_2, WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_UPSAMPLE2_STREAM_H();
    VAE_TGROW1_STREAM_H(WORLD_VAE_DEC_TGROW7);

    VAE_CONV_TO_STREAM_H(WORLD_VAE_DEC_CONV8, 128);
    VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB9_0, WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB9_1, WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB9_2, WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_UPSAMPLE2_STREAM_H();

    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux);
    if (taehv_run_conv_h_nhwc(dec, cur, tmp, &dec->convs[WORLD_VAE_DEC_TGROW13], 1, H, W)) return 1;
    taehv_tgrow_reshape_nhwc_h_kernel<<<div_up_i64((int64_t)2 * H * W * C, 256), 256>>>(tmp, dec->hstream_branch0, 1, C, H, W, 2);
    CUDA_OK(cudaGetLastError());

    int branch0_C = C;
    int branch0_H = H;
    int branch0_W = W;
    for (int b = 0; b < 2; ++b) {
        int64_t branch_elems = (int64_t)branch0_H * branch0_W * branch0_C;
        CUDA_OK(cudaMemcpy(buf0, dec->hstream_branch0 + (int64_t)b * branch_elems,
                    branch_elems * sizeof(__half), cudaMemcpyDeviceToDevice));
        cur = buf0;
        C = branch0_C;
        H = branch0_H;
        W = branch0_W;

        VAE_CONV_TO_STREAM_H(WORLD_VAE_DEC_CONV14, 64);
        VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB15_0, WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
        VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB15_1, WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
        VAE_MEMBLOCK_STREAM_H(WORLD_VAE_STREAM_MEM_MB15_2, WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
        VAE_UPSAMPLE2_STREAM_H();

        taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux);
        if (taehv_run_conv_h_nhwc(dec, cur, tmp, &dec->convs[WORLD_VAE_DEC_TGROW19], 1, H, W)) return 1;
        taehv_tgrow_reshape_nhwc_h_kernel<<<div_up_i64((int64_t)2 * H * W * C, 256), 256>>>(tmp, dec->hstream_branch1, 1, C, H, W, 2);
        CUDA_OK(cudaGetLastError());

        int branch1_C = C;
        int branch1_H = H;
        int branch1_W = W;
        for (int s = 0; s < 2; ++s) {
            int64_t sub_elems = (int64_t)branch1_H * branch1_W * branch1_C;
            CUDA_OK(cudaMemcpy(buf0, dec->hstream_branch1 + (int64_t)s * sub_elems,
                        sub_elems * sizeof(__half), cudaMemcpyDeviceToDevice));
            cur = buf0;
            C = branch1_C;
            H = branch1_H;
            W = branch1_W;

            VAE_CONV_TO_STREAM_H(WORLD_VAE_DEC_CONV20, 64);
            VAE_RELU_STREAM_H();
            VAE_CONV_TO_STREAM_H(WORLD_VAE_DEC_CONV_OUT, 12);
            if (emit_rgb) {
                unsigned char *frame_rgb = dec->d_rgb + (size_t)(b * 2 + s) * dec->out_h * dec->out_w * 3;
                taehv_pixel_shuffle_one_u8_nhwc_h_kernel<<<div_up_i64((int64_t)dec->out_h * dec->out_w * 3, 256), 256>>>(
                    cur, frame_rgb, H, W);
                CUDA_OK(cudaGetLastError());
            }
        }
    }

#undef VAE_CONV_TO_STREAM_H
#undef VAE_RELU_STREAM_H
#undef VAE_MEMBLOCK_STREAM_H
#undef VAE_UPSAMPLE2_STREAM_H
#undef VAE_TGROW1_STREAM_H
    return 0;
}

static int taehv_write_ppm(const char *path, const unsigned char *rgb, int width, int height) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "failed to open output image %s: %s\n", path, strerror(errno));
        return 1;
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    size_t n = (size_t)width * height * 3;
    int ok = fwrite(rgb, 1, n, f) == n;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "failed to write output image %s\n", path);
        return 1;
    }
    return 0;
}

static int taehv_make_frame_path(char *out, size_t out_size, const char *path, int frame_idx) {
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    const char *dot = strrchr(name, '.');
    int stem_len = dot ? (int)(dot - path) : (int)strlen(path);
    const char *ext = dot ? dot : "";
    int n = snprintf(out, out_size, "%.*s.%d%s", stem_len, path, frame_idx, ext);
    return n < 0 || (size_t)n >= out_size;
}

static int taehv_write_ppm_frames(const char *path, const unsigned char *rgb, int frame_count, int width, int height, int frame_offset) {
    size_t frame_bytes = (size_t)width * height * 3;
    if (frame_offset == 0) {
        if (taehv_write_ppm(path, rgb, width, height)) return 1;
        fprintf(stderr, "wrote RGB image: %s\n", path);
    }
    for (int i = 0; i < frame_count; ++i) {
        char frame_path[4096];
        int global_frame = frame_offset + i;
        if (taehv_make_frame_path(frame_path, sizeof(frame_path), path, global_frame)) {
            fprintf(stderr, "output frame path too long for %s frame %d\n", path, global_frame);
            return 1;
        }
        if (taehv_write_ppm(frame_path, rgb + (size_t)i * frame_bytes, width, height)) return 1;
        fprintf(stderr, "wrote RGB frame %d: %s\n", global_frame, frame_path);
    }
    return 0;
}

static int world_cuda_decode_vae_to_rgb_h_nhwc(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *d_latent,
        const unsigned char **rgb_out,
        int *frame_count_out,
        int *width_out,
        int *height_out) {
    if (!dec || !dec->hbuf0 || !dec->fp16_nhwc_enabled) return 1;

    fprintf(stderr, "VAE decode FP16/NHWC streaming: latent [%d,%d,%d] -> RGB %dx%d\n",
            cfg->channels, cfg->height * cfg->patch_h, cfg->width * cfg->patch_w, dec->out_w, dec->out_h);
    taehv_profile_reset(dec);
    if (!dec->stream_started_h) {
        for (int i = 0; i < 3; ++i) {
            if (taehv_stream_decode_one_h_nhwc(cfg, dec, d_latent, 0)) return 1;
        }
        dec->stream_started_h = 1;
    }
    if (taehv_stream_decode_one_h_nhwc(cfg, dec, d_latent, 1)) return 1;
    CUDA_OK(cudaMemcpy(dec->h_rgb, dec->d_rgb, dec->rgb_elems, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());
    taehv_profile_print(dec);
    if (rgb_out) *rgb_out = dec->h_rgb;
    if (frame_count_out) *frame_count_out = 4;
    if (width_out) *width_out = dec->out_w;
    if (height_out) *height_out = dec->out_h;
    return 0;
}

static int world_cuda_decode_vae_to_rgb(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *d_latent,
        const unsigned char **rgb_out,
        int *frame_count_out,
        int *width_out,
        int *height_out) {
    if (!dec || !dec->buf0) return 1;
    if (dec->fp16_nhwc_enabled) {
        if (world_cuda_decode_vae_to_rgb_h_nhwc(cfg, dec, d_latent, rgb_out, frame_count_out, width_out, height_out) == 0) {
            return 0;
        }
        fprintf(stderr, "VAE FP16/NHWC runtime failed, falling back to F32/NCHW\n");
        dec->fp16_nhwc_enabled = 0;
    }

    fprintf(stderr, "VAE decode streaming: latent [%d,%d,%d] -> RGB %dx%d\n",
            cfg->channels, cfg->height * cfg->patch_h, cfg->width * cfg->patch_w, dec->out_w, dec->out_h);
    taehv_profile_reset(dec);
    if (!dec->stream_started_f32) {
        for (int i = 0; i < 3; ++i) {
            if (taehv_stream_decode_one_f32(cfg, dec, d_latent, 0)) return 1;
        }
        dec->stream_started_f32 = 1;
    }
    if (taehv_stream_decode_one_f32(cfg, dec, d_latent, 1)) return 1;
    CUDA_OK(cudaMemcpy(dec->h_rgb, dec->d_rgb, dec->rgb_elems, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());
    taehv_profile_print(dec);
    if (rgb_out) *rgb_out = dec->h_rgb;
    if (frame_count_out) *frame_count_out = 4;
    if (width_out) *width_out = dec->out_w;
    if (height_out) *height_out = dec->out_h;
    return 0;
}

static int world_cuda_decode_vae_to_ppm(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *d_latent,
        const char *out_path,
        int frame_offset) {
    if (!dec || !dec->buf0 || !out_path || !out_path[0]) return 0;
    const unsigned char *rgb = NULL;
    int frames = 0;
    int width = 0;
    int height = 0;
    if (world_cuda_decode_vae_to_rgb(cfg, dec, d_latent, &rgb, &frames, &width, &height)) return 1;
    if (taehv_write_ppm_frames(out_path, rgb, frames, width, height, frame_offset)) return 1;
    return 0;
}

extern "C" int world_cuda_vae_decode_probe(
        const WorldConfig *cfg,
        const float *latent,
        const WorldVaeDecoderWeights *vae,
        const char *out_path) {
    if (!cfg || !latent || !vae || !out_path || !out_path[0]) return 1;
    DeviceVaeDecoder d_vae;
    memset(&d_vae, 0, sizeof(d_vae));
    float *d_latent = NULL;
    int rc = 1;
    size_t latent_elems = (size_t)cfg->channels *
                          (size_t)(cfg->height * cfg->patch_h) *
                          (size_t)(cfg->width * cfg->patch_w);

    if (cudaMalloc((void **)&d_latent, latent_elems * sizeof(float)) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_latent, latent, latent_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (taehv_decoder_init(&d_vae, cfg, vae)) goto cleanup;
    if (world_cuda_decode_vae_to_ppm(cfg, &d_vae, d_latent, out_path, 0)) goto cleanup;
    rc = 0;

cleanup:
    taehv_decoder_free(&d_vae);
    cudaFree(d_latent);
    return rc;
}

extern "C" int world_cuda_vae_decode_sequence_probe(
        const WorldConfig *cfg,
        const float *latents,
        int latent_count,
        const WorldVaeDecoderWeights *vae,
        const char *out_path) {
    if (!cfg || !latents || latent_count <= 0 || !vae || !out_path || !out_path[0]) return 1;
    DeviceVaeDecoder d_vae;
    memset(&d_vae, 0, sizeof(d_vae));
    float *d_latent = NULL;
    int rc = 1;
    size_t latent_elems = (size_t)cfg->channels *
                          (size_t)(cfg->height * cfg->patch_h) *
                          (size_t)(cfg->width * cfg->patch_w);

    if (cudaMalloc((void **)&d_latent, latent_elems * sizeof(float)) != cudaSuccess) goto cleanup;
    if (taehv_decoder_init(&d_vae, cfg, vae)) goto cleanup;
    for (int i = 0; i < latent_count; ++i) {
        const float *src = latents + (size_t)i * latent_elems;
        if (cudaMemcpy(d_latent, src, latent_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
        if (world_cuda_decode_vae_to_ppm(cfg, &d_vae, d_latent, out_path, i * 4)) goto cleanup;
    }
    rc = 0;

cleanup:
    taehv_decoder_free(&d_vae);
    cudaFree(d_latent);
    return rc;
}

struct WorldCudaRuntime {
    WorldConfig cfg;
    int layers_to_run;
    int steps_to_run;
    int next_frame_idx;
    unsigned int seed;
    int noise_mode;
    int frame_ordinal;
    int C;
    int H;
    int W;
    int D;
    int ph;
    int pw;
    int T;
    int d_head;
    int kv_dim;
    int mlp_hidden;
    int ctrl_dim;
    int d_xy;
    int d_t;
    float rms_eps;
    int total_passes;
    int precomputed_total_passes;
    int forced_pass_table_idx;
    int frame_stride;
    int mlp_fc2_splitk_slices;
    int mlp_fc2_splitk_parallel_enabled;
    int fp16_gemm_m64n64_enabled;
    int half_gemm_boundary_enabled;
    int mlp_fc1_silu_epilogue_enabled;
    int w8a8_mask;
    int w8a8_drop_fallback;
    int w8a8_layer_begin;
    int w8a8_layer_end;
    size_t latent_elems;
    size_t token_elems;
    size_t kv_rope_elems;
    size_t q_rope_elems;
    size_t linear_half_elems;
    size_t w8a8_x_elems;
    size_t w8a8_acc_elems;
    size_t splitk_workspace_bytes;
    const float *h_latent_override;
    float *h_latent;
    float *h_noise;
    float *h_xy;
    float *h_inv_t;
    int64_t *h_x_pos;
    int64_t *h_y_pos;
    int64_t *h_t_pos;
    float *d_latent;
    float *d_patch;
    float *d_patch_rows;
    float *d_noise;
    float *d_noise_hidden;
    float *d_cond;
    float *d_cond_act;
    float *d_denoise_fc1;
    float *d_denoise_fc2;
    float *d_control_input;
    float *d_ctrl_emb_fc1_w;
    float *d_ctrl_emb_fc2_w;
    float *d_ctrl_emb_hidden;
    float *d_ctrl_emb;
    float *d_ctrl_emb_norm;
    float *d_out_norm_w;
    float *d_out_mod_table;
    float *d_final_tokens;
    float *d_unpatch_w;
    float *d_unpatch_b;
    float *d_latent_out;
    float *d_layer_mod_table;
    float *d_tokens;
    float *d_norm;
    float *d_qkv_raw;
    float *d_q;
    float *d_k;
    float *d_v;
    float *d_v_first;
    float *d_attn;
    float *d_attn_out;
    float *d_tokens_after_attn;
    float *d_ctrl_norm;
    float *d_ctrl_cond_by_layer;
    float *d_ctrl_hidden;
    float *d_ctrl_out;
    float *d_tokens_after_ctrl;
    float *d_mlp_in;
    float *d_mlp_hidden;
    float *d_mlp_out;
    float *d_tokens_after_mlp;
    float *d_xy_table;
    float *d_inv_t;
    int64_t *d_x_pos;
    int64_t *d_y_pos;
    int64_t *d_t_pos;
    __half *d_linear_half;
    __half *d_mlp_hidden_half;
    int8_t *d_w8a8_x;
    float *d_w8a8_x_scales;
    int32_t *d_w8a8_acc;
    void *d_splitk_workspace;
    int attn_cutlass_enabled;
    int attn_cutlass_grouped_enabled;
    int attn_cutlass_fmha_enabled;
    int attn_sparse_fmha_enabled;
    int attn_max_capacity;
    __half *d_attn_q_half;
    __half *d_attn_k_compact;
    __half *d_attn_v_compact;
    __half *d_attn_out_half;
    float *d_attn_scores;
    __half *d_attn_probs_half;
    int attn_flash_enabled;
    int attn_q4_shared_enabled;
    int attn_half_cache_enabled;
    int attn_half_flash_enabled;
    DeviceWorldLayerWeights *d_layers;
    DeviceWorldLayerCache *d_caches;
    cudaEvent_t ev_step_start;
    cudaEvent_t ev_after_setup;
    cudaEvent_t ev_after_transformer;
    cudaEvent_t ev_after_vae;
    int profile_enabled;
    cudaEvent_t prof_start;
    cudaEvent_t prof_stop;
    float prof_patch_ms;
    float prof_norm_ms;
    float prof_qkv_gemm_ms;
    float prof_qkv_rope_ms;
    float prof_cache_ms;
    float prof_attn_ms;
    float prof_attn_out_gemm_ms;
    float prof_attn_residual_ms;
    float prof_ctrl_ms;
    float prof_mlp_fc1_ms;
    float prof_mlp_silu_ms;
    float prof_mlp_fc2_ms;
    float prof_mlp_residual_ms;
    float prof_out_ms;
    int prof_patch_calls;
    int prof_norm_calls;
    int prof_qkv_gemm_calls;
    int prof_qkv_rope_calls;
    int prof_cache_calls;
    int prof_attn_calls;
    int prof_attn_out_gemm_calls;
    int prof_attn_residual_calls;
    int prof_ctrl_calls;
    int prof_mlp_fc1_calls;
    int prof_mlp_silu_calls;
    int prof_mlp_fc2_calls;
    int prof_mlp_residual_calls;
    int prof_out_calls;
    DeviceVaeDecoder d_vae;
};

static int precompute_runtime_layer_mods(WorldCudaRuntime *rt) {
    if (!rt || !rt->d_layer_mod_table || !rt->d_out_mod_table) return 1;
    const WorldConfig *cfg = &rt->cfg;
    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= rt->steps_to_run;
        float sigma_step = is_cache_pass ? 0.0f : cfg->scheduler_sigmas[pass_idx];

        fill_noise_embedding(rt->h_noise, sigma_step);
        CUDA_OK(cudaMemcpy(rt->d_noise, rt->h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
        if (row_major_linear(rt->d_noise, rt->d_denoise_fc1, rt->d_noise_hidden, 1, 512, rt->mlp_hidden)) return 1;
        silu_f32_kernel<<<div_up_i64(rt->mlp_hidden, 256), 256>>>(rt->d_noise_hidden, rt->d_noise_hidden, rt->mlp_hidden);
        CUDA_OK(cudaGetLastError());
        if (row_major_linear(rt->d_noise_hidden, rt->d_denoise_fc2, rt->d_cond, 1, rt->mlp_hidden, rt->D)) return 1;
        silu_f32_kernel<<<div_up_i64(rt->D, 256), 256>>>(rt->d_cond, rt->d_cond_act, rt->D);
        CUDA_OK(cudaGetLastError());
        if (row_major_linear(
                    rt->d_cond_act,
                    rt->d_out_norm_w,
                    rt->d_out_mod_table + (int64_t)pass_idx * 2 * rt->D,
                    1,
                    rt->D,
                    2 * rt->D)) return 1;

        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            const DeviceWorldLayerWeights *lw = &rt->d_layers[layer_idx];
            float *dst = rt->d_layer_mod_table + ((int64_t)pass_idx * rt->layers_to_run + layer_idx) * (6 * rt->D);
            add_bias_silu_f32_kernel<<<div_up_i64(rt->D, 256), 256>>>(rt->d_cond, lw->cond_bias, rt->d_cond_act, rt->D);
            CUDA_OK(cudaGetLastError());
            if (row_major_linear(rt->d_cond_act, lw->cond_proj_weight, dst, 1, rt->D, 6 * rt->D)) return 1;
        }
    }
    CUDA_OK(cudaGetLastError());
    return 0;
}

static void release_runtime_precompute_and_w8a8_fallback_weights(WorldCudaRuntime *rt) {
    if (!rt || !rt->d_layers) return;
    size_t released_bytes = 0;
    size_t omitted_bytes = 0;
    for (int i = 0; i < rt->layers_to_run; ++i) {
        DeviceWorldLayerWeights *lw = &rt->d_layers[i];

        if (lw->cond_proj_weight) {
            cudaFree(lw->cond_proj_weight);
            lw->cond_proj_weight = NULL;
            released_bytes += (size_t)6 * rt->D * rt->D * sizeof(float);
        }

        if (!rt->w8a8_drop_fallback) continue;
        if ((rt->w8a8_mask & WORLD_W8A8_QKV) && lw->qkv_proj_weight_i8) {
            size_t elems = (size_t)(rt->D + 2 * rt->kv_dim) * rt->D;
            if (lw->qkv_proj_weight) {
                cudaFree(lw->qkv_proj_weight);
                released_bytes += elems * sizeof(float);
            } else {
                omitted_bytes += elems * sizeof(float);
            }
            if (lw->qkv_proj_weight_h) {
                cudaFree(lw->qkv_proj_weight_h);
                released_bytes += elems * sizeof(__half);
            } else {
                omitted_bytes += elems * sizeof(__half);
            }
            lw->qkv_proj_weight = NULL;
            lw->qkv_proj_weight_h = NULL;
        }
        if ((rt->w8a8_mask & WORLD_W8A8_OUT) && lw->out_proj_weight_i8) {
            size_t elems = (size_t)rt->D * rt->D;
            if (lw->out_proj_weight) {
                cudaFree(lw->out_proj_weight);
                released_bytes += elems * sizeof(float);
            } else {
                omitted_bytes += elems * sizeof(float);
            }
            if (lw->out_proj_weight_h) {
                cudaFree(lw->out_proj_weight_h);
                released_bytes += elems * sizeof(__half);
            } else {
                omitted_bytes += elems * sizeof(__half);
            }
            lw->out_proj_weight = NULL;
            lw->out_proj_weight_h = NULL;
        }
        if ((rt->w8a8_mask & WORLD_W8A8_CTRL) && lw->has_ctrl &&
                lw->ctrl_fc1_x_weight_i8 && lw->ctrl_fc2_weight_i8) {
            size_t elems = (size_t)rt->D * rt->D;
            float *f32_weights[] = {lw->ctrl_fc1_x_weight, lw->ctrl_fc2_weight};
            __half *f16_weights[] = {lw->ctrl_fc1_x_weight_h, lw->ctrl_fc2_weight_h};
            for (int j = 0; j < 2; ++j) {
                if (f32_weights[j]) {
                    cudaFree(f32_weights[j]);
                    released_bytes += elems * sizeof(float);
                } else {
                    omitted_bytes += elems * sizeof(float);
                }
                if (f16_weights[j]) {
                    cudaFree(f16_weights[j]);
                    released_bytes += elems * sizeof(__half);
                } else {
                    omitted_bytes += elems * sizeof(__half);
                }
            }
            lw->ctrl_fc1_x_weight = NULL;
            lw->ctrl_fc1_x_weight_h = NULL;
            lw->ctrl_fc2_weight = NULL;
            lw->ctrl_fc2_weight_h = NULL;
        }
        if ((rt->w8a8_mask & WORLD_W8A8_MLP) &&
                lw->dit_mlp_fc1_weight_i8 && lw->dit_mlp_fc2_weight_i8) {
            size_t elems = (size_t)rt->D * rt->mlp_hidden;
            float *f32_weights[] = {lw->dit_mlp_fc1_weight, lw->dit_mlp_fc2_weight};
            __half *f16_weights[] = {lw->dit_mlp_fc1_weight_h, lw->dit_mlp_fc2_weight_h};
            for (int j = 0; j < 2; ++j) {
                if (f32_weights[j]) {
                    cudaFree(f32_weights[j]);
                    released_bytes += elems * sizeof(float);
                } else {
                    omitted_bytes += elems * sizeof(float);
                }
                if (f16_weights[j]) {
                    cudaFree(f16_weights[j]);
                    released_bytes += elems * sizeof(__half);
                } else {
                    omitted_bytes += elems * sizeof(__half);
                }
            }
            lw->dit_mlp_fc1_weight = NULL;
            lw->dit_mlp_fc1_weight_h = NULL;
            lw->dit_mlp_fc2_weight = NULL;
            lw->dit_mlp_fc2_weight_h = NULL;
        }
    }
    fprintf(stderr,
            "released %.2f GiB of init-only/fallback layer weights; "
            "skipped %.2f GiB of W8A8 FP fallback allocations\n",
            (double)released_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)omitted_bytes / (1024.0 * 1024.0 * 1024.0));
}

static int preflight_runtime_w8a8(WorldCudaRuntime *rt) {
    if (!rt || !rt->w8a8_mask) return 0;
    int checked_qkv = 0;
    int checked_out = 0;
    int checked_ctrl = 0;
    int checked_mlp = 0;
    for (int i = 0; i < rt->layers_to_run; ++i) {
        const DeviceWorldLayerWeights *lw = &rt->d_layers[i];
        if (!checked_qkv && lw->qkv_proj_weight_i8) {
            if (row_major_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->qkv_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D + 2 * rt->kv_dim)) return 1;
            checked_qkv = 1;
        }
        if (!checked_out && lw->out_proj_weight_i8) {
            if (row_major_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->out_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            checked_out = 1;
        }
        if (!checked_ctrl && lw->ctrl_fc1_x_weight_i8 && lw->ctrl_fc2_weight_i8) {
            if (row_major_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->ctrl_fc1_x_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            if (row_major_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->ctrl_fc2_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            checked_ctrl = 1;
        }
        if (!checked_mlp && lw->dit_mlp_fc1_weight_i8 && lw->dit_mlp_fc2_weight_i8) {
            if (row_major_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->dit_mlp_fc1_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->mlp_hidden)) return 1;
            if (row_major_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->dit_mlp_fc2_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->mlp_hidden,
                    rt->D)) return 1;
            checked_mlp = 1;
        }
    }
    fprintf(stderr, "W8A8 CUTLASS shape/layout preflight passed\n");
    return 0;
}

static double monotonic_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    LARGE_INTEGER now;
    if (!have_freq) {
        QueryPerformanceFrequency(&freq);
        have_freq = 1;
    }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
#endif
}

static void runtime_profile_reset(WorldCudaRuntime *rt) {
    if (!rt || !rt->profile_enabled) return;
    rt->prof_patch_ms = 0.0f;
    rt->prof_norm_ms = 0.0f;
    rt->prof_qkv_gemm_ms = 0.0f;
    rt->prof_qkv_rope_ms = 0.0f;
    rt->prof_cache_ms = 0.0f;
    rt->prof_attn_ms = 0.0f;
    rt->prof_attn_out_gemm_ms = 0.0f;
    rt->prof_attn_residual_ms = 0.0f;
    rt->prof_ctrl_ms = 0.0f;
    rt->prof_mlp_fc1_ms = 0.0f;
    rt->prof_mlp_silu_ms = 0.0f;
    rt->prof_mlp_fc2_ms = 0.0f;
    rt->prof_mlp_residual_ms = 0.0f;
    rt->prof_out_ms = 0.0f;
    rt->prof_patch_calls = 0;
    rt->prof_norm_calls = 0;
    rt->prof_qkv_gemm_calls = 0;
    rt->prof_qkv_rope_calls = 0;
    rt->prof_cache_calls = 0;
    rt->prof_attn_calls = 0;
    rt->prof_attn_out_gemm_calls = 0;
    rt->prof_attn_residual_calls = 0;
    rt->prof_ctrl_calls = 0;
    rt->prof_mlp_fc1_calls = 0;
    rt->prof_mlp_silu_calls = 0;
    rt->prof_mlp_fc2_calls = 0;
    rt->prof_mlp_residual_calls = 0;
    rt->prof_out_calls = 0;
}

static int runtime_profile_begin(WorldCudaRuntime *rt) {
    if (!rt || !rt->profile_enabled) return 0;
    CUDA_OK(cudaEventRecord(rt->prof_start, 0));
    return 0;
}

static int runtime_profile_accum(WorldCudaRuntime *rt, float *accum, int *calls) {
    if (!rt || !rt->profile_enabled) return 0;
    float ms = 0.0f;
    CUDA_OK(cudaEventRecord(rt->prof_stop, 0));
    CUDA_OK(cudaEventSynchronize(rt->prof_stop));
    CUDA_OK(cudaEventElapsedTime(&ms, rt->prof_start, rt->prof_stop));
    *accum += ms;
    *calls += 1;
    return 0;
}

static void runtime_profile_print(const WorldCudaRuntime *rt) {
    if (!rt || !rt->profile_enabled) return;
    fprintf(stderr,
            "transformer profile: patch=%.3fms/%d norm=%.3fms/%d qkv_gemm=%.3fms/%d qkv_rope=%.3fms/%d cache=%.3fms/%d attn=%.3fms/%d attn_out_gemm=%.3fms/%d attn_residual=%.3fms/%d ctrl=%.3fms/%d mlp_fc1=%.3fms/%d mlp_silu=%.3fms/%d mlp_fc2=%.3fms/%d mlp_residual=%.3fms/%d out=%.3fms/%d\n",
            rt->prof_patch_ms, rt->prof_patch_calls,
            rt->prof_norm_ms, rt->prof_norm_calls,
            rt->prof_qkv_gemm_ms, rt->prof_qkv_gemm_calls,
            rt->prof_qkv_rope_ms, rt->prof_qkv_rope_calls,
            rt->prof_cache_ms, rt->prof_cache_calls,
            rt->prof_attn_ms, rt->prof_attn_calls,
            rt->prof_attn_out_gemm_ms, rt->prof_attn_out_gemm_calls,
            rt->prof_attn_residual_ms, rt->prof_attn_residual_calls,
            rt->prof_ctrl_ms, rt->prof_ctrl_calls,
            rt->prof_mlp_fc1_ms, rt->prof_mlp_fc1_calls,
            rt->prof_mlp_silu_ms, rt->prof_mlp_silu_calls,
            rt->prof_mlp_fc2_ms, rt->prof_mlp_fc2_calls,
            rt->prof_mlp_residual_ms, rt->prof_mlp_residual_calls,
            rt->prof_out_ms, rt->prof_out_calls);
}

extern "C" void world_cuda_runtime_destroy(WorldCudaRuntime *rt) {
    if (!rt) return;
    taehv_decoder_free(&rt->d_vae);
    if (rt->prof_stop) cudaEventDestroy(rt->prof_stop);
    if (rt->prof_start) cudaEventDestroy(rt->prof_start);
    if (rt->ev_after_vae) cudaEventDestroy(rt->ev_after_vae);
    if (rt->ev_after_transformer) cudaEventDestroy(rt->ev_after_transformer);
    if (rt->ev_after_setup) cudaEventDestroy(rt->ev_after_setup);
    if (rt->ev_step_start) cudaEventDestroy(rt->ev_step_start);
    free_device_world_layers(rt->d_layers, rt->layers_to_run);
    free_device_world_caches(rt->d_caches, rt->layers_to_run);
    cudaFree(rt->d_latent);
    cudaFree(rt->d_patch);
    cudaFree(rt->d_patch_rows);
    cudaFree(rt->d_noise);
    cudaFree(rt->d_noise_hidden);
    cudaFree(rt->d_cond);
    cudaFree(rt->d_cond_act);
    cudaFree(rt->d_denoise_fc1);
    cudaFree(rt->d_denoise_fc2);
    cudaFree(rt->d_control_input);
    cudaFree(rt->d_ctrl_emb_fc1_w);
    cudaFree(rt->d_ctrl_emb_fc2_w);
    cudaFree(rt->d_ctrl_emb_hidden);
    cudaFree(rt->d_ctrl_emb);
    cudaFree(rt->d_ctrl_emb_norm);
    cudaFree(rt->d_out_norm_w);
    cudaFree(rt->d_out_mod_table);
    cudaFree(rt->d_final_tokens);
    cudaFree(rt->d_unpatch_w);
    cudaFree(rt->d_unpatch_b);
    cudaFree(rt->d_latent_out);
    cudaFree(rt->d_layer_mod_table);
    cudaFree(rt->d_tokens);
    cudaFree(rt->d_norm);
    cudaFree(rt->d_qkv_raw);
    cudaFree(rt->d_q);
    cudaFree(rt->d_k);
    cudaFree(rt->d_v);
    cudaFree(rt->d_v_first);
    cudaFree(rt->d_attn);
    cudaFree(rt->d_attn_out);
    cudaFree(rt->d_tokens_after_attn);
    cudaFree(rt->d_ctrl_norm);
    cudaFree(rt->d_ctrl_cond_by_layer);
    cudaFree(rt->d_ctrl_hidden);
    cudaFree(rt->d_ctrl_out);
    cudaFree(rt->d_tokens_after_ctrl);
    cudaFree(rt->d_mlp_in);
    cudaFree(rt->d_mlp_hidden);
    cudaFree(rt->d_mlp_out);
    cudaFree(rt->d_tokens_after_mlp);
    cudaFree(rt->d_xy_table);
    cudaFree(rt->d_inv_t);
    cudaFree(rt->d_x_pos);
    cudaFree(rt->d_y_pos);
    cudaFree(rt->d_t_pos);
    cudaFree(rt->d_linear_half);
    cudaFree(rt->d_mlp_hidden_half);
    cudaFree(rt->d_w8a8_x);
    cudaFree(rt->d_w8a8_x_scales);
    cudaFree(rt->d_w8a8_acc);
    cudaFree(rt->d_splitk_workspace);
    cudaFree(rt->d_attn_q_half);
    cudaFree(rt->d_attn_k_compact);
    cudaFree(rt->d_attn_v_compact);
    cudaFree(rt->d_attn_out_half);
    cudaFree(rt->d_attn_scores);
    cudaFree(rt->d_attn_probs_half);
    free(rt->h_latent);
    free(rt->h_noise);
    free(rt->h_xy);
    free(rt->h_inv_t);
    free(rt->h_x_pos);
    free(rt->h_y_pos);
    free(rt->h_t_pos);
    free(rt);
}

extern "C" int world_cuda_runtime_create(
        WorldCudaRuntime **out,
        const WorldConfig *cfg,
        const WorldModelProbeWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frame_idx,
        unsigned int seed,
        int noise_mode,
        const WorldVaeDecoderWeights *vae) {
    if (!out || !cfg || !weights || !weights->layers || !vae) return 1;
    *out = NULL;
    if (world_config_validate(cfg)) return 1;
    if (layers_to_run <= 0 || layers_to_run > weights->n_layers) {
        fprintf(stderr, "invalid runtime layers_to_run=%d n_layers=%d\n", layers_to_run, weights->n_layers);
        return 1;
    }
    if (steps_to_run <= 0 || steps_to_run >= cfg->scheduler_sigmas_count) {
        fprintf(stderr, "invalid runtime steps_to_run=%d scheduler_count=%d\n", steps_to_run, cfg->scheduler_sigmas_count);
        return 1;
    }
    if (frame_idx < 0) {
        fprintf(stderr, "invalid runtime frame_idx=%d\n", frame_idx);
        return 1;
    }

    WorldCudaRuntime *rt = (WorldCudaRuntime *)calloc(1, sizeof(*rt));
    if (!rt) return 1;
    rt->cfg = *cfg;
    rt->layers_to_run = layers_to_run;
    rt->steps_to_run = steps_to_run;
    rt->next_frame_idx = frame_idx;
    rt->seed = seed;
    rt->noise_mode = noise_mode;
    rt->C = cfg->channels;
    rt->H = cfg->height * cfg->patch_h;
    rt->W = cfg->width * cfg->patch_w;
    rt->D = cfg->d_model;
    rt->ph = cfg->patch_h;
    rt->pw = cfg->patch_w;
    rt->T = cfg->height * cfg->width;
    rt->d_head = rt->D / cfg->n_heads;
    rt->kv_dim = cfg->n_kv_heads * rt->d_head;
    rt->mlp_hidden = rt->D * cfg->mlp_ratio;
    rt->ctrl_dim = cfg->n_buttons + 3;
    rt->d_xy = rt->d_head / 8;
    rt->d_t = rt->d_head / 4;
    rt->rms_eps = 1.0e-6f;
    {
        const char *rms_eps_env = getenv("WORLD_RMS_EPS");
        if (rms_eps_env && rms_eps_env[0]) {
            float v = (float)atof(rms_eps_env);
            if (v > 0.0f) rt->rms_eps = v;
        }
        fprintf(stderr, "runtime RMSNorm eps: %.8g\n", rt->rms_eps);
    }
    {
        const char *profile_env = getenv("WORLD_TRANSFORMER_PROFILE");
        rt->profile_enabled = profile_env ? profile_env[0] != '0' : 0;
        if (rt->profile_enabled) {
            fprintf(stderr, "transformer profiling enabled by WORLD_TRANSFORMER_PROFILE=1\n");
        }
    }
    {
        const char *enable_env = getenv("WORLD_W8A8");
        const char *ops_env = getenv("WORLD_W8A8_OPS");
        int enabled = enable_env && enable_env[0] &&
            strcmp(enable_env, "0") != 0 && strcmp(enable_env, "off") != 0 &&
            strcmp(enable_env, "none") != 0;
        if (enabled) {
            if (ops_env && ops_env[0]) {
                rt->w8a8_mask = parse_w8a8_ops(ops_env);
            } else if (strcmp(enable_env, "1") == 0) {
                // The M=128 attention out projection is below the crossover
                // point on Ada; its dynamic A8 boundary costs more than the
                // INT8 GEMM saves.  Keep it enabled for the M=512 main model.
                rt->w8a8_mask = WORLD_W8A8_ALL;
                if (rt->T <= 256) rt->w8a8_mask &= ~WORLD_W8A8_OUT;
            } else {
                rt->w8a8_mask = parse_w8a8_ops(enable_env);
            }
            if (rt->w8a8_mask < 0) {
                world_cuda_runtime_destroy(rt);
                return 1;
            }
            if (rt->w8a8_mask == 0) {
                fprintf(stderr, "W8A8 disabled: no operations selected\n");
                enabled = 0;
            }
        }
        if (enabled) {
            int device = 0;
            cudaDeviceProp prop;
            cudaError_t device_err = cudaGetDevice(&device);
            if (device_err == cudaSuccess) {
                device_err = cudaGetDeviceProperties(&prop, device);
            }
            if (device_err != cudaSuccess) {
                fprintf(stderr, "W8A8 device query failed: %s\n", cudaGetErrorString(device_err));
                world_cuda_runtime_destroy(rt);
                return 1;
            }
            if (prop.major < 8) {
                fprintf(stderr,
                        "W8A8 requires SM80+; detected %s compute capability %d.%d\n",
                        prop.name, prop.major, prop.minor);
                world_cuda_runtime_destroy(rt);
                return 1;
            }
            const char *drop_env = getenv("WORLD_W8A8_DROP_FALLBACK");
            rt->w8a8_drop_fallback = drop_env ? drop_env[0] != '0' : 1;
            const char *layer_begin_env = getenv("WORLD_W8A8_LAYER_BEGIN");
            const char *layer_end_env = getenv("WORLD_W8A8_LAYER_END");
            rt->w8a8_layer_begin = layer_begin_env && layer_begin_env[0]
                ? atoi(layer_begin_env) : 0;
            rt->w8a8_layer_end = layer_end_env && layer_end_env[0]
                ? atoi(layer_end_env) : layers_to_run;
            if (rt->w8a8_layer_begin < 0) rt->w8a8_layer_begin = 0;
            if (rt->w8a8_layer_end > layers_to_run) rt->w8a8_layer_end = layers_to_run;
            if (rt->w8a8_layer_end <= rt->w8a8_layer_begin) {
                fprintf(stderr,
                        "invalid W8A8 layer range [%d,%d) for %d layers\n",
                        rt->w8a8_layer_begin, rt->w8a8_layer_end, layers_to_run);
                world_cuda_runtime_destroy(rt);
                return 1;
            }
            fprintf(stderr,
                    "W8A8 enabled: ops=%s%s%s%s layers=[%d,%d) fallback_weights=%s\n",
                    (rt->w8a8_mask & WORLD_W8A8_MLP) ? "mlp," : "",
                    (rt->w8a8_mask & WORLD_W8A8_QKV) ? "qkv," : "",
                    (rt->w8a8_mask & WORLD_W8A8_OUT) ? "out," : "",
                    (rt->w8a8_mask & WORLD_W8A8_CTRL) ? "ctrl" : "",
                    rt->w8a8_layer_begin,
                    rt->w8a8_layer_end,
                    rt->w8a8_drop_fallback ? "drop-after-init" : "keep");
            fprintf(stderr,
                    "W8A8 warning: experimental row/channel-wise PTQ; "
                    "validate long autoregressive rollouts before deployment\n");
        }
    }
    {
        const char *tile_env = getenv("WORLD_FP16_GEMM_TILE");
        rt->fp16_gemm_m64n64_enabled =
            !(tile_env && tile_env[0] && (strcmp(tile_env, "base") == 0 || strcmp(tile_env, "128x128") == 0));
        if (rt->fp16_gemm_m64n64_enabled) {
            fprintf(stderr, "FP16 tensor-op GEMM small-M tile: m64n64\n");
        } else {
            fprintf(stderr, "FP16 tensor-op GEMM small-M tile disabled by WORLD_FP16_GEMM_TILE=base\n");
        }
    }
    {
        const char *boundary_env = getenv("WORLD_HALF_GEMM_BOUNDARY");
        rt->half_gemm_boundary_enabled = boundary_env ? boundary_env[0] != '0' : 1;
        if (rt->half_gemm_boundary_enabled) {
            fprintf(stderr, "FP16 activation boundaries enabled for supported CUTLASS GEMMs\n");
        }
    }
    {
        const char *splitk_env = getenv("WORLD_MLP_FC2_SPLITK");
        const char *splitk_parallel_env = getenv("WORLD_MLP_FC2_SPLITK_PARALLEL");
        if (splitk_env && splitk_env[0]) {
            rt->mlp_fc2_splitk_slices = atoi(splitk_env);
        } else if (rt->T <= 256 && rt->mlp_hidden >= 8192 && rt->D >= 2048) {
            rt->mlp_fc2_splitk_slices = 4;
        } else {
            rt->mlp_fc2_splitk_slices = 1;
        }
        if (rt->mlp_fc2_splitk_slices < 1) rt->mlp_fc2_splitk_slices = 1;
        if (rt->mlp_fc2_splitk_slices > 32) rt->mlp_fc2_splitk_slices = 32;
        rt->mlp_fc2_splitk_parallel_enabled = splitk_parallel_env ? splitk_parallel_env[0] != '0' : 0;
        if (rt->mlp_fc2_splitk_parallel_enabled && rt->mlp_fc2_splitk_slices <= 1) {
            fprintf(stderr, "WORLD_MLP_FC2_SPLITK_PARALLEL ignored because split-K is disabled\n");
            rt->mlp_fc2_splitk_parallel_enabled = 0;
        }
        if (rt->mlp_fc2_splitk_slices > 1) {
            fprintf(stderr,
                    "MLP fc2 CUTLASS %ssplit-K enabled: slices=%d\n",
                    rt->mlp_fc2_splitk_parallel_enabled ? "parallel " : "",
                    rt->mlp_fc2_splitk_slices);
        }
    }
    {
        const char *epilogue_env = getenv("WORLD_MLP_FC1_SILU_EPILOGUE");
        rt->mlp_fc1_silu_epilogue_enabled = epilogue_env ? epilogue_env[0] != '0' : 1;
        if (rt->mlp_fc1_silu_epilogue_enabled) {
            fprintf(stderr, "MLP fc1 CUTLASS SiLU-to-half epilogue enabled\n");
        }
    }
    rt->total_passes = steps_to_run + 1;
    rt->precomputed_total_passes = rt->total_passes;
    rt->forced_pass_table_idx = -1;
    int fps_div = cfg->temporal_compression > 0 ? cfg->inference_fps / cfg->temporal_compression : 0;
    rt->frame_stride = fps_div > 0 ? cfg->base_fps / fps_div : 1;
    if (rt->frame_stride <= 0) rt->frame_stride = 1;
    rt->latent_elems = (size_t)rt->C * rt->H * rt->W;
    rt->token_elems = (size_t)rt->T * rt->D;
    rt->q_rope_elems = (size_t)cfg->n_heads * rt->T * rt->d_head;
    rt->kv_rope_elems = (size_t)cfg->n_kv_heads * rt->T * rt->d_head;
    rt->linear_half_elems = rt->token_elems;
    if ((size_t)rt->T * rt->mlp_hidden > rt->linear_half_elems) {
        rt->linear_half_elems = (size_t)rt->T * rt->mlp_hidden;
    }
    rt->w8a8_x_elems = (size_t)rt->T *
        (size_t)(rt->mlp_hidden > rt->D ? rt->mlp_hidden : rt->D);
    {
        int max_n = rt->D + 2 * rt->kv_dim;
        if (rt->mlp_hidden > max_n) max_n = rt->mlp_hidden;
        rt->w8a8_acc_elems = (size_t)rt->T * max_n;
    }
    size_t qkv_token_elems = rt->token_elems + 2 * ((size_t)rt->T * rt->kv_dim);
    size_t patch_weight_elems = (size_t)rt->D * rt->C * rt->ph * rt->pw;
    size_t patch_row_elems = (size_t)rt->T * rt->C * rt->ph * rt->pw;
    size_t out_norm_weight_elems = (size_t)2 * rt->D * rt->D;
    size_t unpatch_weight_elems = (size_t)rt->D * rt->C * rt->ph * rt->pw;
    size_t layer_mod_table_elems = (size_t)rt->total_passes * layers_to_run * 6 * rt->D;
    size_t out_mod_table_elems = (size_t)rt->total_passes * 2 * rt->D;
    const char *flash_attn_env = NULL;
    int half_cache_requested = 0;

#define RT_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto fail; \
    } \
} while (0)

    rt->h_latent = (float *)malloc(rt->latent_elems * sizeof(float));
    rt->h_noise = (float *)malloc(512 * sizeof(float));
    rt->h_xy = (float *)malloc((size_t)rt->d_xy * sizeof(float));
    rt->h_inv_t = (float *)malloc((size_t)rt->d_t * sizeof(float));
    rt->h_x_pos = (int64_t *)malloc((size_t)rt->T * sizeof(int64_t));
    rt->h_y_pos = (int64_t *)malloc((size_t)rt->T * sizeof(int64_t));
    rt->h_t_pos = (int64_t *)malloc((size_t)rt->T * sizeof(int64_t));
    if (!rt->h_latent || !rt->h_noise || !rt->h_xy || !rt->h_inv_t || !rt->h_x_pos || !rt->h_y_pos || !rt->h_t_pos) {
        fprintf(stderr, "runtime host allocation failed\n");
        goto fail;
    }
    fill_rope_tables(rt->h_xy, rt->h_inv_t, rt->d_head, cfg->height, cfg->width);

    RT_CUDA(cudaMalloc((void **)&rt->d_latent, rt->latent_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_patch_rows, patch_row_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_noise, 512 * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_noise_hidden, (size_t)rt->mlp_hidden * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_cond, (size_t)rt->D * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_cond_act, (size_t)rt->D * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_control_input, (size_t)rt->ctrl_dim * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_emb_hidden, (size_t)rt->mlp_hidden * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_emb, (size_t)rt->D * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_emb_norm, (size_t)rt->D * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_out_mod_table, out_mod_table_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_final_tokens, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_latent_out, rt->latent_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_layer_mod_table, layer_mod_table_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_tokens, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_norm, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_qkv_raw, qkv_token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_q, rt->q_rope_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_k, rt->kv_rope_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_v, rt->kv_rope_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_v_first, rt->kv_rope_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_attn, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_attn_out, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_tokens_after_attn, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_norm, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_cond_by_layer, (size_t)layers_to_run * rt->D * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_hidden, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_ctrl_out, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_tokens_after_ctrl, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_mlp_in, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_mlp_hidden, (size_t)rt->T * rt->mlp_hidden * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_mlp_out, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_tokens_after_mlp, rt->token_elems * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_xy_table, (size_t)rt->d_xy * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_inv_t, (size_t)rt->d_t * sizeof(float)));
    RT_CUDA(cudaMalloc((void **)&rt->d_x_pos, (size_t)rt->T * sizeof(int64_t)));
    RT_CUDA(cudaMalloc((void **)&rt->d_y_pos, (size_t)rt->T * sizeof(int64_t)));
    RT_CUDA(cudaMalloc((void **)&rt->d_t_pos, (size_t)rt->T * sizeof(int64_t)));
    RT_CUDA(cudaMalloc((void **)&rt->d_linear_half, rt->linear_half_elems * sizeof(__half)));
    RT_CUDA(cudaMalloc((void **)&rt->d_mlp_hidden_half, (size_t)rt->T * rt->mlp_hidden * sizeof(__half)));
    if (rt->w8a8_mask) {
        RT_CUDA(cudaMalloc((void **)&rt->d_w8a8_x, rt->w8a8_x_elems * sizeof(int8_t)));
        RT_CUDA(cudaMalloc((void **)&rt->d_w8a8_x_scales, (size_t)rt->T * sizeof(float)));
        RT_CUDA(cudaMalloc((void **)&rt->d_w8a8_acc, rt->w8a8_acc_elems * sizeof(int32_t)));
        fprintf(stderr,
                "W8A8 shared scratch: %.2f MiB (A8 %.2f, INT32 %.2f)\n",
                ((double)rt->w8a8_x_elems +
                 (double)rt->T * sizeof(float) +
                 (double)rt->w8a8_acc_elems * sizeof(int32_t)) / (1024.0 * 1024.0),
                (double)rt->w8a8_x_elems / (1024.0 * 1024.0),
                (double)rt->w8a8_acc_elems * sizeof(int32_t) / (1024.0 * 1024.0));
    }
    if (rt->mlp_fc2_splitk_slices > 1) {
        if (rt->mlp_fc2_splitk_parallel_enabled) {
            rt->splitk_workspace_bytes = row_major_linear_fp16_input_weight_tensorop_splitk_parallel_workspace_size(
                rt->T, rt->mlp_hidden, rt->D, rt->mlp_fc2_splitk_slices);
        } else {
            rt->splitk_workspace_bytes = row_major_linear_fp16_weight_tensorop_splitk_workspace_size(
                rt->T, rt->mlp_hidden, rt->D, rt->mlp_fc2_splitk_slices);
        }
        if (rt->splitk_workspace_bytes > 0) {
            RT_CUDA(cudaMalloc(&rt->d_splitk_workspace, rt->splitk_workspace_bytes));
        }
        fprintf(stderr,
                "MLP fc2 split-K workspace: %.2f MiB\n",
                (double)rt->splitk_workspace_bytes / (1024.0 * 1024.0));
    }

    if (copy_f32_to_device(&rt->d_patch, weights->patchify_weight, patch_weight_elems)) goto fail;
    if (copy_f32_to_device(&rt->d_denoise_fc1, weights->denoise_fc1_weight, (size_t)rt->mlp_hidden * 512)) goto fail;
    if (copy_f32_to_device(&rt->d_denoise_fc2, weights->denoise_fc2_weight, (size_t)rt->D * rt->mlp_hidden)) goto fail;
    if (copy_f32_to_device(&rt->d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)rt->mlp_hidden * rt->ctrl_dim)) goto fail;
    if (copy_f32_to_device(&rt->d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)rt->D * rt->mlp_hidden)) goto fail;
    if (copy_f32_to_device(&rt->d_out_norm_w, weights->out_norm_fc_weight, out_norm_weight_elems)) goto fail;
    if (copy_f32_to_device(&rt->d_unpatch_w, weights->unpatchify_weight, unpatch_weight_elems)) goto fail;
    if (copy_f32_to_device(&rt->d_unpatch_b, weights->unpatchify_bias, (size_t)rt->C)) goto fail;
    if (copy_world_layers_to_device(
                &rt->d_layers,
                weights->layers,
                layers_to_run,
                rt->D,
                rt->kv_dim,
                rt->mlp_hidden,
                rt->w8a8_drop_fallback,
                rt->w8a8_mask,
                rt->w8a8_layer_begin,
                rt->w8a8_layer_end)) goto fail;
    if (preflight_runtime_w8a8(rt)) goto fail;
    flash_attn_env = getenv("WORLD_FLASH_ATTN");
    rt->attn_flash_enabled = flash_attn_env ? flash_attn_env[0] != '0' : 0;
    {
        const char *q4_env = getenv("WORLD_ATTN_D64_Q4_SHARED");
        rt->attn_q4_shared_enabled = q4_env ? q4_env[0] != '0' : 0;
    }
    {
        const char *half_env = getenv("WORLD_ATTN_D64_HALF_CACHE");
        const char *half_flash_env = getenv("WORLD_ATTN_D64_HALF_FLASH");
        const char *cutlass_attn_env = getenv("WORLD_ATTN_D64_CUTLASS");
        const char *cutlass_grouped_env = getenv("WORLD_ATTN_D64_CUTLASS_GROUPED");
        const char *cutlass_fmha_env = getenv("WORLD_ATTN_D64_FMHA");
        const char *sparse_fmha_env = getenv("WORLD_ATTN_D64_SPARSE_FMHA");
        half_cache_requested = half_env ? half_env[0] != '0' : 0;
        rt->attn_half_cache_enabled = half_cache_requested;
        rt->attn_half_flash_enabled = half_flash_env ? half_flash_env[0] != '0' : 0;
        rt->attn_cutlass_enabled = cutlass_attn_env ? cutlass_attn_env[0] != '0' : 1;
        rt->attn_cutlass_grouped_enabled = cutlass_grouped_env ? cutlass_grouped_env[0] != '0' : 0;
        rt->attn_cutlass_fmha_enabled = cutlass_fmha_env ? cutlass_fmha_env[0] != '0' : 0;
        rt->attn_sparse_fmha_enabled = sparse_fmha_env ? sparse_fmha_env[0] != '0' : 0;
#if !WORLD_HAS_CUTLASS_FMHA
        if (rt->attn_cutlass_fmha_enabled || rt->attn_sparse_fmha_enabled) {
            fprintf(stderr, "CUTLASS FMHA options ignored because the example headers are unavailable\n");
            rt->attn_cutlass_fmha_enabled = 0;
            rt->attn_sparse_fmha_enabled = 0;
        }
#endif
        if (rt->attn_cutlass_grouped_enabled) {
            rt->attn_cutlass_enabled = 1;
        }
        if (rt->attn_cutlass_fmha_enabled) {
            if (rt->attn_cutlass_grouped_enabled) {
                fprintf(stderr, "WORLD_ATTN_D64_FMHA disables WORLD_ATTN_D64_CUTLASS_GROUPED for this run\n");
            }
            rt->attn_cutlass_enabled = 1;
            rt->attn_cutlass_grouped_enabled = 0;
        }
        if (rt->attn_sparse_fmha_enabled) {
            if (rt->attn_cutlass_fmha_enabled || rt->attn_cutlass_grouped_enabled) {
                fprintf(stderr, "WORLD_ATTN_D64_SPARSE_FMHA takes precedence over the other CUTLASS attention probes\n");
            }
            rt->attn_cutlass_enabled = 1;
            rt->attn_cutlass_fmha_enabled = 0;
            rt->attn_cutlass_grouped_enabled = 0;
        }
        if (rt->attn_cutlass_enabled) {
            rt->attn_half_cache_enabled = 1;
            rt->attn_half_flash_enabled = 0;
        }
        if (rt->attn_half_cache_enabled && rt->d_head != 64) {
            fprintf(stderr, "D=64 attention probes ignored because d_head=%d\n", rt->d_head);
            rt->attn_half_cache_enabled = 0;
            rt->attn_half_flash_enabled = 0;
            rt->attn_cutlass_enabled = 0;
            rt->attn_cutlass_grouped_enabled = 0;
            rt->attn_cutlass_fmha_enabled = 0;
            rt->attn_sparse_fmha_enabled = 0;
        }
        if (rt->attn_cutlass_enabled && cfg->n_heads % cfg->n_kv_heads != 0) {
            fprintf(stderr,
                    "WORLD_ATTN_D64_CUTLASS ignored because n_heads=%d is not divisible by n_kv_heads=%d\n",
                    cfg->n_heads, cfg->n_kv_heads);
            rt->attn_cutlass_enabled = 0;
            rt->attn_cutlass_grouped_enabled = 0;
            rt->attn_cutlass_fmha_enabled = 0;
            rt->attn_sparse_fmha_enabled = 0;
            if (!half_cache_requested) rt->attn_half_cache_enabled = 0;
        }
        if (rt->attn_sparse_fmha_enabled && (rt->T % 128) != 0) {
            fprintf(stderr,
                    "WORLD_ATTN_D64_SPARSE_FMHA ignored because tokens_per_frame=%d is not divisible by 128\n",
                    rt->T);
            rt->attn_sparse_fmha_enabled = 0;
        }
        if (rt->attn_half_cache_enabled) {
            if (rt->attn_flash_enabled || rt->attn_q4_shared_enabled) {
                fprintf(stderr, "WORLD_ATTN_D64_HALF_CACHE disables the f32 flash/q4 attention probes\n");
            }
            rt->attn_flash_enabled = 0;
            rt->attn_q4_shared_enabled = 0;
            if (!rt->attn_cutlass_enabled) {
                fprintf(stderr, "D=64 attention FP16 KV cache enabled by WORLD_ATTN_D64_HALF_CACHE=1%s\n",
                        rt->attn_half_flash_enabled ? " with group-flash probe" : "");
            }
        } else if (rt->attn_half_flash_enabled) {
            fprintf(stderr, "WORLD_ATTN_D64_HALF_FLASH ignored because WORLD_ATTN_D64_HALF_CACHE is off\n");
            rt->attn_half_flash_enabled = 0;
        }
    }
    if (rt->attn_cutlass_enabled) {
        int max_window = cfg->local_window > cfg->global_window ? cfg->local_window : cfg->global_window;
        size_t attn_scratch_limit_mib = 2048;
        const char *limit_env = getenv("WORLD_ATTN_D64_CUTLASS_MAX_SCRATCH_MIB");
        if (limit_env && limit_env[0]) {
            long v = atol(limit_env);
            if (v > 0) attn_scratch_limit_mib = (size_t)v;
        }
        rt->attn_max_capacity = max_window * rt->T + rt->T;
        size_t q_half_elems = rt->q_rope_elems;
        size_t kv_compact_heads = rt->attn_cutlass_fmha_enabled ? (size_t)cfg->n_kv_heads : (size_t)cfg->n_heads;
        size_t kv_compact_elems = kv_compact_heads * rt->attn_max_capacity * 64;
        size_t score_elems = (size_t)cfg->n_heads * rt->T * rt->attn_max_capacity;
        size_t scratch_bytes = q_half_elems * sizeof(__half);
        if (rt->attn_sparse_fmha_enabled) {
            scratch_bytes += q_half_elems * sizeof(__half);
        } else {
            scratch_bytes += 2 * kv_compact_elems * sizeof(__half);
            if (rt->attn_cutlass_fmha_enabled) {
                scratch_bytes += q_half_elems * sizeof(__half);
            } else {
                scratch_bytes += score_elems * sizeof(float) + score_elems * sizeof(__half);
            }
        }
        size_t scratch_limit_bytes = attn_scratch_limit_mib * 1024ull * 1024ull;
        if (scratch_bytes > scratch_limit_bytes) {
            fprintf(stderr,
                    "WORLD_ATTN_D64_CUTLASS disabled: scratch %.2f MiB exceeds limit %zu MiB (override with WORLD_ATTN_D64_CUTLASS_MAX_SCRATCH_MIB)\n",
                    (double)scratch_bytes / (1024.0 * 1024.0),
                    attn_scratch_limit_mib);
            rt->attn_cutlass_enabled = 0;
            rt->attn_cutlass_grouped_enabled = 0;
            rt->attn_cutlass_fmha_enabled = 0;
            rt->attn_sparse_fmha_enabled = 0;
            if (!half_cache_requested) rt->attn_half_cache_enabled = 0;
        } else {
            RT_CUDA(cudaMalloc((void **)&rt->d_attn_q_half, q_half_elems * sizeof(__half)));
            if (!rt->attn_sparse_fmha_enabled) {
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_k_compact, kv_compact_elems * sizeof(__half)));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_v_compact, kv_compact_elems * sizeof(__half)));
            }
            if (rt->attn_sparse_fmha_enabled) {
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_out_half, q_half_elems * sizeof(__half)));
                fprintf(stderr,
                        "D=64 native sparse GQA CUTLASS FMHA enabled (block=128 capacity=%d tokens scratch=%.2f MiB)\n",
                        rt->attn_max_capacity,
                        (double)scratch_bytes / (1024.0 * 1024.0));
            } else if (rt->attn_cutlass_fmha_enabled) {
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_out_half, q_half_elems * sizeof(__half)));
                fprintf(stderr,
                        "D=64 attention CUTLASS FMHA GQA bridge enabled (capacity=%d tokens scratch=%.2f MiB)\n",
                        rt->attn_max_capacity,
                        (double)scratch_bytes / (1024.0 * 1024.0));
            } else {
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_scores, score_elems * sizeof(float)));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_probs_half, score_elems * sizeof(__half)));
                fprintf(stderr,
                        "D=64 attention CUTLASS materialized QK/AV enabled%s (capacity=%d tokens scratch=%.2f MiB)\n",
                        rt->attn_cutlass_grouped_enabled ? " with grouped-M GQA path" : "",
                        rt->attn_max_capacity,
                        (double)scratch_bytes / (1024.0 * 1024.0));
            }
        }
    }
    if (alloc_device_world_caches(&rt->d_caches, cfg, layers_to_run, rt->T, cfg->n_kv_heads, rt->d_head, rt->attn_half_cache_enabled)) goto fail;
    if (rt->attn_flash_enabled) {
        fprintf(stderr, "tiled flash-like attention enabled by WORLD_FLASH_ATTN=1 for D=64 cache layers\n");
    }
    if (rt->attn_q4_shared_enabled) {
        fprintf(stderr, "D=64 attention q4 shared-KV kernel enabled\n");
    }
    RT_CUDA(cudaMemcpy(rt->d_xy_table, rt->h_xy, (size_t)rt->d_xy * sizeof(float), cudaMemcpyHostToDevice));
    RT_CUDA(cudaMemcpy(rt->d_inv_t, rt->h_inv_t, (size_t)rt->d_t * sizeof(float), cudaMemcpyHostToDevice));
    if (precompute_runtime_layer_mods(rt)) goto fail;
    release_runtime_precompute_and_w8a8_fallback_weights(rt);
    RT_CUDA(cudaEventCreate(&rt->ev_step_start));
    RT_CUDA(cudaEventCreate(&rt->ev_after_setup));
    RT_CUDA(cudaEventCreate(&rt->ev_after_transformer));
    RT_CUDA(cudaEventCreate(&rt->ev_after_vae));
    if (rt->profile_enabled) {
        RT_CUDA(cudaEventCreate(&rt->prof_start));
        RT_CUDA(cudaEventCreate(&rt->prof_stop));
    }
    if (taehv_decoder_init(&rt->d_vae, cfg, vae)) goto fail;
    RT_CUDA(cudaDeviceSynchronize());

    *out = rt;
#undef RT_CUDA
    return 0;

fail:
#undef RT_CUDA
    world_cuda_runtime_destroy(rt);
    return 1;
}

extern "C" int world_cuda_runtime_init_vae_encoder(
        WorldCudaRuntime *rt,
        const WorldVaeEncoderWeights *encoder) {
    if (!rt || !encoder || !rt->d_vae.buf0) return 1;
    CUDA_OK(cudaDeviceSynchronize());
    taehv_free_weights(rt->d_vae.encoder_convs, WORLD_VAE_ENCODER_CONV_COUNT);
    if (taehv_copy_weights(rt->d_vae.encoder_convs, encoder->convs,
                WORLD_VAE_ENCODER_CONV_COUNT, 0)) {
        taehv_free_weights(rt->d_vae.encoder_convs, WORLD_VAE_ENCODER_CONV_COUNT);
        return 1;
    }
    rt->d_vae.encoder_enabled = 1;
    fprintf(stderr, "VAE encoder initialized: RGB %dx%d -> latent [%d,%d,%d]\n",
            rt->W * 16, rt->H * 16, rt->C, rt->H, rt->W);
    return 0;
}

extern "C" int world_cuda_runtime_encode_image_rgb(
        WorldCudaRuntime *rt,
        const float *rgb,
        int width,
        int height,
        float *latent_out,
        float *seconds_out) {
    if (!rt || !rgb || !latent_out) return 1;
    double t0 = monotonic_seconds();
    if (taehv_encode_image_rgb(&rt->cfg, &rt->d_vae, rgb, width, height, latent_out)) return 1;
    float elapsed = (float)(monotonic_seconds() - t0);
    if (seconds_out) *seconds_out = elapsed;
    const char *dump_path = getenv("WORLD_DUMP_VAE_LATENT");
    if (dump_path && dump_path[0]) {
        FILE *dump = fopen(dump_path, "wb");
        size_t elems = (size_t)rt->C * rt->H * rt->W;
        if (!dump || fwrite(latent_out, sizeof(float), elems, dump) != elems) {
            fprintf(stderr, "failed to write VAE latent dump: %s\n", dump_path);
            if (dump) fclose(dump);
            return 1;
        }
        fclose(dump);
    }
    fprintf(stderr, "VAE encode CUDA: RGB %dx%d -> latent [%d,%d,%d] in %.3fms\n",
            width, height, rt->C, rt->H, rt->W, elapsed * 1000.0f);
    return 0;
}

extern "C" int world_cuda_runtime_reset(WorldCudaRuntime *rt, int frame_idx, unsigned int seed) {
    if (!rt || frame_idx < 0) return 1;
    CUDA_OK(cudaDeviceSynchronize());
    for (int layer = 0; layer < rt->layers_to_run; ++layer) {
        DeviceWorldLayerCache *cache = &rt->d_caches[layer];
        init_cache_written_kernel<<<div_up_i64(cache->capacity, 256), 256>>>(
            cache->written, cache->ring_length, rt->T);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemset(cache->index_count, 0, sizeof(int)));
        CUDA_OK(cudaMemset(cache->indices, 0, (size_t)cache->capacity * sizeof(int64_t)));
        CUDA_OK(cudaMemset(cache->block_ids, 0,
                    (size_t)div_up_i64(cache->capacity, 128) * sizeof(int32_t)));
        memset(cache->h_slot_written, 0, (size_t)cache->slot_count);
        cache->h_slot_written[cache->slot_count - 1] = 1;
    }
    for (int i = 0; i < WORLD_VAE_STREAM_MEM_COUNT; ++i) {
        size_t elems = taehv_stream_mem_elems(&rt->cfg, i);
        if (rt->d_vae.stream_mem[i]) {
            CUDA_OK(cudaMemset(rt->d_vae.stream_mem[i], 0, elems * sizeof(float)));
        }
        if (rt->d_vae.hstream_mem[i]) {
            CUDA_OK(cudaMemset(rt->d_vae.hstream_mem[i], 0, elems * sizeof(__half)));
        }
    }
    rt->d_vae.stream_started_f32 = 0;
    rt->d_vae.stream_started_h = 0;
    rt->h_latent_override = NULL;
    rt->next_frame_idx = frame_idx;
    rt->frame_ordinal = 0;
    rt->seed = seed;
    CUDA_OK(cudaDeviceSynchronize());
    fprintf(stderr, "CUDA runtime reset: frame_idx=%d seed=%u\n", frame_idx, seed);
    return 0;
}

extern "C" int world_cuda_runtime_step_rgb(
        WorldCudaRuntime *rt,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    if (!rt || !control_input) return 1;
    const WorldConfig *cfg = &rt->cfg;
    double t0 = monotonic_seconds();
    float setup_ms = 0.0f;
    float transformer_ms = 0.0f;
    float vae_ms = 0.0f;
    float total_ms = 0.0f;
    const char *fp16_gemm_env = getenv("WORLD_FP16_GEMM");
    int use_fp16_gemm = fp16_gemm_env ? fp16_gemm_env[0] != '0' : 1;
    int use_fp16_tensorop = fp16_gemm_env && strcmp(fp16_gemm_env, "simt") == 0 ? 0 : 1;

#define STEP_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)
#define STEP_LINEAR(x, w, y, m, k, n) do { \
    if (row_major_linear((x), (w), (y), (m), (k), (n))) return 1; \
} while (0)
#define STEP_LINEAR_FAST(x, w, wh, y, m, k, n) do { \
    if (use_fp16_gemm && (wh) && (m) > 1) { \
        if (use_fp16_tensorop) { \
            if (should_use_m64n64_tensorop(rt->fp16_gemm_m64n64_enabled, (m), (k), (n))) { \
                if (row_major_linear_fp16_weight_tensorop_m64n64((x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
            } else { \
                if (row_major_linear_fp16_weight_tensorop((x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
            } \
        } else { \
            if (row_major_linear_fp16_weight_simt((x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
        } \
    } else { \
        STEP_LINEAR((x), (w), (y), (m), (k), (n)); \
    } \
} while (0)
#define STEP_PROFILE_BEGIN() do { \
    if (runtime_profile_begin(rt)) return 1; \
} while (0)
#define STEP_PROFILE_ACCUM(ms_field, calls_field) do { \
    if (runtime_profile_accum(rt, &rt->ms_field, &rt->calls_field)) return 1; \
} while (0)

    STEP_CUDA(cudaEventRecord(rt->ev_step_start, 0));
    int current_frame_idx = rt->next_frame_idx;
    int frame_timestamp = current_frame_idx * rt->frame_stride;
    STEP_CUDA(cudaMemcpy(rt->d_control_input, control_input, (size_t)rt->ctrl_dim * sizeof(float), cudaMemcpyHostToDevice));
    STEP_LINEAR(rt->d_control_input, rt->d_ctrl_emb_fc1_w, rt->d_ctrl_emb_hidden, 1, rt->ctrl_dim, rt->mlp_hidden);
    silu_f32_kernel<<<div_up_i64(rt->mlp_hidden, 256), 256>>>(rt->d_ctrl_emb_hidden, rt->d_ctrl_emb_hidden, rt->mlp_hidden);
    STEP_CUDA(cudaGetLastError());
    STEP_LINEAR(rt->d_ctrl_emb_hidden, rt->d_ctrl_emb_fc2_w, rt->d_ctrl_emb, 1, rt->mlp_hidden, rt->D);
    rms_norm_rows_f32_kernel<<<1, 256>>>(rt->d_ctrl_emb, rt->d_ctrl_emb_norm, 1, rt->D, rt->rms_eps);
    STEP_CUDA(cudaGetLastError());
    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
        const DeviceWorldLayerWeights *lw = &rt->d_layers[layer_idx];
        if (lw->has_ctrl) {
            STEP_LINEAR(rt->d_ctrl_emb_norm, lw->ctrl_fc1_c_weight,
                        rt->d_ctrl_cond_by_layer + (size_t)layer_idx * rt->D,
                        1, rt->D, rt->D);
        }
    }

    if (rt->h_latent_override) {
        memcpy(rt->h_latent, rt->h_latent_override, rt->latent_elems * sizeof(float));
    } else {
        fill_latent(rt->h_latent, (int)rt->latent_elems, rt->seed + (unsigned int)rt->frame_ordinal, rt->noise_mode);
    }
    fill_positions(rt->h_x_pos, rt->h_y_pos, rt->h_t_pos, rt->T, cfg->width, frame_timestamp);
    STEP_CUDA(cudaMemcpy(rt->d_latent, rt->h_latent, rt->latent_elems * sizeof(float), cudaMemcpyHostToDevice));
    STEP_CUDA(cudaMemcpy(rt->d_x_pos, rt->h_x_pos, (size_t)rt->T * sizeof(int64_t), cudaMemcpyHostToDevice));
    STEP_CUDA(cudaMemcpy(rt->d_y_pos, rt->h_y_pos, (size_t)rt->T * sizeof(int64_t), cudaMemcpyHostToDevice));
    STEP_CUDA(cudaMemcpy(rt->d_t_pos, rt->h_t_pos, (size_t)rt->T * sizeof(int64_t), cudaMemcpyHostToDevice));
    fprintf(stderr, "live frame %d: frame_idx=%d frame_timestamp=%d\n",
            rt->frame_ordinal, current_frame_idx, frame_timestamp);
    STEP_CUDA(cudaEventRecord(rt->ev_after_setup, 0));
    runtime_profile_reset(rt);

    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= rt->steps_to_run;
        int table_pass_idx = rt->forced_pass_table_idx >= 0 ? rt->forced_pass_table_idx : pass_idx;
        int frozen_pass = !is_cache_pass;
        float sigma_step = is_cache_pass ? 0.0f : cfg->scheduler_sigmas[pass_idx];
        float next_sigma = is_cache_pass ? 0.0f : cfg->scheduler_sigmas[pass_idx + 1];
        float dsigma = next_sigma - sigma_step;
        if (table_pass_idx < 0 || table_pass_idx >= rt->precomputed_total_passes) {
            fprintf(stderr,
                    "invalid runtime pass table index %d for precomputed_total_passes=%d\n",
                    table_pass_idx, rt->precomputed_total_passes);
            return 1;
        }

        int patch_elems = rt->C * rt->ph * rt->pw;
        STEP_PROFILE_BEGIN();
        patchify_im2row_f32_kernel<<<div_up_i64((int64_t)rt->T * patch_elems, 256), 256>>>(
            rt->d_latent, rt->d_patch_rows, rt->C, rt->H, rt->W, rt->ph, rt->pw, cfg->height, cfg->width);
        STEP_CUDA(cudaGetLastError());
        STEP_LINEAR(rt->d_patch_rows, rt->d_patch, rt->d_tokens, rt->T, patch_elems, rt->D);
        STEP_PROFILE_ACCUM(prof_patch_ms, prof_patch_calls);
        float *d_tokens_cur = rt->d_tokens;
        float *d_tokens_next = rt->d_tokens_after_mlp;

        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            const DeviceWorldLayerWeights *lw = &rt->d_layers[layer_idx];
            DeviceWorldLayerCache *cache = &rt->d_caches[layer_idx];

            float *d_layer_mod = rt->d_layer_mod_table + ((int64_t)table_pass_idx * rt->layers_to_run + layer_idx) * (6 * rt->D);
            float *d_s0 = d_layer_mod;
            float *d_b0 = d_layer_mod + rt->D;
            float *d_g0 = d_layer_mod + 2 * rt->D;
            float *d_s1 = d_layer_mod + 3 * rt->D;
            float *d_b1 = d_layer_mod + 4 * rt->D;
            float *d_g1 = d_layer_mod + 5 * rt->D;

            int qkv_w8a8 =
                (rt->w8a8_mask & WORLD_W8A8_QKV) &&
                lw->qkv_proj_weight_i8 && lw->qkv_proj_weight_i8_scales;
            int out_w8a8 =
                (rt->w8a8_mask & WORLD_W8A8_OUT) &&
                lw->out_proj_weight_i8 && lw->out_proj_weight_i8_scales;
            int qkv_half_boundary =
                !qkv_w8a8 &&
                use_fp16_gemm && use_fp16_tensorop && rt->half_gemm_boundary_enabled &&
                lw->qkv_proj_weight_h &&
                should_use_m64n64_tensorop(
                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->D + 2 * rt->kv_dim);
            STEP_PROFILE_BEGIN();
            if (qkv_w8a8) {
                rms_norm_quantize_rows_i8_kernel<<<rt->T, 256>>>(
                    d_tokens_cur,
                    d_s0,
                    d_b0,
                    rt->d_w8a8_x,
                    rt->d_w8a8_x_scales,
                    rt->T,
                    rt->D,
                    rt->rms_eps);
            } else if (qkv_half_boundary) {
                ada_rms_norm_single_f16_kernel<<<rt->T, 256>>>(
                    d_tokens_cur, d_s0, d_b0, rt->d_linear_half, rt->T, rt->D, rt->rms_eps);
            } else {
                ada_rms_norm_single_f32_kernel<<<rt->T, 256>>>(
                    d_tokens_cur, d_s0, d_b0, rt->d_norm, rt->T, rt->D, rt->rms_eps);
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_norm_ms, prof_norm_calls);
            STEP_PROFILE_BEGIN();
            if (qkv_w8a8) {
                if (row_major_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->qkv_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D + 2 * rt->kv_dim)) return 1;
            } else if (qkv_half_boundary) {
                if (row_major_linear_fp16_input_weight_tensorop_m64n64(
                            rt->d_linear_half,
                            lw->qkv_proj_weight_h,
                            rt->d_qkv_raw,
                            rt->T,
                            rt->D,
                            rt->D + 2 * rt->kv_dim)) return 1;
            } else {
                STEP_LINEAR_FAST(rt->d_norm, lw->qkv_proj_weight, lw->qkv_proj_weight_h,
                                 rt->d_qkv_raw, rt->T, rt->D, rt->D + 2 * rt->kv_dim);
            }
            STEP_PROFILE_ACCUM(prof_qkv_gemm_ms, prof_qkv_gemm_calls);
            float *d_v_cur = (cfg->value_residual && layer_idx == 0) ? rt->d_v_first : rt->d_v;
            STEP_PROFILE_BEGIN();
            {
                dim3 grid(rt->T, cfg->n_heads + 2 * cfg->n_kv_heads);
                size_t smem = (size_t)(rt->d_head + 256) * sizeof(float);
                if (qkv_w8a8) {
                    qkv_fused_rms_rope_i32_dequant_kernel<<<grid, 256, smem>>>(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->qkv_proj_weight_i8_scales,
                        rt->d_q,
                        rt->d_k,
                        d_v_cur,
                        rt->d_x_pos,
                        rt->d_y_pos,
                        rt->d_t_pos,
                        rt->d_xy_table,
                        rt->d_inv_t,
                        rt->T,
                        cfg->n_heads,
                        cfg->n_kv_heads,
                        rt->d_head,
                        cfg->width,
                        cfg->height,
                        rt->rms_eps);
                } else {
                    qkv_fused_rms_rope_f32_kernel<<<grid, 256, smem>>>(
                        rt->d_qkv_raw, rt->d_q, rt->d_k, d_v_cur,
                        rt->d_x_pos, rt->d_y_pos, rt->d_t_pos, rt->d_xy_table, rt->d_inv_t,
                        rt->T, cfg->n_heads, cfg->n_kv_heads, rt->d_head, cfg->width, cfg->height, rt->rms_eps);
                }
            }
            STEP_CUDA(cudaGetLastError());
            if (cfg->value_residual && layer_idx != 0) {
                lerp_inplace_f32_kernel<<<div_up_i64((int64_t)rt->kv_rope_elems, 256), 256>>>(
                    rt->d_v, rt->d_v_first, lw->v_lamb, (int64_t)rt->kv_rope_elems);
                STEP_CUDA(cudaGetLastError());
            }
            STEP_PROFILE_ACCUM(prof_qkv_rope_ms, prof_qkv_rope_calls);

            STEP_PROFILE_BEGIN();
            int bucket = (current_frame_idx + (cache->pinned_dilation - 1)) / cache->pinned_dilation;
            int num_buckets = (cache->ring_length / rt->T) / cache->pinned_dilation;
            int base = (bucket % num_buckets) * rt->T;
            bool write_step = (current_frame_idx % cache->pinned_dilation) == 0;
            int host_index_count = cache_host_index_count(cache, rt->T, base, write_step);
            if (rt->attn_half_cache_enabled) {
                kv_cache_upsert_copy_f16_kernel<<<div_up_i64((int64_t)cfg->n_kv_heads * rt->T * rt->d_head, 256), 256>>>(
                    cache->k_h, cache->v_h, rt->d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, rt->T, rt->d_head, cache->ring_length, base, write_step, (bool)frozen_pass);
            } else {
                kv_cache_upsert_copy_f32_kernel<<<div_up_i64((int64_t)cfg->n_kv_heads * rt->T * rt->d_head, 256), 256>>>(
                    cache->k, cache->v, rt->d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, rt->T, rt->d_head, cache->ring_length, base, write_step, (bool)frozen_pass);
            }
            STEP_CUDA(cudaGetLastError());
            collect_cache_frame_indices_kernel<<<cache->capacity / rt->T, 256>>>(
                cache->written, cache->indices, cache->block_ids, cache->index_count,
                cache->capacity, rt->T, base, write_step);
            STEP_CUDA(cudaGetLastError());
            cache_host_note_upsert(cache, rt->T, base, write_step, (bool)frozen_pass);
            STEP_PROFILE_ACCUM(prof_cache_ms, prof_cache_calls);
            STEP_PROFILE_BEGIN();
            int attn_output_half_ready = 0;
            if (rt->d_head == 64) {
                int attn_done = 0;
                if (rt->attn_half_cache_enabled) {
                    int group = cfg->n_heads % cfg->n_kv_heads == 0 ? cfg->n_heads / cfg->n_kv_heads : 0;
                    if (rt->attn_cutlass_enabled && group > 0) {
                        int cutlass_rc = 0;
                        if (rt->attn_sparse_fmha_enabled) {
                            int fmha_half_output =
                                !out_w8a8 &&
                                use_fp16_gemm && use_fp16_tensorop && rt->half_gemm_boundary_enabled &&
                                lw->out_proj_weight_h &&
                                should_use_m64n64_tensorop(
                                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->D);
                            cutlass_rc = sparse_attention_cache_d64_fmha_f16_kv(
                                    rt->d_q,
                                    cache->k_h,
                                    cache->v_h,
                                    cache->block_ids,
                                    host_index_count / 128,
                                    rt->d_attn,
                                    rt->d_linear_half,
                                    fmha_half_output,
                                    rt->d_attn_q_half,
                                    rt->d_attn_out_half,
                                    cfg->n_heads,
                                    cfg->n_kv_heads,
                                    rt->T,
                                    cache->capacity,
                                    1.0f / 8.0f);
                            attn_output_half_ready = fmha_half_output;
                        } else if (rt->attn_cutlass_fmha_enabled) {
                            int fmha_half_output =
                                !out_w8a8 &&
                                use_fp16_gemm && use_fp16_tensorop && rt->half_gemm_boundary_enabled &&
                                lw->out_proj_weight_h &&
                                should_use_m64n64_tensorop(
                                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->D);
                            cutlass_rc = indexed_attention_cache_d64_fmha_f16_kv(
                                    rt->d_q,
                                    cache->k_h,
                                    cache->v_h,
                                    cache->indices,
                                    host_index_count,
                                    rt->d_attn,
                                    rt->d_linear_half,
                                    fmha_half_output,
                                    rt->d_attn_q_half,
                                    rt->d_attn_k_compact,
                                    rt->d_attn_v_compact,
                                    rt->d_attn_out_half,
                                    cfg->n_heads,
                                    cfg->n_kv_heads,
                                    rt->T,
                                    cache->capacity,
                                    1.0f / 8.0f);
                            attn_output_half_ready = fmha_half_output;
                        } else if (rt->attn_cutlass_grouped_enabled) {
                            cutlass_rc = indexed_attention_cache_d64_cutlass_grouped_f16_kv(
                                    rt->d_q,
                                    cache->k_h,
                                    cache->v_h,
                                    cache->indices,
                                    host_index_count,
                                    rt->d_attn,
                                    rt->d_attn_q_half,
                                    rt->d_attn_k_compact,
                                    rt->d_attn_v_compact,
                                    rt->d_attn_scores,
                                    rt->d_attn_probs_half,
                                    cfg->n_heads,
                                    cfg->n_kv_heads,
                                    rt->T,
                                    cache->capacity,
                                    1.0f / 8.0f);
                        } else {
                            cutlass_rc = indexed_attention_cache_d64_cutlass_f16_kv(
                                    rt->d_q,
                                    cache->k_h,
                                    cache->v_h,
                                    cache->indices,
                                    host_index_count,
                                    rt->d_attn,
                                    rt->d_attn_q_half,
                                    rt->d_attn_k_compact,
                                    rt->d_attn_v_compact,
                                    rt->d_attn_scores,
                                    rt->d_attn_probs_half,
                                    cfg->n_heads,
                                    cfg->n_kv_heads,
                                    rt->T,
                                    cache->capacity,
                                    1.0f / 8.0f);
                        }
                        if (cutlass_rc) return 1;
                    } else if (rt->attn_half_flash_enabled && group > 0 && group <= WORLD_ATTN_D64_FLASH_WARPS) {
                        int q_per_h = WORLD_ATTN_D64_FLASH_WARPS / group;
                        if (q_per_h < 1) q_per_h = 1;
                        int q_blocks = div_up_i64(rt->T, q_per_h);
                        size_t smem = (size_t)2 * WORLD_ATTN_D64_K_BLOCK * 64 * sizeof(__half);
                        indexed_attention_cache_d64_flash_f16_kv_kernel<<<cfg->n_kv_heads * q_blocks, 32 * WORLD_ATTN_D64_FLASH_WARPS, smem>>>(
                            rt->d_q, cache->k_h, cache->v_h, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f);
                    } else {
                        indexed_attention_cache_d64_warp_f16_kv_kernel<<<div_up_i64((int64_t)cfg->n_heads * rt->T, 4), 128>>>(
                            rt->d_q, cache->k_h, cache->v_h, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f);
                    }
                    attn_done = 1;
                }
                if (rt->attn_flash_enabled && cfg->n_heads % cfg->n_kv_heads == 0 && (cfg->n_heads / cfg->n_kv_heads) <= WORLD_ATTN_D64_FLASH_WARPS) {
                    int group = cfg->n_heads / cfg->n_kv_heads;
                    int q_per_h = WORLD_ATTN_D64_FLASH_WARPS / group;
                    if (q_per_h < 1) q_per_h = 1;
                    int q_blocks = div_up_i64(rt->T, q_per_h);
                    size_t smem = (size_t)2 * WORLD_ATTN_D64_K_BLOCK * 64 * sizeof(float);
                    indexed_attention_cache_d64_flash_f32_kernel<<<cfg->n_kv_heads * q_blocks, 32 * WORLD_ATTN_D64_FLASH_WARPS, smem>>>(
                        rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                        cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                        1.0f / 8.0f);
                    attn_done = 1;
                }
                if (!attn_done) {
                    if (rt->attn_q4_shared_enabled) {
                        int q_blocks = div_up_i64(rt->T, WORLD_ATTN_D64_Q_BLOCK);
                        size_t smem = (size_t)2 * WORLD_ATTN_D64_K_BLOCK * 64 * sizeof(float);
                        indexed_attention_cache_d64_q4_shared_f32_kernel<<<cfg->n_heads * q_blocks, 128, smem>>>(
                            rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f);
                    } else {
                        indexed_attention_cache_d64_warp_f32_kernel<<<div_up_i64((int64_t)cfg->n_heads * rt->T, 4), 128>>>(
                            rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f);
                    }
                }
            } else {
                indexed_attention_cache_f32_kernel<<<cfg->n_heads * rt->T, 256>>>(
                    rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                    cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity, rt->d_head,
                    1.0f / sqrtf((float)rt->d_head));
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_attn_ms, prof_attn_calls);
            STEP_PROFILE_BEGIN();
            if (out_w8a8) {
                quantize_rows_f32_i8_kernel<<<rt->T, 256>>>(
                    rt->d_attn,
                    rt->d_w8a8_x,
                    rt->d_w8a8_x_scales,
                    rt->T,
                    rt->D);
                STEP_CUDA(cudaGetLastError());
                if (row_major_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->out_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            } else if (attn_output_half_ready) {
                if (row_major_linear_fp16_input_weight_tensorop_m64n64(
                            rt->d_linear_half,
                            lw->out_proj_weight_h,
                            rt->d_attn_out,
                            rt->T,
                            rt->D,
                            rt->D)) return 1;
            } else {
                STEP_LINEAR_FAST(rt->d_attn, lw->out_proj_weight, lw->out_proj_weight_h,
                                 rt->d_attn_out, rt->T, rt->D, rt->D);
            }
            STEP_PROFILE_ACCUM(prof_attn_out_gemm_ms, prof_attn_out_gemm_calls);
            STEP_PROFILE_BEGIN();
            if (out_w8a8) {
                dequant_gated_residual_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                    rt->d_w8a8_acc,
                    rt->d_w8a8_x_scales,
                    lw->out_proj_weight_i8_scales,
                    d_tokens_cur,
                    d_g0,
                    rt->d_tokens_after_attn,
                    rt->T,
                    rt->D);
            } else {
                gated_residual_add_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                    d_tokens_cur, rt->d_attn_out, d_g0, rt->d_tokens_after_attn, rt->T, rt->D);
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_attn_residual_ms, prof_attn_residual_calls);

            float *d_tokens_ctrl = rt->d_tokens_after_attn;
            if (lw->has_ctrl) {
                STEP_PROFILE_BEGIN();
                int ctrl_w8a8 =
                    (rt->w8a8_mask & WORLD_W8A8_CTRL) &&
                    lw->ctrl_fc1_x_weight_i8 && lw->ctrl_fc1_x_weight_i8_scales &&
                    lw->ctrl_fc2_weight_i8 && lw->ctrl_fc2_weight_i8_scales;
                if (ctrl_w8a8) {
                    rms_norm_quantize_rows_i8_kernel<<<rt->T, 256>>>(
                        rt->d_tokens_after_attn,
                        NULL,
                        NULL,
                        rt->d_w8a8_x,
                        rt->d_w8a8_x_scales,
                        rt->T,
                        rt->D,
                        rt->rms_eps);
                    STEP_CUDA(cudaGetLastError());
                    if (row_major_gemm_i8_i32(
                        rt->d_w8a8_x,
                        lw->ctrl_fc1_x_weight_i8,
                        rt->d_w8a8_acc,
                        rt->T,
                        rt->D,
                        rt->D)) return 1;
                    dequant_silu_quantize_rows_i8_kernel<<<rt->T, 256>>>(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->ctrl_fc1_x_weight_i8_scales,
                        rt->d_ctrl_cond_by_layer + (size_t)layer_idx * rt->D,
                        rt->d_w8a8_x,
                        rt->d_w8a8_x_scales,
                        rt->T,
                        rt->D);
                    STEP_CUDA(cudaGetLastError());
                    if (row_major_gemm_i8_i32(
                        rt->d_w8a8_x,
                        lw->ctrl_fc2_weight_i8,
                        rt->d_w8a8_acc,
                        rt->T,
                        rt->D,
                        rt->D)) return 1;
                    dequant_add_residual_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->ctrl_fc2_weight_i8_scales,
                        rt->d_tokens_after_attn,
                        rt->d_tokens_after_ctrl,
                        rt->T,
                        rt->D);
                    STEP_CUDA(cudaGetLastError());
                } else {
                    rms_norm_rows_f32_kernel<<<rt->T, 256>>>(
                        rt->d_tokens_after_attn, rt->d_ctrl_norm, rt->T, rt->D, rt->rms_eps);
                    STEP_CUDA(cudaGetLastError());
                    STEP_LINEAR_FAST(rt->d_ctrl_norm, lw->ctrl_fc1_x_weight, lw->ctrl_fc1_x_weight_h,
                                     rt->d_ctrl_hidden, rt->T, rt->D, rt->D);
                    add_channel_silu_inplace_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                        rt->d_ctrl_hidden, rt->d_ctrl_cond_by_layer + (size_t)layer_idx * rt->D, rt->T, rt->D);
                    STEP_CUDA(cudaGetLastError());
                    STEP_LINEAR_FAST(rt->d_ctrl_hidden, lw->ctrl_fc2_weight, lw->ctrl_fc2_weight_h,
                                     rt->d_ctrl_out, rt->T, rt->D, rt->D);
                    add_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                        rt->d_tokens_after_attn, rt->d_ctrl_out, rt->d_tokens_after_ctrl, rt->token_elems);
                    STEP_CUDA(cudaGetLastError());
                }
                d_tokens_ctrl = rt->d_tokens_after_ctrl;
                STEP_PROFILE_ACCUM(prof_ctrl_ms, prof_ctrl_calls);
            }

            int mlp_w8a8 =
                (rt->w8a8_mask & WORLD_W8A8_MLP) &&
                lw->dit_mlp_fc1_weight_i8 && lw->dit_mlp_fc1_weight_i8_scales &&
                lw->dit_mlp_fc2_weight_i8 && lw->dit_mlp_fc2_weight_i8_scales;
            int mlp_fc1_half_boundary =
                !mlp_w8a8 &&
                use_fp16_gemm && use_fp16_tensorop && rt->half_gemm_boundary_enabled &&
                rt->mlp_fc1_silu_epilogue_enabled &&
                lw->dit_mlp_fc1_weight_h && lw->dit_mlp_fc2_weight_h &&
                rt->mlp_fc2_splitk_slices > 1 &&
                should_use_m64n64_tensorop(
                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->mlp_hidden);
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                rms_norm_quantize_rows_i8_kernel<<<rt->T, 256>>>(
                    d_tokens_ctrl,
                    d_s1,
                    d_b1,
                    rt->d_w8a8_x,
                    rt->d_w8a8_x_scales,
                    rt->T,
                    rt->D,
                    rt->rms_eps);
            } else if (mlp_fc1_half_boundary) {
                ada_rms_norm_single_f16_kernel<<<rt->T, 256>>>(
                    d_tokens_ctrl, d_s1, d_b1, rt->d_linear_half, rt->T, rt->D, rt->rms_eps);
            } else {
                ada_rms_norm_single_f32_kernel<<<rt->T, 256>>>(
                    d_tokens_ctrl, d_s1, d_b1, rt->d_mlp_in, rt->T, rt->D, rt->rms_eps);
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_norm_ms, prof_norm_calls);
            int mlp_hidden_half_ready = 0;
            __half *d_mlp_hidden_half_cur = NULL;
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                if (row_major_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->dit_mlp_fc1_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->mlp_hidden)) return 1;
            } else if (use_fp16_gemm && use_fp16_tensorop &&
                    rt->mlp_fc1_silu_epilogue_enabled &&
                    lw->dit_mlp_fc1_weight_h && lw->dit_mlp_fc2_weight_h &&
                    rt->mlp_fc2_splitk_slices > 1 &&
                    should_use_m64n64_tensorop(rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->mlp_hidden)) {
                if (mlp_fc1_half_boundary) {
                    if (row_major_linear_fp16_input_weight_tensorop_m64n64_silu_half(
                                rt->d_linear_half,
                                lw->dit_mlp_fc1_weight_h,
                                rt->d_mlp_hidden_half,
                                rt->T,
                                rt->D,
                                rt->mlp_hidden)) return 1;
                } else {
                    if (row_major_linear_fp16_weight_tensorop_m64n64_silu_half(
                                rt->d_mlp_in,
                                rt->d_linear_half,
                                lw->dit_mlp_fc1_weight_h,
                                rt->d_mlp_hidden_half,
                                rt->T,
                                rt->D,
                                rt->mlp_hidden)) return 1;
                }
                mlp_hidden_half_ready = 1;
                d_mlp_hidden_half_cur = rt->d_mlp_hidden_half;
            } else {
                STEP_LINEAR_FAST(rt->d_mlp_in, lw->dit_mlp_fc1_weight, lw->dit_mlp_fc1_weight_h,
                                 rt->d_mlp_hidden, rt->T, rt->D, rt->mlp_hidden);
            }
            STEP_PROFILE_ACCUM(prof_mlp_fc1_ms, prof_mlp_fc1_calls);
            if (!mlp_hidden_half_ready) {
                STEP_PROFILE_BEGIN();
                if (mlp_w8a8) {
                    dequant_silu_quantize_rows_i8_kernel<<<rt->T, 256>>>(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->dit_mlp_fc1_weight_i8_scales,
                        NULL,
                        rt->d_w8a8_x,
                        rt->d_w8a8_x_scales,
                        rt->T,
                        rt->mlp_hidden);
                } else if (use_fp16_gemm && use_fp16_tensorop && lw->dit_mlp_fc2_weight_h && rt->mlp_fc2_splitk_slices > 1) {
                    silu_f32_to_f16_kernel<<<div_up_i64((int64_t)rt->T * rt->mlp_hidden, 256), 256>>>(
                        rt->d_mlp_hidden, rt->d_mlp_hidden_half, (int64_t)rt->T * rt->mlp_hidden);
                    mlp_hidden_half_ready = 1;
                    d_mlp_hidden_half_cur = rt->d_mlp_hidden_half;
                } else {
                    silu_f32_kernel<<<div_up_i64((int64_t)rt->T * rt->mlp_hidden, 256), 256>>>(
                        rt->d_mlp_hidden, rt->d_mlp_hidden, (int64_t)rt->T * rt->mlp_hidden);
                }
                STEP_CUDA(cudaGetLastError());
                STEP_PROFILE_ACCUM(prof_mlp_silu_ms, prof_mlp_silu_calls);
            }
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                if (row_major_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->dit_mlp_fc2_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->mlp_hidden,
                    rt->D)) return 1;
            } else if (mlp_hidden_half_ready) {
                if (!d_mlp_hidden_half_cur) return 1;
                if (rt->mlp_fc2_splitk_parallel_enabled) {
                    if (row_major_linear_fp16_input_weight_tensorop_splitk_parallel(
                                d_mlp_hidden_half_cur,
                                lw->dit_mlp_fc2_weight_h,
                                rt->d_mlp_out,
                                rt->T,
                                rt->mlp_hidden,
                                rt->D,
                                rt->mlp_fc2_splitk_slices,
                                rt->d_splitk_workspace,
                                rt->splitk_workspace_bytes)) return 1;
                } else {
                    if (row_major_linear_fp16_input_weight_tensorop_splitk(
                                d_mlp_hidden_half_cur,
                                lw->dit_mlp_fc2_weight_h,
                                rt->d_mlp_out,
                                rt->T,
                                rt->mlp_hidden,
                                rt->D,
                                rt->mlp_fc2_splitk_slices,
                                rt->d_splitk_workspace,
                                rt->splitk_workspace_bytes)) return 1;
                }
            } else if (use_fp16_gemm && use_fp16_tensorop && lw->dit_mlp_fc2_weight_h &&
                    should_use_m64n64_tensorop(rt->fp16_gemm_m64n64_enabled, rt->T, rt->mlp_hidden, rt->D)) {
                if (row_major_linear_fp16_weight_tensorop_m64n64(
                            rt->d_mlp_hidden,
                            rt->d_linear_half,
                            lw->dit_mlp_fc2_weight_h,
                            rt->d_mlp_out,
                            rt->T,
                            rt->mlp_hidden,
                            rt->D)) return 1;
            } else {
                STEP_LINEAR_FAST(rt->d_mlp_hidden, lw->dit_mlp_fc2_weight, lw->dit_mlp_fc2_weight_h,
                                 rt->d_mlp_out, rt->T, rt->mlp_hidden, rt->D);
            }
            STEP_PROFILE_ACCUM(prof_mlp_fc2_ms, prof_mlp_fc2_calls);
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                dequant_gated_residual_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                    rt->d_w8a8_acc,
                    rt->d_w8a8_x_scales,
                    lw->dit_mlp_fc2_weight_i8_scales,
                    d_tokens_ctrl,
                    d_g1,
                    d_tokens_next,
                    rt->T,
                    rt->D);
            } else {
                gated_residual_add_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                    d_tokens_ctrl, rt->d_mlp_out, d_g1, d_tokens_next, rt->T, rt->D);
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_mlp_residual_ms, prof_mlp_residual_calls);

            float *d_swap = d_tokens_cur;
            d_tokens_cur = d_tokens_next;
            d_tokens_next = d_swap;
        }

        if (is_cache_pass) continue;

        float *d_out_mod = rt->d_out_mod_table + (int64_t)table_pass_idx * 2 * rt->D;
        STEP_PROFILE_BEGIN();
        out_norm_silu_f32_kernel<<<rt->T, 256>>>(d_tokens_cur, d_out_mod, rt->d_final_tokens, rt->T, rt->D, rt->rms_eps);
        STEP_CUDA(cudaGetLastError());
        unpatchify_orig_f32_kernel<<<rt->T * (rt->C * rt->ph * rt->pw), 256>>>(
            rt->d_final_tokens, rt->d_unpatch_w, rt->d_unpatch_b, rt->d_latent_out,
            rt->T, rt->D, rt->C, rt->H, rt->W, rt->ph, rt->pw, cfg->width, rt->C * rt->ph * rt->pw);
        STEP_CUDA(cudaGetLastError());
        latent_update_f32_kernel<<<div_up_i64((int64_t)rt->latent_elems, 256), 256>>>(
            rt->d_latent, rt->d_latent_out, dsigma, (int64_t)rt->latent_elems);
        STEP_CUDA(cudaGetLastError());
        STEP_PROFILE_ACCUM(prof_out_ms, prof_out_calls);
    }

    STEP_CUDA(cudaEventRecord(rt->ev_after_transformer, 0));
    {
        const char *dump_latent_path = getenv("WORLD_DUMP_RUNTIME_LATENT");
        if (dump_latent_path && dump_latent_path[0]) {
            float *h_dump_latent = (float *)malloc(rt->latent_elems * sizeof(float));
            if (!h_dump_latent) return 1;
            STEP_CUDA(cudaMemcpy(h_dump_latent, rt->d_latent, rt->latent_elems * sizeof(float), cudaMemcpyDeviceToHost));
            FILE *f = fopen(dump_latent_path, rt->frame_ordinal == 0 ? "wb" : "ab");
            if (!f) {
                free(h_dump_latent);
                return 1;
            }
            fwrite(h_dump_latent, sizeof(float), rt->latent_elems, f);
            fclose(f);
            free(h_dump_latent);
        }
    }
    if (world_cuda_decode_vae_to_rgb(cfg, &rt->d_vae, rt->d_latent, rgb_out, frames_out, width_out, height_out)) return 1;
    STEP_CUDA(cudaEventRecord(rt->ev_after_vae, 0));
    STEP_CUDA(cudaEventSynchronize(rt->ev_after_vae));
    STEP_CUDA(cudaEventElapsedTime(&setup_ms, rt->ev_step_start, rt->ev_after_setup));
    STEP_CUDA(cudaEventElapsedTime(&transformer_ms, rt->ev_after_setup, rt->ev_after_transformer));
    STEP_CUDA(cudaEventElapsedTime(&vae_ms, rt->ev_after_transformer, rt->ev_after_vae));
    STEP_CUDA(cudaEventElapsedTime(&total_ms, rt->ev_step_start, rt->ev_after_vae));
    int cache_tokens_l0 = -1;
    if (rt->layers_to_run > 0) {
        STEP_CUDA(cudaMemcpy(&cache_tokens_l0, rt->d_caches[0].index_count, sizeof(int), cudaMemcpyDeviceToHost));
    }
    rt->frame_ordinal += 1;
    rt->next_frame_idx += 1;
    if (seconds_out) *seconds_out = (float)(monotonic_seconds() - t0);
    {
        int frames = frames_out ? *frames_out : 4;
        float total_s = total_ms * 1.0e-3f;
        fprintf(stderr,
                "live timing: setup=%.3fms transformer=%.3fms vae=%.3fms total=%.3fms chunk_fps=%.3f rgb_fps=%.3f cache_tokens_l0=%d cache_frames_l0=%.1f\n",
                setup_ms, transformer_ms, vae_ms, total_ms,
                total_s > 0.0f ? 1.0f / total_s : 0.0f,
                total_s > 0.0f ? (float)frames / total_s : 0.0f,
                cache_tokens_l0,
                (rt->T > 0 && cache_tokens_l0 >= 0) ? (float)cache_tokens_l0 / (float)rt->T : -1.0f);
    }
    runtime_profile_print(rt);
#undef STEP_CUDA
#undef STEP_LINEAR
#undef STEP_LINEAR_FAST
#undef STEP_PROFILE_BEGIN
#undef STEP_PROFILE_ACCUM
    return 0;
}

extern "C" int world_cuda_runtime_seed_latent_rgb(
        WorldCudaRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    if (!rt || !latent || !control_input) return 1;
    int old_steps_to_run = rt->steps_to_run;
    int old_total_passes = rt->total_passes;
    int old_forced_pass_table_idx = rt->forced_pass_table_idx;
    const float *old_override = rt->h_latent_override;

    rt->h_latent_override = latent;
    rt->steps_to_run = 0;
    rt->total_passes = 1;
    rt->forced_pass_table_idx = old_steps_to_run;
    int rc = world_cuda_runtime_step_rgb(
        rt, control_input, rgb_out, width_out, height_out, frames_out, seconds_out);

    rt->h_latent_override = old_override;
    rt->steps_to_run = old_steps_to_run;
    rt->total_passes = old_total_passes;
    rt->forced_pass_table_idx = old_forced_pass_table_idx;
    return rc;
}

extern "C" int world_cuda_generation_probe(
        const WorldConfig *cfg,
        const float *patchify_weight,
        const float *q_proj_weight,
        unsigned int seed) {
    int C = cfg->channels;
    int H = cfg->height * cfg->patch_h;
    int W = cfg->width * cfg->patch_w;
    int D = cfg->d_model;
    int ph = cfg->patch_h;
    int pw = cfg->patch_w;
    int Hp = H / ph;
    int Wp = W / pw;
    int T = Hp * Wp;

    size_t latent_elems = (size_t)C * H * W;
    size_t patch_weight_elems = (size_t)D * C * ph * pw;
    size_t token_elems = (size_t)T * D;
    size_t q_weight_elems = (size_t)D * D;

    float *h_latent = (float *)malloc(latent_elems * sizeof(float));
    float *h_tokens = (float *)malloc(token_elems * sizeof(float));
    float *h_q = (float *)malloc(token_elems * sizeof(float));
    if (!h_latent || !h_tokens || !h_q) {
        fprintf(stderr, "host allocation failed\n");
        free(h_latent);
        free(h_tokens);
        free(h_q);
        return 1;
    }
    fill_latent(h_latent, (int)latent_elems, seed, WORLD_NOISE_UNIFORM);

    float *d_latent = NULL;
    float *d_patch = NULL;
    float *d_tokens = NULL;
    float *d_q_weight = NULL;
    float *d_q = NULL;
    CUDA_OK(cudaMalloc((void **)&d_latent, latent_elems * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_patch, patch_weight_elems * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_tokens, token_elems * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_q_weight, q_weight_elems * sizeof(float)));
    CUDA_OK(cudaMalloc((void **)&d_q, token_elems * sizeof(float)));

    CUDA_OK(cudaMemcpy(d_latent, h_latent, latent_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_patch, patchify_weight, patch_weight_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_q_weight, q_proj_weight, q_weight_elems * sizeof(float), cudaMemcpyHostToDevice));

    int rows = T * D;
    patchify_f32_kernel<<<rows, 256>>>(d_latent, d_patch, d_tokens, C, H, W, D, ph, pw, Hp, Wp);
    CUDA_OK(cudaGetLastError());

    if (row_major_linear(d_tokens, d_q_weight, d_q, T, D, D)) return 1;

    CUDA_OK(cudaMemcpy(h_tokens, d_tokens, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_q, d_q, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());

    fprintf(stderr, "generation probe: latent -> patchify tokens -> layer0 q_proj\n");
    print_stats("latent", h_latent, (int)latent_elems);
    print_stats("tokens", h_tokens, (int)token_elems);
    print_stats("layer0_q", h_q, (int)token_elems);

    cudaFree(d_latent);
    cudaFree(d_patch);
    cudaFree(d_tokens);
    cudaFree(d_q_weight);
    cudaFree(d_q);
    free(h_latent);
    free(h_tokens);
    free(h_q);
    return 0;
}

extern "C" int world_cuda_layer0_probe(
        const WorldConfig *cfg,
        const WorldLayer0ProbeWeights *weights,
        float sigma,
        unsigned int seed,
        int noise_mode,
        const char *dump_prefix) {
    int rc = 1;
    int C = cfg->channels;
    int H = cfg->height * cfg->patch_h;
    int W = cfg->width * cfg->patch_w;
    int D = cfg->d_model;
    int ph = cfg->patch_h;
    int pw = cfg->patch_w;
    int T = cfg->height * cfg->width;
    int d_head = D / cfg->n_heads;
    int kv_dim = cfg->n_kv_heads * d_head;
    int mlp_hidden = D * cfg->mlp_ratio;
    int ctrl_dim = cfg->n_buttons + 3;
    int d_xy = d_head / 8;
    int d_t = d_head / 4;

    size_t latent_elems = (size_t)C * H * W;
    size_t patch_weight_elems = (size_t)D * C * ph * pw;
    size_t token_elems = (size_t)T * D;
    size_t kv_token_elems = (size_t)T * kv_dim;
    size_t q_rope_elems = (size_t)cfg->n_heads * T * d_head;
    size_t kv_rope_elems = (size_t)cfg->n_kv_heads * T * d_head;

    float *h_latent = (float *)malloc(latent_elems * sizeof(float));
    float *h_noise = (float *)malloc(512 * sizeof(float));
    float *h_tokens = (float *)malloc(token_elems * sizeof(float));
    float *h_cond = (float *)malloc((size_t)D * sizeof(float));
    float *h_s0 = (float *)malloc((size_t)D * sizeof(float));
    float *h_b0 = (float *)malloc((size_t)D * sizeof(float));
    float *h_g0 = (float *)malloc((size_t)D * sizeof(float));
    float *h_s1 = (float *)malloc((size_t)D * sizeof(float));
    float *h_b1 = (float *)malloc((size_t)D * sizeof(float));
    float *h_g1 = (float *)malloc((size_t)D * sizeof(float));
    float *h_norm = (float *)malloc(token_elems * sizeof(float));
    float *h_q_raw = (float *)malloc(token_elems * sizeof(float));
    float *h_k_raw = (float *)malloc(kv_token_elems * sizeof(float));
    float *h_v_raw = (float *)malloc(kv_token_elems * sizeof(float));
    float *h_q = (float *)malloc(q_rope_elems * sizeof(float));
    float *h_k = (float *)malloc(kv_rope_elems * sizeof(float));
    float *h_v = (float *)malloc(kv_rope_elems * sizeof(float));
    float *h_attn = (float *)malloc(token_elems * sizeof(float));
    float *h_attn_out = (float *)malloc(token_elems * sizeof(float));
    float *h_tokens_after_attn = (float *)malloc(token_elems * sizeof(float));
    float *h_ctrl_out = (float *)malloc(token_elems * sizeof(float));
    float *h_tokens_after_ctrl = (float *)malloc(token_elems * sizeof(float));
    float *h_mlp_in = (float *)malloc(token_elems * sizeof(float));
    float *h_mlp_out = (float *)malloc(token_elems * sizeof(float));
    float *h_tokens_after_mlp = (float *)malloc(token_elems * sizeof(float));
    float *h_xy = (float *)malloc((size_t)d_xy * sizeof(float));
    float *h_inv_t = (float *)malloc((size_t)d_t * sizeof(float));
    int64_t *h_x_pos = (int64_t *)malloc((size_t)T * sizeof(int64_t));
    int64_t *h_y_pos = (int64_t *)malloc((size_t)T * sizeof(int64_t));
    int64_t *h_t_pos = (int64_t *)malloc((size_t)T * sizeof(int64_t));

    if (!h_latent || !h_noise || !h_tokens || !h_cond || !h_s0 || !h_b0 || !h_g0 ||
        !h_s1 || !h_b1 || !h_g1 ||
        !h_norm || !h_q_raw || !h_k_raw || !h_v_raw || !h_q || !h_k || !h_v ||
        !h_attn || !h_attn_out || !h_tokens_after_attn ||
        !h_ctrl_out || !h_tokens_after_ctrl || !h_mlp_in || !h_mlp_out || !h_tokens_after_mlp ||
        !h_xy || !h_inv_t || !h_x_pos || !h_y_pos || !h_t_pos) {
        fprintf(stderr, "host allocation failed\n");
        free(h_latent);
        free(h_noise);
        free(h_tokens);
        free(h_cond);
        free(h_s0);
        free(h_b0);
        free(h_g0);
        free(h_s1);
        free(h_b1);
        free(h_g1);
        free(h_norm);
        free(h_q_raw);
        free(h_k_raw);
        free(h_v_raw);
        free(h_q);
        free(h_k);
        free(h_v);
        free(h_attn);
        free(h_attn_out);
        free(h_tokens_after_attn);
        free(h_ctrl_out);
        free(h_tokens_after_ctrl);
        free(h_mlp_in);
        free(h_mlp_out);
        free(h_tokens_after_mlp);
        free(h_xy);
        free(h_inv_t);
        free(h_x_pos);
        free(h_y_pos);
        free(h_t_pos);
        return 1;
    }
    if (weights->initial_latent) {
        memcpy(h_latent, weights->initial_latent, latent_elems * sizeof(float));
    } else {
        fill_latent(h_latent, (int)latent_elems, seed, noise_mode);
    }
    fill_noise_embedding(h_noise, sigma);
    fill_positions(h_x_pos, h_y_pos, h_t_pos, T, cfg->width, 0);
    fill_rope_tables(h_xy, h_inv_t, d_head, cfg->height, cfg->width);

    float *d_latent = NULL;
    float *d_patch = NULL;
    float *d_noise = NULL;
    float *d_noise_hidden = NULL;
    float *d_cond = NULL;
    float *d_cond_act = NULL;
    float *d_denoise_fc1 = NULL;
    float *d_denoise_fc2 = NULL;
    float *d_control_input = NULL;
    float *d_ctrl_emb_fc1_w = NULL;
    float *d_ctrl_emb_fc2_w = NULL;
    float *d_ctrl_emb_hidden = NULL;
    float *d_ctrl_emb = NULL;
    float *d_ctrl_emb_norm = NULL;
    float *d_cond_bias = NULL;
    float *d_cond_s_w = NULL;
    float *d_cond_b_w = NULL;
    float *d_cond_g_w = NULL;
    float *d_mlp_cond_s_w = NULL;
    float *d_mlp_cond_b_w = NULL;
    float *d_mlp_cond_g_w = NULL;
    float *d_s0 = NULL;
    float *d_b0 = NULL;
    float *d_g0 = NULL;
    float *d_s1 = NULL;
    float *d_b1 = NULL;
    float *d_g1 = NULL;
    float *d_tokens = NULL;
    float *d_norm = NULL;
    float *d_q_w = NULL;
    float *d_k_w = NULL;
    float *d_v_w = NULL;
    float *d_out_w = NULL;
    float *d_ctrl_fc1_x_w = NULL;
    float *d_ctrl_fc1_c_w = NULL;
    float *d_ctrl_fc2_w = NULL;
    float *d_dit_mlp_fc1_w = NULL;
    float *d_dit_mlp_fc2_w = NULL;
    float *d_q_raw = NULL;
    float *d_k_raw = NULL;
    float *d_v_raw = NULL;
    float *d_q = NULL;
    float *d_k = NULL;
    float *d_v = NULL;
    float *d_attn = NULL;
    float *d_attn_out = NULL;
    float *d_tokens_after_attn = NULL;
    float *d_ctrl_norm = NULL;
    float *d_ctrl_cond = NULL;
    float *d_ctrl_hidden = NULL;
    float *d_ctrl_out = NULL;
    float *d_tokens_after_ctrl = NULL;
    float *d_mlp_in = NULL;
    float *d_mlp_hidden = NULL;
    float *d_mlp_out = NULL;
    float *d_tokens_after_mlp = NULL;
    float *d_xy_table = NULL;
    float *d_inv_t = NULL;
    int64_t *d_x_pos = NULL;
    int64_t *d_y_pos = NULL;
    int64_t *d_t_pos = NULL;

#define TRY_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto cleanup_device; \
    } \
} while (0)
#define TRY_LINEAR(x, w, y, m, k, n) do { \
    if (row_major_linear((x), (w), (y), (m), (k), (n))) goto cleanup_device; \
} while (0)

    TRY_CUDA(cudaMalloc((void **)&d_latent, latent_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_noise, 512 * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_noise_hidden, (size_t)mlp_hidden * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_cond, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_cond_act, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_control_input, (size_t)ctrl_dim * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_emb_hidden, (size_t)mlp_hidden * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_emb, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_emb_norm, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_s0, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_b0, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_g0, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_s1, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_b1, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_g1, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_tokens, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_norm, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_q_raw, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_k_raw, kv_token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_v_raw, kv_token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_q, q_rope_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_k, kv_rope_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_v, kv_rope_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_attn, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_attn_out, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_tokens_after_attn, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_norm, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_cond, (size_t)D * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_hidden, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_ctrl_out, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_tokens_after_ctrl, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_mlp_in, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_mlp_hidden, (size_t)T * mlp_hidden * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_mlp_out, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_tokens_after_mlp, token_elems * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_xy_table, (size_t)d_xy * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_inv_t, (size_t)d_t * sizeof(float)));
    TRY_CUDA(cudaMalloc((void **)&d_x_pos, (size_t)T * sizeof(int64_t)));
    TRY_CUDA(cudaMalloc((void **)&d_y_pos, (size_t)T * sizeof(int64_t)));
    TRY_CUDA(cudaMalloc((void **)&d_t_pos, (size_t)T * sizeof(int64_t)));

    if (copy_f32_to_device(&d_patch, weights->patchify_weight, patch_weight_elems)) goto cleanup_device;
    if (copy_f32_to_device(&d_denoise_fc1, weights->denoise_fc1_weight, (size_t)mlp_hidden * 512)) goto cleanup_device;
    if (copy_f32_to_device(&d_denoise_fc2, weights->denoise_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)mlp_hidden * ctrl_dim)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (copy_f32_to_device(&d_cond_bias, weights->layer0_cond_bias, (size_t)D)) goto cleanup_device;
    if (copy_f32_to_device(&d_cond_s_w, weights->layer0_attn_cond_s_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_cond_b_w, weights->layer0_attn_cond_b_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_cond_g_w, weights->layer0_attn_cond_g_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_q_w, weights->layer0_q_proj_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_k_w, weights->layer0_k_proj_weight, (size_t)kv_dim * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_v_w, weights->layer0_v_proj_weight, (size_t)kv_dim * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_out_w, weights->layer0_out_proj_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_mlp_cond_s_w, weights->layer0_mlp_cond_s_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_mlp_cond_b_w, weights->layer0_mlp_cond_b_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_mlp_cond_g_w, weights->layer0_mlp_cond_g_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_fc1_x_w, weights->layer0_ctrl_fc1_x_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_fc1_c_w, weights->layer0_ctrl_fc1_c_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_fc2_w, weights->layer0_ctrl_fc2_weight, (size_t)D * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_dit_mlp_fc1_w, weights->layer0_dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto cleanup_device;
    if (copy_f32_to_device(&d_dit_mlp_fc2_w, weights->layer0_dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;

    TRY_CUDA(cudaMemcpy(d_latent, h_latent, latent_elems * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_noise, h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_control_input, weights->control_input, (size_t)ctrl_dim * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_xy_table, h_xy, (size_t)d_xy * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_inv_t, h_inv_t, (size_t)d_t * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_x_pos, h_x_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_y_pos, h_y_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_t_pos, h_t_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));

    TRY_LINEAR(d_control_input, d_ctrl_emb_fc1_w, d_ctrl_emb_hidden, 1, ctrl_dim, mlp_hidden);
    silu_f32_kernel<<<div_up_i64(mlp_hidden, 256), 256>>>(d_ctrl_emb_hidden, d_ctrl_emb_hidden, mlp_hidden);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_emb_hidden, d_ctrl_emb_fc2_w, d_ctrl_emb, 1, mlp_hidden, D);
    rms_norm_rows_f32_kernel<<<1, 256>>>(d_ctrl_emb, d_ctrl_emb_norm, 1, D, 1.0e-6f);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_emb_norm, d_ctrl_fc1_c_w, d_ctrl_cond, 1, D, D);

    TRY_LINEAR(d_noise, d_denoise_fc1, d_noise_hidden, 1, 512, mlp_hidden);
    silu_f32_kernel<<<div_up_i64(mlp_hidden, 256), 256>>>(d_noise_hidden, d_noise_hidden, mlp_hidden);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_noise_hidden, d_denoise_fc2, d_cond, 1, mlp_hidden, D);

    add_bias_silu_f32_kernel<<<div_up_i64(D, 256), 256>>>(d_cond, d_cond_bias, d_cond_act, D);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_cond_act, d_cond_s_w, d_s0, 1, D, D);
    TRY_LINEAR(d_cond_act, d_cond_b_w, d_b0, 1, D, D);
    TRY_LINEAR(d_cond_act, d_cond_g_w, d_g0, 1, D, D);
    TRY_LINEAR(d_cond_act, d_mlp_cond_s_w, d_s1, 1, D, D);
    TRY_LINEAR(d_cond_act, d_mlp_cond_b_w, d_b1, 1, D, D);
    TRY_LINEAR(d_cond_act, d_mlp_cond_g_w, d_g1, 1, D, D);

    patchify_f32_kernel<<<T * D, 256>>>(d_latent, d_patch, d_tokens, C, H, W, D, ph, pw, cfg->height, cfg->width);
    TRY_CUDA(cudaGetLastError());
    ada_rms_norm_single_f32_kernel<<<T, 256>>>(d_tokens, d_s0, d_b0, d_norm, T, D, 1.0e-6f);
    TRY_CUDA(cudaGetLastError());

    TRY_LINEAR(d_norm, d_q_w, d_q_raw, T, D, D);
    TRY_LINEAR(d_norm, d_k_w, d_k_raw, T, D, kv_dim);
    TRY_LINEAR(d_norm, d_v_w, d_v_raw, T, D, kv_dim);

    {
        dim3 grid(T, cfg->n_heads + 2 * cfg->n_kv_heads);
        size_t smem = (size_t)(d_head + 256) * sizeof(float);
        qkv_separate_rms_rope_f32_kernel<<<grid, 256, smem>>>(
            d_q_raw, d_k_raw, d_v_raw,
            d_q, d_k, d_v,
            d_x_pos, d_y_pos, d_t_pos, d_xy_table, d_inv_t,
            T, cfg->n_heads, cfg->n_kv_heads, d_head, cfg->width, cfg->height, 1.0e-6f);
    }
    TRY_CUDA(cudaGetLastError());

    current_frame_attention_f32_kernel<<<cfg->n_heads * T, 256>>>(
        d_q, d_k, d_v, d_attn,
        cfg->n_heads, cfg->n_kv_heads, T, d_head, 1.0f / sqrtf((float)d_head));
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_attn, d_out_w, d_attn_out, T, D, D);
    gated_residual_add_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
        d_tokens, d_attn_out, d_g0, d_tokens_after_attn, T, D);
    TRY_CUDA(cudaGetLastError());

    rms_norm_rows_f32_kernel<<<T, 256>>>(d_tokens_after_attn, d_ctrl_norm, T, D, 1.0e-6f);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_norm, d_ctrl_fc1_x_w, d_ctrl_hidden, T, D, D);
    add_channel_silu_inplace_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(d_ctrl_hidden, d_ctrl_cond, T, D);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_hidden, d_ctrl_fc2_w, d_ctrl_out, T, D, D);
    add_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
        d_tokens_after_attn, d_ctrl_out, d_tokens_after_ctrl, token_elems);
    TRY_CUDA(cudaGetLastError());

    ada_rms_norm_single_f32_kernel<<<T, 256>>>(d_tokens_after_ctrl, d_s1, d_b1, d_mlp_in, T, D, 1.0e-6f);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_mlp_in, d_dit_mlp_fc1_w, d_mlp_hidden, T, D, mlp_hidden);
    silu_f32_kernel<<<div_up_i64((int64_t)T * mlp_hidden, 256), 256>>>(
        d_mlp_hidden, d_mlp_hidden, (int64_t)T * mlp_hidden);
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_mlp_hidden, d_dit_mlp_fc2_w, d_mlp_out, T, mlp_hidden, D);
    gated_residual_add_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
        d_tokens_after_ctrl, d_mlp_out, d_g1, d_tokens_after_mlp, T, D);
    TRY_CUDA(cudaGetLastError());

    TRY_CUDA(cudaMemcpy(h_tokens, d_tokens, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_cond, d_cond, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_s0, d_s0, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_b0, d_b0, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_g0, d_g0, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_s1, d_s1, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_b1, d_b1, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_g1, d_g1, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_norm, d_norm, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_q_raw, d_q_raw, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_k_raw, d_k_raw, kv_token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_v_raw, d_v_raw, kv_token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_q, d_q, q_rope_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_k, d_k, kv_rope_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_v, d_v, kv_rope_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_attn, d_attn, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_attn_out, d_attn_out, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_tokens_after_attn, d_tokens_after_attn, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_ctrl_out, d_ctrl_out, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_tokens_after_ctrl, d_tokens_after_ctrl, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_mlp_in, d_mlp_in, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_mlp_out, d_mlp_out, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaMemcpy(h_tokens_after_mlp, d_tokens_after_mlp, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaDeviceSynchronize());

    fprintf(stderr, "layer0 probe: full layer0 through attention, ctrl fusion, and DiT MLP\n");
    print_stats("latent", h_latent, (int)latent_elems);
    print_stats("tokens", h_tokens, (int)token_elems);
    print_stats("cond", h_cond, D);
    print_stats("layer0_s0", h_s0, D);
    print_stats("layer0_b0", h_b0, D);
    print_stats("layer0_g0", h_g0, D);
    print_stats("layer0_s1", h_s1, D);
    print_stats("layer0_b1", h_b1, D);
    print_stats("layer0_g1", h_g1, D);
    print_stats("layer0_norm", h_norm, (int)token_elems);
    print_stats("layer0_q_raw", h_q_raw, (int)token_elems);
    print_stats("layer0_k_raw", h_k_raw, (int)kv_token_elems);
    print_stats("layer0_v_raw", h_v_raw, (int)kv_token_elems);
    print_stats("layer0_q_rope", h_q, (int)q_rope_elems);
    print_stats("layer0_k_rope", h_k, (int)kv_rope_elems);
    print_stats("layer0_v", h_v, (int)kv_rope_elems);
    print_stats("layer0_attn", h_attn, (int)token_elems);
    print_stats("layer0_attn_out", h_attn_out, (int)token_elems);
    print_stats("layer0_tokens_after_attn", h_tokens_after_attn, (int)token_elems);
    print_stats("layer0_ctrl_out", h_ctrl_out, (int)token_elems);
    print_stats("layer0_tokens_after_ctrl", h_tokens_after_ctrl, (int)token_elems);
    print_stats("layer0_mlp_in", h_mlp_in, (int)token_elems);
    print_stats("layer0_mlp_out", h_mlp_out, (int)token_elems);
    print_stats("layer0_tokens_after_mlp", h_tokens_after_mlp, (int)token_elems);

    if (dump_f32(dump_prefix, "latent", h_latent, latent_elems) ||
        dump_f32(dump_prefix, "tokens", h_tokens, token_elems) ||
        dump_f32(dump_prefix, "cond", h_cond, (size_t)D) ||
        dump_f32(dump_prefix, "s0", h_s0, (size_t)D) ||
        dump_f32(dump_prefix, "b0", h_b0, (size_t)D) ||
        dump_f32(dump_prefix, "g0", h_g0, (size_t)D) ||
        dump_f32(dump_prefix, "s1", h_s1, (size_t)D) ||
        dump_f32(dump_prefix, "b1", h_b1, (size_t)D) ||
        dump_f32(dump_prefix, "g1", h_g1, (size_t)D) ||
        dump_f32(dump_prefix, "norm", h_norm, token_elems) ||
        dump_f32(dump_prefix, "q_raw", h_q_raw, token_elems) ||
        dump_f32(dump_prefix, "k_raw", h_k_raw, kv_token_elems) ||
        dump_f32(dump_prefix, "v_raw", h_v_raw, kv_token_elems) ||
        dump_f32(dump_prefix, "q", h_q, q_rope_elems) ||
        dump_f32(dump_prefix, "k", h_k, kv_rope_elems) ||
        dump_f32(dump_prefix, "v", h_v, kv_rope_elems) ||
        dump_f32(dump_prefix, "attn", h_attn, token_elems) ||
        dump_f32(dump_prefix, "attn_out", h_attn_out, token_elems) ||
        dump_f32(dump_prefix, "tokens_after_attn", h_tokens_after_attn, token_elems) ||
        dump_f32(dump_prefix, "ctrl_out", h_ctrl_out, token_elems) ||
        dump_f32(dump_prefix, "tokens_after_ctrl", h_tokens_after_ctrl, token_elems) ||
        dump_f32(dump_prefix, "mlp_in", h_mlp_in, token_elems) ||
        dump_f32(dump_prefix, "mlp_out", h_mlp_out, token_elems) ||
        dump_f32(dump_prefix, "tokens_after_mlp", h_tokens_after_mlp, token_elems)) {
        goto cleanup_device;
    }

    rc = 0;

cleanup_device:
    cudaFree(d_latent);
    cudaFree(d_patch);
    cudaFree(d_noise);
    cudaFree(d_noise_hidden);
    cudaFree(d_cond);
    cudaFree(d_cond_act);
    cudaFree(d_denoise_fc1);
    cudaFree(d_denoise_fc2);
    cudaFree(d_control_input);
    cudaFree(d_ctrl_emb_fc1_w);
    cudaFree(d_ctrl_emb_fc2_w);
    cudaFree(d_ctrl_emb_hidden);
    cudaFree(d_ctrl_emb);
    cudaFree(d_ctrl_emb_norm);
    cudaFree(d_cond_bias);
    cudaFree(d_cond_s_w);
    cudaFree(d_cond_b_w);
    cudaFree(d_cond_g_w);
    cudaFree(d_mlp_cond_s_w);
    cudaFree(d_mlp_cond_b_w);
    cudaFree(d_mlp_cond_g_w);
    cudaFree(d_s0);
    cudaFree(d_b0);
    cudaFree(d_g0);
    cudaFree(d_s1);
    cudaFree(d_b1);
    cudaFree(d_g1);
    cudaFree(d_tokens);
    cudaFree(d_norm);
    cudaFree(d_q_w);
    cudaFree(d_k_w);
    cudaFree(d_v_w);
    cudaFree(d_out_w);
    cudaFree(d_ctrl_fc1_x_w);
    cudaFree(d_ctrl_fc1_c_w);
    cudaFree(d_ctrl_fc2_w);
    cudaFree(d_dit_mlp_fc1_w);
    cudaFree(d_dit_mlp_fc2_w);
    cudaFree(d_q_raw);
    cudaFree(d_k_raw);
    cudaFree(d_v_raw);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_attn);
    cudaFree(d_attn_out);
    cudaFree(d_tokens_after_attn);
    cudaFree(d_ctrl_norm);
    cudaFree(d_ctrl_cond);
    cudaFree(d_ctrl_hidden);
    cudaFree(d_ctrl_out);
    cudaFree(d_tokens_after_ctrl);
    cudaFree(d_mlp_in);
    cudaFree(d_mlp_hidden);
    cudaFree(d_mlp_out);
    cudaFree(d_tokens_after_mlp);
    cudaFree(d_xy_table);
    cudaFree(d_inv_t);
    cudaFree(d_x_pos);
    cudaFree(d_y_pos);
    cudaFree(d_t_pos);

#undef TRY_CUDA
#undef TRY_LINEAR

    free(h_latent);
    free(h_noise);
    free(h_tokens);
    free(h_cond);
    free(h_s0);
    free(h_b0);
    free(h_g0);
    free(h_s1);
    free(h_b1);
    free(h_g1);
    free(h_norm);
    free(h_q_raw);
    free(h_k_raw);
    free(h_v_raw);
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_attn);
    free(h_attn_out);
    free(h_tokens_after_attn);
    free(h_ctrl_out);
    free(h_tokens_after_ctrl);
    free(h_mlp_in);
    free(h_mlp_out);
    free(h_tokens_after_mlp);
    free(h_xy);
    free(h_inv_t);
    free(h_x_pos);
    free(h_y_pos);
    free(h_t_pos);
    return rc;
}

extern "C" int world_cuda_transformer_probe(
        const WorldConfig *cfg,
        const WorldModelProbeWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frames_to_run,
        int frame_idx,
        int cache_pass,
        float sigma,
        unsigned int seed,
        int noise_mode,
        const char *dump_prefix,
        const WorldVaeDecoderWeights *vae,
        const char *out_path) {
    int rc = 1;
    int C = cfg->channels;
    int H = cfg->height * cfg->patch_h;
    int W = cfg->width * cfg->patch_w;
    int D = cfg->d_model;
    int ph = cfg->patch_h;
    int pw = cfg->patch_w;
    int T = cfg->height * cfg->width;
    int d_head = D / cfg->n_heads;
    int kv_dim = cfg->n_kv_heads * d_head;
    int mlp_hidden = D * cfg->mlp_ratio;
    int ctrl_dim = cfg->n_buttons + 3;
    int d_xy = d_head / 8;
    int d_t = d_head / 4;

    if (layers_to_run <= 0 || layers_to_run > weights->n_layers) {
        fprintf(stderr, "invalid layers_to_run=%d n_layers=%d\n", layers_to_run, weights->n_layers);
        return 1;
    }
    if (steps_to_run <= 0 || steps_to_run >= cfg->scheduler_sigmas_count) {
        fprintf(stderr, "invalid steps_to_run=%d scheduler_count=%d\n", steps_to_run, cfg->scheduler_sigmas_count);
        return 1;
    }
    if (frames_to_run <= 0) {
        fprintf(stderr, "invalid frames_to_run=%d\n", frames_to_run);
        return 1;
    }
    if (frame_idx < 0) {
        fprintf(stderr, "invalid frame_idx=%d\n", frame_idx);
        return 1;
    }

    int fps_div = cfg->temporal_compression > 0 ? cfg->inference_fps / cfg->temporal_compression : 0;
    int frame_stride = fps_div > 0 ? cfg->base_fps / fps_div : 1;

    size_t latent_elems = (size_t)C * H * W;
    size_t patch_weight_elems = (size_t)D * C * ph * pw;
    size_t token_elems = (size_t)T * D;
    size_t kv_token_elems = (size_t)T * kv_dim;
    size_t qkv_token_elems = token_elems + 2 * kv_token_elems;
    size_t q_rope_elems = (size_t)cfg->n_heads * T * d_head;
    size_t kv_rope_elems = (size_t)cfg->n_kv_heads * T * d_head;
    size_t out_norm_weight_elems = (size_t)2 * D * D;
    size_t unpatch_weight_elems = (size_t)D * C * ph * pw;
    int out_dim = C * ph * pw;
    int total_passes = steps_to_run + (cache_pass ? 1 : 0);
    int decoded_frames_per_latent = 4;

    float *h_latent = (float *)malloc(latent_elems * sizeof(float));
    float *h_noise = (float *)malloc(512 * sizeof(float));
    float *h_tokens = (float *)malloc(token_elems * sizeof(float));
    float *h_latent_out = (float *)malloc(latent_elems * sizeof(float));
    float *h_xy = (float *)malloc((size_t)d_xy * sizeof(float));
    float *h_inv_t = (float *)malloc((size_t)d_t * sizeof(float));
    int64_t *h_x_pos = (int64_t *)malloc((size_t)T * sizeof(int64_t));
    int64_t *h_y_pos = (int64_t *)malloc((size_t)T * sizeof(int64_t));
    int64_t *h_t_pos = (int64_t *)malloc((size_t)T * sizeof(int64_t));
    if (!h_latent || !h_noise || !h_tokens || !h_latent_out || !h_xy || !h_inv_t || !h_x_pos || !h_y_pos || !h_t_pos) {
        fprintf(stderr, "host allocation failed\n");
        free(h_latent);
        free(h_noise);
        free(h_tokens);
        free(h_latent_out);
        free(h_xy);
        free(h_inv_t);
        free(h_x_pos);
        free(h_y_pos);
        free(h_t_pos);
        return 1;
    }
    fill_rope_tables(h_xy, h_inv_t, d_head, cfg->height, cfg->width);

    float *d_latent = NULL;
    float *d_patch = NULL;
    float *d_noise = NULL;
    float *d_noise_hidden = NULL;
    float *d_cond = NULL;
    float *d_cond_act = NULL;
    float *d_denoise_fc1 = NULL;
    float *d_denoise_fc2 = NULL;
    float *d_control_input = NULL;
    float *d_ctrl_emb_fc1_w = NULL;
    float *d_ctrl_emb_fc2_w = NULL;
    float *d_ctrl_emb_hidden = NULL;
    float *d_ctrl_emb = NULL;
    float *d_ctrl_emb_norm = NULL;
    float *d_out_norm_w = NULL;
    float *d_out_mod = NULL;
    float *d_final_tokens = NULL;
    float *d_unpatch_w = NULL;
    float *d_unpatch_b = NULL;
    float *d_latent_out = NULL;
    float *d_layer_mod = NULL;
    float *d_tokens = NULL;
    float *d_norm = NULL;
    float *d_qkv_raw = NULL;
    float *d_q = NULL;
    float *d_k = NULL;
    float *d_v = NULL;
    float *d_v_first = NULL;
    float *d_attn = NULL;
    float *d_attn_out = NULL;
    float *d_tokens_after_attn = NULL;
    float *d_ctrl_norm = NULL;
    float *d_ctrl_cond_by_layer = NULL;
    float *d_ctrl_hidden = NULL;
    float *d_ctrl_out = NULL;
    float *d_tokens_after_ctrl = NULL;
    float *d_mlp_in = NULL;
    float *d_mlp_hidden = NULL;
    float *d_mlp_out = NULL;
    float *d_tokens_after_mlp = NULL;
    float *d_xy_table = NULL;
    float *d_inv_t = NULL;
    int64_t *d_x_pos = NULL;
    int64_t *d_y_pos = NULL;
    int64_t *d_t_pos = NULL;
    DeviceWorldLayerWeights *d_layers = NULL;
    DeviceWorldLayerCache *d_caches = NULL;
    DeviceVaeDecoder d_vae;
    memset(&d_vae, 0, sizeof(d_vae));
    int have_d_vae = 0;

#define TRY_CUDA2(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto cleanup_device; \
    } \
} while (0)
#define TRY_LINEAR2(x, w, y, m, k, n) do { \
    if (row_major_linear((x), (w), (y), (m), (k), (n))) goto cleanup_device; \
} while (0)

    TRY_CUDA2(cudaMalloc((void **)&d_latent, latent_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_noise, 512 * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_noise_hidden, (size_t)mlp_hidden * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_cond, (size_t)D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_cond_act, (size_t)D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_control_input, (size_t)ctrl_dim * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_emb_hidden, (size_t)mlp_hidden * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_emb, (size_t)D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_emb_norm, (size_t)D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_out_mod, (size_t)2 * D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_final_tokens, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_latent_out, latent_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_layer_mod, (size_t)6 * D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_tokens, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_norm, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_qkv_raw, qkv_token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_q, q_rope_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_k, kv_rope_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_v, kv_rope_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_v_first, kv_rope_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_attn, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_attn_out, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_tokens_after_attn, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_norm, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_cond_by_layer, (size_t)layers_to_run * D * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_hidden, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_ctrl_out, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_tokens_after_ctrl, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_mlp_in, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_mlp_hidden, (size_t)T * mlp_hidden * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_mlp_out, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_tokens_after_mlp, token_elems * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_xy_table, (size_t)d_xy * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_inv_t, (size_t)d_t * sizeof(float)));
    TRY_CUDA2(cudaMalloc((void **)&d_x_pos, (size_t)T * sizeof(int64_t)));
    TRY_CUDA2(cudaMalloc((void **)&d_y_pos, (size_t)T * sizeof(int64_t)));
    TRY_CUDA2(cudaMalloc((void **)&d_t_pos, (size_t)T * sizeof(int64_t)));

    if (copy_f32_to_device(&d_patch, weights->patchify_weight, patch_weight_elems)) goto cleanup_device;
    if (copy_f32_to_device(&d_denoise_fc1, weights->denoise_fc1_weight, (size_t)mlp_hidden * 512)) goto cleanup_device;
    if (copy_f32_to_device(&d_denoise_fc2, weights->denoise_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)mlp_hidden * ctrl_dim)) goto cleanup_device;
    if (copy_f32_to_device(&d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (copy_f32_to_device(&d_out_norm_w, weights->out_norm_fc_weight, out_norm_weight_elems)) goto cleanup_device;
    if (copy_f32_to_device(&d_unpatch_w, weights->unpatchify_weight, unpatch_weight_elems)) goto cleanup_device;
    if (copy_f32_to_device(&d_unpatch_b, weights->unpatchify_bias, (size_t)C)) goto cleanup_device;
    if (copy_world_layers_to_device(
                &d_layers,
                weights->layers,
                layers_to_run,
                D,
                kv_dim,
                mlp_hidden,
                0,
                0,
                0,
                0)) goto cleanup_device;
    if (alloc_device_world_caches(&d_caches, cfg, layers_to_run, T, cfg->n_kv_heads, d_head, 0)) goto cleanup_device;

    TRY_CUDA2(cudaMemcpy(d_xy_table, h_xy, (size_t)d_xy * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA2(cudaMemcpy(d_inv_t, h_inv_t, (size_t)d_t * sizeof(float), cudaMemcpyHostToDevice));
    if (vae && out_path && out_path[0]) {
        if (taehv_decoder_init(&d_vae, cfg, vae)) goto cleanup_device;
        have_d_vae = 1;
    }

    for (int frame_ordinal = 0; frame_ordinal < frames_to_run; ++frame_ordinal) {
        int current_frame_idx = frame_idx + frame_ordinal;
        int frame_timestamp = current_frame_idx * frame_stride;
        TRY_CUDA2(cudaMemcpy(
            d_control_input,
            weights->control_inputs + (size_t)frame_ordinal * ctrl_dim,
            (size_t)ctrl_dim * sizeof(float),
            cudaMemcpyHostToDevice));
        TRY_LINEAR2(d_control_input, d_ctrl_emb_fc1_w, d_ctrl_emb_hidden, 1, ctrl_dim, mlp_hidden);
        silu_f32_kernel<<<div_up_i64(mlp_hidden, 256), 256>>>(d_ctrl_emb_hidden, d_ctrl_emb_hidden, mlp_hidden);
        TRY_CUDA2(cudaGetLastError());
        TRY_LINEAR2(d_ctrl_emb_hidden, d_ctrl_emb_fc2_w, d_ctrl_emb, 1, mlp_hidden, D);
        rms_norm_rows_f32_kernel<<<1, 256>>>(d_ctrl_emb, d_ctrl_emb_norm, 1, D, 1.0e-6f);
        TRY_CUDA2(cudaGetLastError());
        for (int layer_idx = 0; layer_idx < layers_to_run; ++layer_idx) {
            const DeviceWorldLayerWeights *lw = &d_layers[layer_idx];
            if (lw->has_ctrl) {
                TRY_LINEAR2(d_ctrl_emb_norm, lw->ctrl_fc1_c_weight,
                            d_ctrl_cond_by_layer + (size_t)layer_idx * D,
                            1, D, D);
            }
        }
        if (weights->initial_latents) {
            memcpy(h_latent,
                   weights->initial_latents + (size_t)frame_ordinal * latent_elems,
                   latent_elems * sizeof(float));
        } else {
            fill_latent(h_latent, (int)latent_elems, seed + (unsigned int)frame_ordinal, noise_mode);
        }
        fill_positions(h_x_pos, h_y_pos, h_t_pos, T, cfg->width, frame_timestamp);
        TRY_CUDA2(cudaMemcpy(d_latent, h_latent, latent_elems * sizeof(float), cudaMemcpyHostToDevice));
        TRY_CUDA2(cudaMemcpy(d_x_pos, h_x_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
        TRY_CUDA2(cudaMemcpy(d_y_pos, h_y_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
        TRY_CUDA2(cudaMemcpy(d_t_pos, h_t_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
        fprintf(stderr,
                "frame %02d/%02d: frame_idx=%d frame_timestamp=%d\n",
                frame_ordinal + 1, frames_to_run, current_frame_idx, frame_timestamp);

    for (int pass_idx = 0; pass_idx < total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= steps_to_run;
        int frozen_pass = !is_cache_pass;
        float sigma_step = is_cache_pass ? 0.0f : (steps_to_run == 1 ? sigma : cfg->scheduler_sigmas[pass_idx]);
        float next_sigma = is_cache_pass ? 0.0f : (steps_to_run == 1 ? cfg->scheduler_sigmas[1] : cfg->scheduler_sigmas[pass_idx + 1]);
        float dsigma = next_sigma - sigma_step;
        int is_last_step = !is_cache_pass && pass_idx == steps_to_run - 1;
        if (is_cache_pass) {
            fprintf(stderr, "cache pass: sigma=0 frame_idx=%d frozen=false\n", current_frame_idx);
        } else {
            fprintf(stderr,
                    "scheduler step %02d/%02d: sigma=%.6g next=%.6g dsigma=%.6g frame_idx=%d frozen=true\n",
                    pass_idx + 1, steps_to_run, sigma_step, next_sigma, dsigma, current_frame_idx);
        }

        fill_noise_embedding(h_noise, sigma_step);
        TRY_CUDA2(cudaMemcpy(d_noise, h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
        TRY_LINEAR2(d_noise, d_denoise_fc1, d_noise_hidden, 1, 512, mlp_hidden);
        silu_f32_kernel<<<div_up_i64(mlp_hidden, 256), 256>>>(d_noise_hidden, d_noise_hidden, mlp_hidden);
        TRY_CUDA2(cudaGetLastError());
        TRY_LINEAR2(d_noise_hidden, d_denoise_fc2, d_cond, 1, mlp_hidden, D);

        patchify_f32_kernel<<<T * D, 256>>>(d_latent, d_patch, d_tokens, C, H, W, D, ph, pw, cfg->height, cfg->width);
        TRY_CUDA2(cudaGetLastError());
        float *d_tokens_cur = d_tokens;
        float *d_tokens_next = d_tokens_after_mlp;

        for (int layer_idx = 0; layer_idx < layers_to_run; ++layer_idx) {
            const DeviceWorldLayerWeights *lw = &d_layers[layer_idx];
            DeviceWorldLayerCache *cache = &d_caches[layer_idx];
            fprintf(stderr, "  standalone layer %02d/%02d\n", layer_idx, layers_to_run);

            add_bias_silu_f32_kernel<<<div_up_i64(D, 256), 256>>>(d_cond, lw->cond_bias, d_cond_act, D);
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_cond_act, lw->cond_proj_weight, d_layer_mod, 1, D, 6 * D);
            float *d_s0 = d_layer_mod;
            float *d_b0 = d_layer_mod + D;
            float *d_g0 = d_layer_mod + 2 * D;
            float *d_s1 = d_layer_mod + 3 * D;
            float *d_b1 = d_layer_mod + 4 * D;
            float *d_g1 = d_layer_mod + 5 * D;

            ada_rms_norm_single_f32_kernel<<<T, 256>>>(d_tokens_cur, d_s0, d_b0, d_norm, T, D, 1.0e-6f);
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_norm, lw->qkv_proj_weight, d_qkv_raw, T, D, D + 2 * kv_dim);
            float *d_v_cur = (cfg->value_residual && layer_idx == 0) ? d_v_first : d_v;

            {
                dim3 grid(T, cfg->n_heads + 2 * cfg->n_kv_heads);
                size_t smem = (size_t)(d_head + 256) * sizeof(float);
                qkv_fused_rms_rope_f32_kernel<<<grid, 256, smem>>>(
                    d_qkv_raw,
                    d_q, d_k, d_v_cur,
                    d_x_pos, d_y_pos, d_t_pos, d_xy_table, d_inv_t,
                    T, cfg->n_heads, cfg->n_kv_heads, d_head, cfg->width, cfg->height, 1.0e-6f);
            }
            TRY_CUDA2(cudaGetLastError());

            if (cfg->value_residual) {
                if (layer_idx != 0) {
                    lerp_inplace_f32_kernel<<<div_up_i64((int64_t)kv_rope_elems, 256), 256>>>(
                        d_v, d_v_first, lw->v_lamb, (int64_t)kv_rope_elems);
                    TRY_CUDA2(cudaGetLastError());
                }
            }

            {
                int bucket = (current_frame_idx + (cache->pinned_dilation - 1)) / cache->pinned_dilation;
                int num_buckets = (cache->ring_length / T) / cache->pinned_dilation;
                int base = (bucket % num_buckets) * T;
                bool write_step = (current_frame_idx % cache->pinned_dilation) == 0;
                kv_cache_upsert_copy_f32_kernel<<<div_up_i64((int64_t)cfg->n_kv_heads * T * d_head, 256), 256>>>(
                    cache->k, cache->v, d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, T, d_head, cache->ring_length, base, write_step, (bool)frozen_pass);
                TRY_CUDA2(cudaGetLastError());
                collect_cache_frame_indices_kernel<<<cache->capacity / T, 256>>>(
                    cache->written, cache->indices, cache->block_ids, cache->index_count,
                    cache->capacity, T, base, write_step);
                TRY_CUDA2(cudaGetLastError());
                indexed_attention_cache_f32_kernel<<<cfg->n_heads * T, 256>>>(
                    d_q, cache->k, cache->v, cache->indices, cache->index_count, d_attn,
                    cfg->n_heads, cfg->n_kv_heads, T, cache->capacity, d_head,
                    1.0f / sqrtf((float)d_head));
            }
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_attn, lw->out_proj_weight, d_attn_out, T, D, D);
            gated_residual_add_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
                d_tokens_cur, d_attn_out, d_g0, d_tokens_after_attn, T, D);
            TRY_CUDA2(cudaGetLastError());

            float *d_tokens_ctrl = d_tokens_after_attn;
            if (lw->has_ctrl) {
                rms_norm_rows_f32_kernel<<<T, 256>>>(d_tokens_after_attn, d_ctrl_norm, T, D, 1.0e-6f);
                TRY_CUDA2(cudaGetLastError());
                TRY_LINEAR2(d_ctrl_norm, lw->ctrl_fc1_x_weight, d_ctrl_hidden, T, D, D);
                add_channel_silu_inplace_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
                    d_ctrl_hidden, d_ctrl_cond_by_layer + (size_t)layer_idx * D, T, D);
                TRY_CUDA2(cudaGetLastError());
                TRY_LINEAR2(d_ctrl_hidden, lw->ctrl_fc2_weight, d_ctrl_out, T, D, D);
                add_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
                    d_tokens_after_attn, d_ctrl_out, d_tokens_after_ctrl, token_elems);
                TRY_CUDA2(cudaGetLastError());
                d_tokens_ctrl = d_tokens_after_ctrl;
            }

            ada_rms_norm_single_f32_kernel<<<T, 256>>>(d_tokens_ctrl, d_s1, d_b1, d_mlp_in, T, D, 1.0e-6f);
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_mlp_in, lw->dit_mlp_fc1_weight, d_mlp_hidden, T, D, mlp_hidden);
            silu_f32_kernel<<<div_up_i64((int64_t)T * mlp_hidden, 256), 256>>>(
                d_mlp_hidden, d_mlp_hidden, (int64_t)T * mlp_hidden);
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_mlp_hidden, lw->dit_mlp_fc2_weight, d_mlp_out, T, mlp_hidden, D);
            gated_residual_add_f32_kernel<<<div_up_i64(token_elems, 256), 256>>>(
                d_tokens_ctrl, d_mlp_out, d_g1, d_tokens_next, T, D);
            TRY_CUDA2(cudaGetLastError());

            float *d_swap = d_tokens_cur;
            d_tokens_cur = d_tokens_next;
            d_tokens_next = d_swap;
        }

        if (is_cache_pass) {
            continue;
        }

        silu_f32_kernel<<<div_up_i64(D, 256), 256>>>(d_cond, d_cond_act, D);
        TRY_CUDA2(cudaGetLastError());
        TRY_LINEAR2(d_cond_act, d_out_norm_w, d_out_mod, 1, D, 2 * D);
        out_norm_silu_f32_kernel<<<T, 256>>>(d_tokens_cur, d_out_mod, d_final_tokens, T, D, 1.0e-6f);
        TRY_CUDA2(cudaGetLastError());
        unpatchify_orig_f32_kernel<<<T * out_dim, 256>>>(
            d_final_tokens, d_unpatch_w, d_unpatch_b, d_latent_out,
            T, D, C, H, W, ph, pw, cfg->width, out_dim);
        TRY_CUDA2(cudaGetLastError());

        if (is_last_step) {
            TRY_CUDA2(cudaMemcpy(h_tokens, d_tokens_cur, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
            TRY_CUDA2(cudaMemcpy(h_latent_out, d_latent_out, latent_elems * sizeof(float), cudaMemcpyDeviceToHost));
        }

        latent_update_f32_kernel<<<div_up_i64((int64_t)latent_elems, 256), 256>>>(
            d_latent, d_latent_out, dsigma, (int64_t)latent_elems);
        TRY_CUDA2(cudaGetLastError());
    }

        if (have_d_vae &&
            world_cuda_decode_vae_to_ppm(cfg, &d_vae, d_latent, out_path, frame_ordinal * decoded_frames_per_latent)) {
            goto cleanup_device;
        }
    }

    TRY_CUDA2(cudaMemcpy(h_latent, d_latent, latent_elems * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA2(cudaDeviceSynchronize());
    fprintf(stderr,
            "transformer probe: completed %d frame(s) x %d scheduler steps x %d layers\n",
            frames_to_run, steps_to_run, layers_to_run);
    print_stats("transformer_tokens", h_tokens, (int)token_elems);
    print_stats("latent_out", h_latent_out, (int)latent_elems);
    print_stats("latent_final", h_latent, (int)latent_elems);
    if (cache_pass && dump_cache_written_counts(dump_prefix, d_caches, layers_to_run)) goto cleanup_device;
    if (dump_f32(dump_prefix, "transformer_tokens", h_tokens, token_elems) ||
        dump_f32(dump_prefix, "latent_out", h_latent_out, latent_elems) ||
        dump_f32(dump_prefix, "latent_final", h_latent, latent_elems)) goto cleanup_device;
    rc = 0;

cleanup_device:
    taehv_decoder_free(&d_vae);
    free_device_world_layers(d_layers, layers_to_run);
    free_device_world_caches(d_caches, layers_to_run);
    cudaFree(d_latent);
    cudaFree(d_patch);
    cudaFree(d_noise);
    cudaFree(d_noise_hidden);
    cudaFree(d_cond);
    cudaFree(d_cond_act);
    cudaFree(d_denoise_fc1);
    cudaFree(d_denoise_fc2);
    cudaFree(d_control_input);
    cudaFree(d_ctrl_emb_fc1_w);
    cudaFree(d_ctrl_emb_fc2_w);
    cudaFree(d_ctrl_emb_hidden);
    cudaFree(d_ctrl_emb);
    cudaFree(d_ctrl_emb_norm);
    cudaFree(d_out_norm_w);
    cudaFree(d_out_mod);
    cudaFree(d_final_tokens);
    cudaFree(d_unpatch_w);
    cudaFree(d_unpatch_b);
    cudaFree(d_latent_out);
    cudaFree(d_layer_mod);
    cudaFree(d_tokens);
    cudaFree(d_norm);
    cudaFree(d_qkv_raw);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_v_first);
    cudaFree(d_attn);
    cudaFree(d_attn_out);
    cudaFree(d_tokens_after_attn);
    cudaFree(d_ctrl_norm);
    cudaFree(d_ctrl_cond_by_layer);
    cudaFree(d_ctrl_hidden);
    cudaFree(d_ctrl_out);
    cudaFree(d_tokens_after_ctrl);
    cudaFree(d_mlp_in);
    cudaFree(d_mlp_hidden);
    cudaFree(d_mlp_out);
    cudaFree(d_tokens_after_mlp);
    cudaFree(d_xy_table);
    cudaFree(d_inv_t);
    cudaFree(d_x_pos);
    cudaFree(d_y_pos);
    cudaFree(d_t_pos);

#undef TRY_CUDA2
#undef TRY_LINEAR2

    free(h_latent);
    free(h_noise);
    free(h_tokens);
    free(h_latent_out);
    free(h_xy);
    free(h_inv_t);
    free(h_x_pos);
    free(h_y_pos);
    free(h_t_pos);
    return rc;
}
