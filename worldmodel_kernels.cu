#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#include <cmath>
#include <climits>
#include <cstdint>
#include <vector>

#define WM_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define WM_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define WM_CHECK_F32(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")

#define WM_ATTN_D64_Q_BLOCK 4
#define WM_ATTN_D64_K_BLOCK 64
#define WM_ATTN_D64_FLASH_WARPS 16

static int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

static void check_last_cuda_error(const char *name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
}

static void check_cutlass_status(cutlass::Status status, const char *name) {
    TORCH_CHECK(status == cutlass::Status::kSuccess, name, " failed: ", cutlassGetStatusString(status));
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

__global__ void silu_f32_kernel(const float *x, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = wm_silu(x[i]);
    }
}

__global__ void rms_norm_f32_kernel(const float *x, float *y, int rows, int dim, float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    const float *row_x = x + (int64_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = row_x[d];
        sum += v * v;
    }

    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)dim + eps);
    float *row_y = y + (int64_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        row_y[d] = row_x[d] * inv;
    }
}

__global__ void ada_rms_norm_f32_kernel(
        const float *x,
        const float *scale,
        const float *bias,
        float *y,
        int B,
        int T,
        int N,
        int D,
        float eps) {
    __shared__ float red[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / T;
    int t = row - b * T;
    int m = T / N;
    int n = t / m;

    const float *row_x = x + (int64_t)row * D;
    const float *row_s = scale + ((int64_t)b * N + n) * D;
    const float *row_b = bias + ((int64_t)b * N + n) * D;

    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = row_x[d];
        sum += v * v;
    }

    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)D + eps);
    float *row_y = y + (int64_t)row * D;
    for (int d = tid; d < D; d += blockDim.x) {
        row_y[d] = row_x[d] * inv * (1.0f + row_s[d]) + row_b[d];
    }
}

__global__ void ortho_rope_f32_kernel(
        const float *x,
        float *y,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int B,
        int H,
        int T,
        int D,
        int width,
        int height) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    int64_t total = (int64_t)B * H * T * half;
    if (i >= total) return;

    int p = i % half;
    int64_t q = i / half;
    int t = q % T;
    q /= T;
    int h = q % H;
    int b = q / H;

    int d_xy = D / 8;
    float phase = wm_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, width, height, d_xy);
    float c = cosf(phase);
    float s = sinf(phase);

    int64_t base = (((int64_t)b * H + h) * T + t) * D;
    float x0 = x[base + 2 * p];
    float x1 = x[base + 2 * p + 1];
    y[base + p] = x0 * c - x1 * s;
    y[base + half + p] = x1 * c + x0 * s;
}

__global__ void qkv_rms_rope_f32_kernel(
        const float *qkv,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int B,
        int T,
        int n_heads,
        int n_kv_heads,
        int D,
        int width,
        int height,
        float eps) {
    extern __shared__ float sh[];
    float *vals = sh;
    float *red = sh + D;

    int bt = blockIdx.x;
    int role = blockIdx.y;
    int tid = threadIdx.x;
    int b = bt / T;
    int t = bt - b * T;
    int total_heads = n_heads + 2 * n_kv_heads;
    int qkv_stride = total_heads * D;
    int half = D / 2;
    int d_xy = D / 8;

    if (role < n_heads + n_kv_heads) {
        int is_k = role >= n_heads;
        int h = is_k ? role - n_heads : role;
        int src_head = is_k ? n_heads + h : h;
        const float *src = qkv + ((int64_t)b * T + t) * qkv_stride + src_head * D;

        float sum = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
            float z = src[d];
            vals[d] = z;
            sum += z * z;
        }
        red[tid] = sum;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) {
                red[tid] += red[tid + step];
            }
            __syncthreads();
        }

        float inv = rsqrtf(red[0] / (float)D + eps);
        float *dst = is_k
            ? k + (((int64_t)b * n_kv_heads + h) * T + t) * D
            : q + (((int64_t)b * n_heads + h) * T + t) * D;

        for (int p = tid; p < half; p += blockDim.x) {
            float phase = wm_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, width, height, d_xy);
            float c = cosf(phase);
            float s = sinf(phase);
            float a = vals[2 * p] * inv;
            float bb = vals[2 * p + 1] * inv;
            dst[p] = a * c - bb * s;
            dst[half + p] = bb * c + a * s;
        }
    } else {
        int h = role - n_heads - n_kv_heads;
        const float *src = qkv + ((int64_t)b * T + t) * qkv_stride + (n_heads + n_kv_heads + h) * D;
        float *dst = v + (((int64_t)b * n_kv_heads + h) * T + t) * D;
        for (int d = tid; d < D; d += blockDim.x) {
            dst[d] = src[d];
        }
    }
}

__global__ void masked_attention_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        const bool *written,
        float *out,
        int B,
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
    int hq = (row / Tq) % Hq;
    int b = row / (Tq * Hq);
    int group = Hq / Hkv;
    int hk = hq / group;

    const float *qrow = q + (((int64_t)b * Hq + hq) * Tq + tq) * D;
    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * D;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * D;
    float *orow = out + (((int64_t)b * Hq + hq) * Tq + tq) * D;

    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int tk = 0; tk < Tk; ++tk) {
        float partial = 0.0f;
        if (written[tk]) {
            const float *krow = kbase + (int64_t)tk * D;
            for (int d = tid; d < D; d += blockDim.x) {
                partial += qrow[d] * krow[d];
            }
        }

        red[tid] = partial;
        __syncthreads();

        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (tid < step) {
                red[tid] += red[tid + step];
            }
            __syncthreads();
        }

        if (!written[tk]) {
            continue;
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
        orow[tid] = l > 0.0f ? acc / l : 0.0f;
    }
}

__global__ void indexed_attention_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        const int64_t *indices,
        float *out,
        int B,
        int Hq,
        int Hkv,
        int Tq,
        int Nkv,
        int Tk,
        int D,
        float scale) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int tq = row % Tq;
    int hq = (row / Tq) % Hq;
    int b = row / (Tq * Hq);
    int group = Hq / Hkv;
    int hk = hq / group;

    const float *qrow = q + (((int64_t)b * Hq + hq) * Tq + tq) * D;
    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * D;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * D;
    float *orow = out + (((int64_t)b * Hq + hq) * Tq + tq) * D;

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
            if (tid < step) {
                red[tid] += red[tid + step];
            }
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

__global__ void indexed_attention_d64_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        const int64_t *indices,
        float *out,
        int B,
        int Hq,
        int Hkv,
        int Tq,
        int Nkv,
        int Tk,
        float scale) {
    __shared__ float warp_partials[2];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int tq = row % Tq;
    int hq = (row / Tq) % Hq;
    int b = row / (Tq * Hq);
    int group = Hq / Hkv;
    int hk = hq / group;

    const float *qrow = q + (((int64_t)b * Hq + hq) * Tq + tq) * 64;
    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * 64;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * 64;
    float *orow = out + (((int64_t)b * Hq + hq) * Tq + tq) * 64;

    float qv = qrow[tid];
    float acc = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n = 0; n < Nkv; ++n) {
        int tk = (int)indices[n];
        const float *krow = kbase + (int64_t)tk * 64;
        float partial = wm_warp_sum(qv * krow[tid]);
        if (lane == 0) warp_partials[warp] = partial;
        __syncthreads();

        float score = (warp_partials[0] + warp_partials[1]) * scale;
        float new_m = fmaxf(m, score);
        float alpha = expf(m - new_m);
        float beta = expf(score - new_m);
        acc = acc * alpha + beta * vbase[(int64_t)tk * 64 + tid];
        l = l * alpha + beta;
        m = new_m;
        __syncthreads();
    }

    orow[tid] = Nkv > 0 ? acc / l : 0.0f;
}

__global__ void indexed_attention_d64_warp_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        const int64_t *indices,
        float *out,
        int B,
        int Hq,
        int Hkv,
        int Tq,
        int Nkv,
        int Tk,
        float scale) {
    int warp_row = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_rows = B * Hq * Tq;
    if (warp_row >= total_rows) return;

    int tq = warp_row % Tq;
    int hq = (warp_row / Tq) % Hq;
    int b = warp_row / (Tq * Hq);
    int group = Hq / Hkv;
    int hk = hq / group;

    const float *qrow = q + (((int64_t)b * Hq + hq) * Tq + tq) * 64;
    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * 64;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * 64;
    float *orow = out + (((int64_t)b * Hq + hq) * Tq + tq) * 64;

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

__global__ void indexed_attention_d64_q4_shared_f32_kernel(
        const float *__restrict__ q,
        const float *__restrict__ k,
        const float *__restrict__ v,
        const int64_t *__restrict__ indices,
        float *__restrict__ out,
        int B,
        int Hq,
        int Hkv,
        int Tq,
        int Nkv,
        int Tk,
        float scale) {
    extern __shared__ float smem[];
    float *sh_k = smem;
    float *sh_v = sh_k + WM_ATTN_D64_K_BLOCK * 64;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int q_blocks = (Tq + WM_ATTN_D64_Q_BLOCK - 1) / WM_ATTN_D64_Q_BLOCK;
    int bh = blockIdx.x / q_blocks;
    int tq_base = (blockIdx.x - bh * q_blocks) * WM_ATTN_D64_Q_BLOCK;
    int tq = tq_base + warp;
    int hq = bh % Hq;
    int b = bh / Hq;
    if (b >= B) return;

    int group = Hq / Hkv;
    int hk = hq / group;
    if (Nkv < 0) Nkv = 0;
    if (Nkv > Tk) Nkv = Tk;

    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * 64;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * 64;
    bool valid_q = tq < Tq;
    const float *qrow = q + (((int64_t)b * Hq + hq) * Tq + tq) * 64;
    float *orow = out + (((int64_t)b * Hq + hq) * Tq + tq) * 64;

    float q0 = valid_q ? qrow[lane] : 0.0f;
    float q1 = valid_q ? qrow[lane + 32] : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n0 = 0; n0 < Nkv; n0 += WM_ATTN_D64_K_BLOCK) {
        int active = Nkv - n0;
        if (active > WM_ATTN_D64_K_BLOCK) active = WM_ATTN_D64_K_BLOCK;
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

__global__ void indexed_attention_d64_hkv_group_flash_f32_kernel(
        const float *__restrict__ q,
        const float *__restrict__ k,
        const float *__restrict__ v,
        const int64_t *__restrict__ indices,
        float *__restrict__ out,
        int B,
        int Hq,
        int Hkv,
        int Tq,
        int Nkv,
        int Tk,
        float scale) {
    extern __shared__ float smem[];
    float *sh_k = smem;
    float *sh_v = sh_k + WM_ATTN_D64_K_BLOCK * 64;

    int group = Hq / Hkv;
    int q_per_h = WM_ATTN_D64_FLASH_WARPS / group;
    if (q_per_h < 1) q_per_h = 1;
    int q_blocks = (Tq + q_per_h - 1) / q_per_h;
    int q_block = blockIdx.x % q_blocks;
    int bhk = blockIdx.x / q_blocks;
    int hk = bhk % Hkv;
    int b = bhk / Hkv;
    if (b >= B) return;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int tid = threadIdx.x;
    int local_h = warp / q_per_h;
    int tq = q_block * q_per_h + (warp - local_h * q_per_h);
    bool valid_q = local_h < group && tq < Tq;
    int hq = hk * group + local_h;

    const float *kbase = k + ((int64_t)b * Hkv + hk) * Tk * 64;
    const float *vbase = v + ((int64_t)b * Hkv + hk) * Tk * 64;
    const float *qrow = valid_q ? q + (((int64_t)b * Hq + hq) * Tq + tq) * 64 : q;
    float *orow = valid_q ? out + (((int64_t)b * Hq + hq) * Tq + tq) * 64 : out;

    float q0 = valid_q ? qrow[lane] : 0.0f;
    float q1 = valid_q ? qrow[lane + 32] : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int n0 = 0; n0 < Nkv; n0 += WM_ATTN_D64_K_BLOCK) {
        int active = Nkv - n0;
        if (active > WM_ATTN_D64_K_BLOCK) active = WM_ATTN_D64_K_BLOCK;
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

__global__ void gather_indexed_kv_d64_f32_kernel(
        const float *__restrict__ k,
        const float *__restrict__ v,
        const int64_t *__restrict__ indices,
        float *__restrict__ k_compact,
        float *__restrict__ v_compact,
        int B,
        int Hq,
        int Hkv,
        int Nkv,
        int Tk) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * Hq * Nkv * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t q = i >> 6;
    int n = (int)(q % Nkv);
    int hq = (int)((q / Nkv) % Hq);
    int b = (int)(q / ((int64_t)Nkv * Hq));
    int group = Hq / Hkv;
    int hk = hq / group;
    int tk = (int)indices[n];
    if (tk < 0) tk = 0;
    if (tk >= Tk) tk = Tk - 1;
    int64_t src = (((int64_t)b * Hkv + hk) * Tk + tk) * 64 + d;
    k_compact[i] = k[src];
    v_compact[i] = v[src];
}

__global__ void gather_indexed_kv_hkv_d64_f32_kernel(
        const float *__restrict__ k,
        const float *__restrict__ v,
        const int64_t *__restrict__ indices,
        float *__restrict__ k_compact,
        float *__restrict__ v_compact,
        int B,
        int Hkv,
        int Nkv,
        int Tk) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * Hkv * Nkv * 64;
    if (i >= total) return;
    int d = (int)(i & 63);
    int64_t q = i >> 6;
    int n = (int)(q % Nkv);
    int hk = (int)((q / Nkv) % Hkv);
    int b = (int)(q / ((int64_t)Nkv * Hkv));
    int tk = (int)indices[n];
    if (tk < 0) tk = 0;
    if (tk >= Tk) tk = Tk - 1;
    int64_t src = (((int64_t)b * Hkv + hk) * Tk + tk) * 64 + d;
    k_compact[i] = k[src];
    v_compact[i] = v[src];
}

__global__ void softmax_rows_inplace_f32_kernel(float *x, int rows, int cols) {
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

__global__ void kv_cache_upsert_copy_f32_kernel(
        float *cache_k,
        float *cache_v,
        const float *k,
        const float *v,
        bool *written,
        int B,
        int H,
        int T,
        int D,
        int L,
        int base,
        bool write_step,
        bool frozen) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * H * T * D;
    if (i >= total) return;

    int d = i % D;
    int64_t q = i / D;
    int t = q % T;
    q /= T;
    int h = q % H;
    int b = q / H;

    int tail_idx = L + t;
    int ring_idx = base + t;
    int dst_idx = (!frozen && write_step) ? ring_idx : tail_idx;

    int64_t src = (((int64_t)b * H + h) * T + t) * D + d;
    int64_t tail = (((int64_t)b * H + h) * (L + T) + tail_idx) * D + d;
    int64_t dst = (((int64_t)b * H + h) * (L + T) + dst_idx) * D + d;

    float kv = k[src];
    float vv = v[src];
    cache_k[tail] = kv;
    cache_v[tail] = vv;
    if (!frozen) {
        cache_k[dst] = kv;
        cache_v[dst] = vv;
    }

    if (b == 0 && h == 0 && d == 0) {
        written[tail_idx] = true;
        if (!frozen) {
            written[dst_idx] = true;
        }
    }
}

__global__ void kv_cache_mask_kernel(
        const bool *written,
        bool *mask_written,
        int capacity,
        int T,
        int base,
        bool write_step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    bool value = written[i];
    if (write_step && i >= base && i < base + T) {
        value = false;
    }
    mask_written[i] = value;
}

__global__ void cache_frame_indices_kernel(
        const bool *written,
        int64_t *indices,
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
    }
}

__global__ void patchify_f32_kernel(
        const float *x,
        const float *weight,
        float *tokens,
        int B,
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
    int b = row / (D * Hp * Wp);
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
        float xv = x[((int64_t)b * C * H + c * H + iy) * W + ix];
        float wv = weight[(((int64_t)d * C + c) * ph + dy) * pw + dx];
        sum += xv * wv;
    }

    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        tokens[((int64_t)b * Hp * Wp + token) * D + d] = red[0];
    }
}

__global__ void patchify_im2row_f32_kernel(
        const float *x,
        float *rows,
        int B,
        int C,
        int H,
        int W,
        int ph,
        int pw,
        int Hp,
        int Wp) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int patch_elems = C * ph * pw;
    int64_t total = (int64_t)B * Hp * Wp * patch_elems;
    if (i >= total) return;
    int p = (int)(i % patch_elems);
    int64_t qtoken = i / patch_elems;
    int token = (int)(qtoken % (Hp * Wp));
    int b = (int)(qtoken / (Hp * Wp));
    int ox = token % Wp;
    int oy = token / Wp;
    int q = p;
    int dx = q % pw;
    q /= pw;
    int dy = q % ph;
    int c = q / ph;
    int iy = oy * ph + dy;
    int ix = ox * pw + dx;
    rows[i] = x[((int64_t)b * C + c) * H * W + iy * W + ix];
}

__global__ void unpatchify_f32_kernel(
        const float *tokens,
        const float *weight,
        const float *bias,
        float *x,
        int B,
        int T,
        int D,
        int C,
        int H,
        int W,
        int ph,
        int pw,
        int Hp,
        int Wp,
        int out_dim) {
    __shared__ float red[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int o = row % out_dim;
    int token = (row / out_dim) % T;
    int b = row / (out_dim * T);

    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        sum += tokens[((int64_t)b * T + token) * D + d] * weight[(int64_t)o * D + d];
    }

    red[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red[tid] += red[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int p = o;
        int dx = p % pw;
        p /= pw;
        int dy = p % ph;
        p /= ph;
        int c = p;
        int oy = token / Wp;
        int ox = token - oy * Wp;
        int iy = oy * ph + dy;
        int ix = ox * pw + dx;
        x[((int64_t)b * C * H + c * H + iy) * W + ix] = red[0] + bias[o];
    }
}

__global__ void taehv_conv2d_nchw_f32_kernel(
        const float *in,
        const float *weight,
        const float *bias,
        float *out,
        int N,
        int C_in,
        int C_out,
        int H,
        int W,
        int K) {
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

    float sum = bias ? bias[co] : 0.0f;
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

__global__ void taehv_add_bias_nchw_f32_kernel(
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

__global__ void taehv_add_bias_nhwc_f32_kernel(
        float *out,
        const float *bias,
        int64_t total,
        int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] += bias[i % C];
}

__global__ void taehv_im2col3x3_nchw_tile_f32_kernel(
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

__global__ void taehv_im2col3x3_nchw_batch_tile_f32_kernel(
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

__global__ void taehv_scatter_conv_tile_to_nchw_f32_kernel(
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

__global__ void taehv_concat_past_nchw_f32_kernel(const float *x, float *out, int N, int C, int H, int W) {
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
        out[i] = x[(((int64_t)n - 1) * C * H + c * H + h) * W + w];
    }
}

__global__ void taehv_upsample2_nchw_f32_kernel(const float *in, float *out, int N, int C, int H, int W) {
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

__global__ void taehv_tgrow_reshape_f32_kernel(const float *in, float *out, int N, int C, int H, int W, int stride) {
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
    out[i] = in[((int64_t)n * (C * stride) * H + (s * C + c) * H + y) * W + x];
}

torch::Tensor silu_cuda(torch::Tensor x) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_F32(x);

    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    silu_f32_kernel<<<div_up_i64(n, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    check_last_cuda_error("silu_f32");
    return y;
}

torch::Tensor row_major_linear_fp16_cuda(torch::Tensor x, torch::Tensor w) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(w);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(w);
    WM_CHECK_F32(x);
    WM_CHECK_F32(w);
    TORCH_CHECK(x.dim() == 2, "x must be [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be [N,K]");
    TORCH_CHECK(x.size(1) == w.size(1), "K mismatch");

    int m = (int)x.size(0);
    int k = (int)x.size(1);
    int n = (int)w.size(0);
    auto x_h = x.to(torch::kFloat16);
    auto w_h = w.to(torch::kFloat16);
    auto y = torch::empty({m, n}, x.options());

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

    const cutlass::half_t *x_ptr = reinterpret_cast<const cutlass::half_t *>(x_h.data_ptr<at::Half>());
    const cutlass::half_t *w_ptr = reinterpret_cast<const cutlass::half_t *>(w_h.data_ptr<at::Half>());
    typename Gemm::Arguments args(
        {m, n, k},
        {x_ptr, k},
        {w_ptr, k},
        {y.data_ptr<float>(), n},
        {y.data_ptr<float>(), n},
        {1.0f, 0.0f});
    Gemm gemm;
    check_cutlass_status(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()), "row_major_linear_fp16_cutlass");
    check_last_cuda_error("row_major_linear_fp16_cutlass");
    return y;
}

torch::Tensor row_major_linear_fp16_tensorop_cuda(torch::Tensor x, torch::Tensor w) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(w);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(w);
    WM_CHECK_F32(x);
    WM_CHECK_F32(w);
    TORCH_CHECK(x.dim() == 2, "x must be [M,K]");
    TORCH_CHECK(w.dim() == 2, "w must be [N,K]");
    TORCH_CHECK(x.size(1) == w.size(1), "K mismatch");

    int m = (int)x.size(0);
    int k = (int)x.size(1);
    int n = (int)w.size(0);
    auto x_h = x.to(torch::kFloat16);
    auto w_h = w.to(torch::kFloat16);
    auto y = torch::empty({m, n}, x.options());

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

    const cutlass::half_t *x_ptr = reinterpret_cast<const cutlass::half_t *>(x_h.data_ptr<at::Half>());
    const cutlass::half_t *w_ptr = reinterpret_cast<const cutlass::half_t *>(w_h.data_ptr<at::Half>());
    typename Gemm::Arguments args(
        {m, n, k},
        {x_ptr, k},
        {w_ptr, k},
        {y.data_ptr<float>(), n},
        {y.data_ptr<float>(), n},
        {1.0f, 0.0f});
    Gemm gemm;
    check_cutlass_status(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()), "row_major_linear_fp16_tensorop_cutlass");
    check_last_cuda_error("row_major_linear_fp16_tensorop_cutlass");
    return y;
}

torch::Tensor rms_norm_cuda(torch::Tensor x, double eps) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_F32(x);
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");

    int64_t D64 = x.size(-1);
    TORCH_CHECK(D64 > 0 && D64 <= 4096, "unsupported RMSNorm dim: ", D64);
    int D = (int)D64;
    int rows = (int)(x.numel() / D64);
    auto y = torch::empty_like(x);

    rms_norm_f32_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), rows, D, (float)eps);
    check_last_cuda_error("rms_norm_f32");
    return y;
}

torch::Tensor ada_rms_norm_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, double eps) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(scale);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(scale);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(scale);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 3, "x must be [B,T,D]");
    TORCH_CHECK(scale.dim() == 3 && bias.dim() == 3, "scale and bias must be [B,N,D]");
    TORCH_CHECK(scale.sizes() == bias.sizes(), "scale and bias shapes must match");
    TORCH_CHECK(x.size(0) == scale.size(0), "B mismatch");
    TORCH_CHECK(x.size(2) == scale.size(2), "D mismatch");
    TORCH_CHECK(x.size(1) % scale.size(1) == 0, "T must be divisible by N");

    int B = (int)x.size(0);
    int T = (int)x.size(1);
    int D = (int)x.size(2);
    int N = (int)scale.size(1);
    auto y = torch::empty_like(x);

    ada_rms_norm_f32_kernel<<<B * T, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        B,
        T,
        N,
        D,
        (float)eps);
    check_last_cuda_error("ada_rms_norm_f32");
    return y;
}

torch::Tensor ortho_rope_cuda(
        torch::Tensor x,
        torch::Tensor x_pos,
        torch::Tensor y_pos,
        torch::Tensor t_pos,
        torch::Tensor xy,
        torch::Tensor inv_t,
        int64_t width,
        int64_t height) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(x_pos);
    WM_CHECK_CUDA(y_pos);
    WM_CHECK_CUDA(t_pos);
    WM_CHECK_CUDA(xy);
    WM_CHECK_CUDA(inv_t);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(x_pos);
    WM_CHECK_CONTIGUOUS(y_pos);
    WM_CHECK_CONTIGUOUS(t_pos);
    WM_CHECK_CONTIGUOUS(xy);
    WM_CHECK_CONTIGUOUS(inv_t);
    WM_CHECK_F32(x);
    WM_CHECK_F32(xy);
    WM_CHECK_F32(inv_t);
    TORCH_CHECK(x_pos.scalar_type() == at::ScalarType::Long, "x_pos must be int64");
    TORCH_CHECK(y_pos.scalar_type() == at::ScalarType::Long, "y_pos must be int64");
    TORCH_CHECK(t_pos.scalar_type() == at::ScalarType::Long, "t_pos must be int64");
    TORCH_CHECK(x.dim() == 4, "x must be [B,H,T,D]");
    TORCH_CHECK(x.size(3) % 8 == 0, "D must be divisible by 8");
    TORCH_CHECK(x_pos.numel() == x.size(2), "x_pos length must equal T");
    TORCH_CHECK(y_pos.numel() == x.size(2), "y_pos length must equal T");
    TORCH_CHECK(t_pos.numel() == x.size(2), "t_pos length must equal T");
    TORCH_CHECK(xy.numel() == x.size(3) / 8, "xy length must equal D/8");
    TORCH_CHECK(inv_t.numel() == x.size(3) / 4, "inv_t length must equal D/4");

    int B = (int)x.size(0);
    int H = (int)x.size(1);
    int T = (int)x.size(2);
    int D = (int)x.size(3);
    auto y = torch::empty_like(x);
    int64_t work = (int64_t)B * H * T * (D / 2);

    ortho_rope_f32_kernel<<<div_up_i64(work, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x_pos.data_ptr<int64_t>(),
        y_pos.data_ptr<int64_t>(),
        t_pos.data_ptr<int64_t>(),
        xy.data_ptr<float>(),
        inv_t.data_ptr<float>(),
        B,
        H,
        T,
        D,
        (int)width,
        (int)height);
    check_last_cuda_error("ortho_rope_f32");
    return y;
}

std::vector<torch::Tensor> qkv_rms_rope_cuda(
        torch::Tensor qkv,
        torch::Tensor x_pos,
        torch::Tensor y_pos,
        torch::Tensor t_pos,
        torch::Tensor xy,
        torch::Tensor inv_t,
        int64_t n_heads,
        int64_t n_kv_heads,
        int64_t width,
        int64_t height,
        double eps) {
    WM_CHECK_CUDA(qkv);
    WM_CHECK_CUDA(x_pos);
    WM_CHECK_CUDA(y_pos);
    WM_CHECK_CUDA(t_pos);
    WM_CHECK_CUDA(xy);
    WM_CHECK_CUDA(inv_t);
    WM_CHECK_CONTIGUOUS(qkv);
    WM_CHECK_CONTIGUOUS(x_pos);
    WM_CHECK_CONTIGUOUS(y_pos);
    WM_CHECK_CONTIGUOUS(t_pos);
    WM_CHECK_CONTIGUOUS(xy);
    WM_CHECK_CONTIGUOUS(inv_t);
    WM_CHECK_F32(qkv);
    WM_CHECK_F32(xy);
    WM_CHECK_F32(inv_t);
    TORCH_CHECK(x_pos.scalar_type() == at::ScalarType::Long, "x_pos must be int64");
    TORCH_CHECK(y_pos.scalar_type() == at::ScalarType::Long, "y_pos must be int64");
    TORCH_CHECK(t_pos.scalar_type() == at::ScalarType::Long, "t_pos must be int64");
    TORCH_CHECK(qkv.dim() == 3, "qkv must be [B,T,(n_heads+2*n_kv_heads)*D]");
    TORCH_CHECK(n_heads > 0 && n_kv_heads > 0, "head counts must be positive");
    TORCH_CHECK(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");

    int B = (int)qkv.size(0);
    int T = (int)qkv.size(1);
    int total_heads = (int)(n_heads + 2 * n_kv_heads);
    TORCH_CHECK(qkv.size(2) % total_heads == 0, "last dim must divide total qkv heads");
    int D = (int)(qkv.size(2) / total_heads);
    TORCH_CHECK(D % 8 == 0, "head dim must be divisible by 8");
    TORCH_CHECK(x_pos.numel() == T && y_pos.numel() == T && t_pos.numel() == T, "position lengths must equal T");
    TORCH_CHECK(xy.numel() == D / 8, "xy length must equal D/8");
    TORCH_CHECK(inv_t.numel() == D / 4, "inv_t length must equal D/4");

    auto opts = qkv.options();
    auto q = torch::empty({B, n_heads, T, D}, opts);
    auto k = torch::empty({B, n_kv_heads, T, D}, opts);
    auto v = torch::empty({B, n_kv_heads, T, D}, opts);

    dim3 grid(B * T, total_heads);
    size_t smem = ((size_t)D + 256) * sizeof(float);
    qkv_rms_rope_f32_kernel<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        qkv.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        x_pos.data_ptr<int64_t>(),
        y_pos.data_ptr<int64_t>(),
        t_pos.data_ptr<int64_t>(),
        xy.data_ptr<float>(),
        inv_t.data_ptr<float>(),
        B,
        T,
        (int)n_heads,
        (int)n_kv_heads,
        D,
        (int)width,
        (int)height,
        (float)eps);
    check_last_cuda_error("qkv_rms_rope_f32");

    return {q, k, v};
}

torch::Tensor masked_attention_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor written,
        double scale) {
    WM_CHECK_CUDA(q);
    WM_CHECK_CUDA(k);
    WM_CHECK_CUDA(v);
    WM_CHECK_CUDA(written);
    WM_CHECK_CONTIGUOUS(q);
    WM_CHECK_CONTIGUOUS(k);
    WM_CHECK_CONTIGUOUS(v);
    WM_CHECK_CONTIGUOUS(written);
    WM_CHECK_F32(q);
    WM_CHECK_F32(k);
    WM_CHECK_F32(v);
    TORCH_CHECK(written.scalar_type() == at::ScalarType::Bool, "written must be bool");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0), "B mismatch");
    TORCH_CHECK(q.size(3) == k.size(3), "D mismatch");
    TORCH_CHECK(written.numel() == k.size(2), "written length must equal Tk");

    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int Tq = (int)q.size(2);
    int D = (int)q.size(3);
    int Hkv = (int)k.size(1);
    int Tk = (int)k.size(2);
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv for GQA");
    TORCH_CHECK(D > 0 && D <= 256, "masked_attention currently supports 1 <= D <= 256");

    auto out = torch::empty_like(q);
    masked_attention_f32_kernel<<<B * Hq * Tq, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        written.data_ptr<bool>(),
        out.data_ptr<float>(),
        B,
        Hq,
        Hkv,
        Tq,
        Tk,
        D,
        (float)scale);
    check_last_cuda_error("masked_attention_f32");
    return out;
}

torch::Tensor indexed_attention_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    WM_CHECK_CUDA(q);
    WM_CHECK_CUDA(k);
    WM_CHECK_CUDA(v);
    WM_CHECK_CUDA(indices);
    WM_CHECK_CONTIGUOUS(q);
    WM_CHECK_CONTIGUOUS(k);
    WM_CHECK_CONTIGUOUS(v);
    WM_CHECK_CONTIGUOUS(indices);
    WM_CHECK_F32(q);
    WM_CHECK_F32(k);
    WM_CHECK_F32(v);
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long, "indices must be int64");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0), "B mismatch");
    TORCH_CHECK(q.size(3) == k.size(3), "D mismatch");

    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int Tq = (int)q.size(2);
    int D = (int)q.size(3);
    int Hkv = (int)k.size(1);
    int Tk = (int)k.size(2);
    int Nkv = (int)indices.numel();
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv for GQA");
    TORCH_CHECK(D > 0 && D <= 256, "indexed_attention currently supports 1 <= D <= 256");

    auto out = torch::empty_like(q);
    if (D == 64) {
        int q_blocks = div_up_i64(Tq, WM_ATTN_D64_Q_BLOCK);
        size_t smem = (size_t)2 * WM_ATTN_D64_K_BLOCK * 64 * sizeof(float);
        indexed_attention_d64_q4_shared_f32_kernel<<<B * Hq * q_blocks, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            B,
            Hq,
            Hkv,
            Tq,
            Nkv,
            Tk,
            (float)scale);
        check_last_cuda_error("indexed_attention_d64_q4_shared_f32");
    } else {
        indexed_attention_f32_kernel<<<B * Hq * Tq, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            B,
            Hq,
            Hkv,
            Tq,
            Nkv,
            Tk,
            D,
            (float)scale);
        check_last_cuda_error("indexed_attention_f32");
    }
    return out;
}

torch::Tensor indexed_attention_flash_cuda(
        torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor indices,
        double scale) {
    WM_CHECK_CUDA(q);
    WM_CHECK_CUDA(k);
    WM_CHECK_CUDA(v);
    WM_CHECK_CUDA(indices);
    WM_CHECK_CONTIGUOUS(q);
    WM_CHECK_CONTIGUOUS(k);
    WM_CHECK_CONTIGUOUS(v);
    WM_CHECK_CONTIGUOUS(indices);
    WM_CHECK_F32(q);
    WM_CHECK_F32(k);
    WM_CHECK_F32(v);
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long, "indices must be int64");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0), "B mismatch");
    TORCH_CHECK(q.size(3) == 64 && k.size(3) == 64, "indexed_attention_flash currently supports D=64");

    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int Tq = (int)q.size(2);
    int Hkv = (int)k.size(1);
    int Tk = (int)k.size(2);
    int Nkv = (int)indices.numel();
    TORCH_CHECK(Hq % Hkv == 0, "Hq must be divisible by Hkv for GQA");
    int group = Hq / Hkv;
    TORCH_CHECK(group > 0 && group <= WM_ATTN_D64_FLASH_WARPS, "unsupported GQA group for flash prototype");

    auto out = torch::empty_like(q);
    int q_per_h = WM_ATTN_D64_FLASH_WARPS / group;
    if (q_per_h < 1) q_per_h = 1;
    int q_blocks = div_up_i64(Tq, q_per_h);
    size_t smem = (size_t)2 * WM_ATTN_D64_K_BLOCK * 64 * sizeof(float);
    indexed_attention_d64_hkv_group_flash_f32_kernel<<<B * Hkv * q_blocks, 32 * WM_ATTN_D64_FLASH_WARPS, smem, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        B,
        Hq,
        Hkv,
        Tq,
        Nkv,
        Tk,
        (float)scale);
    check_last_cuda_error("indexed_attention_d64_hkv_group_flash_f32");
    return out;
}

torch::Tensor kv_cache_upsert_cuda(
        torch::Tensor cache_k,
        torch::Tensor cache_v,
        torch::Tensor written,
        torch::Tensor k,
        torch::Tensor v,
        int64_t frame_idx,
        int64_t ring_length,
        int64_t pinned_dilation,
        bool frozen) {
    WM_CHECK_CUDA(cache_k);
    WM_CHECK_CUDA(cache_v);
    WM_CHECK_CUDA(written);
    WM_CHECK_CUDA(k);
    WM_CHECK_CUDA(v);
    WM_CHECK_CONTIGUOUS(cache_k);
    WM_CHECK_CONTIGUOUS(cache_v);
    WM_CHECK_CONTIGUOUS(written);
    WM_CHECK_CONTIGUOUS(k);
    WM_CHECK_CONTIGUOUS(v);
    WM_CHECK_F32(cache_k);
    WM_CHECK_F32(cache_v);
    WM_CHECK_F32(k);
    WM_CHECK_F32(v);
    TORCH_CHECK(written.scalar_type() == at::ScalarType::Bool, "written must be bool");
    TORCH_CHECK(cache_k.sizes() == cache_v.sizes(), "cache_k/cache_v shape mismatch");
    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shape mismatch");
    TORCH_CHECK(cache_k.dim() == 4 && k.dim() == 4, "cache and k/v must be [B,H,T,D]");
    TORCH_CHECK(cache_k.size(0) == k.size(0), "B mismatch");
    TORCH_CHECK(cache_k.size(1) == k.size(1), "H mismatch");
    TORCH_CHECK(cache_k.size(3) == k.size(3), "D mismatch");
    TORCH_CHECK(frame_idx >= 0, "frame_idx must be non-negative");
    TORCH_CHECK(ring_length > 0, "ring_length must be positive");
    TORCH_CHECK(pinned_dilation > 0, "pinned_dilation must be positive");

    int B = (int)k.size(0);
    int H = (int)k.size(1);
    int T = (int)k.size(2);
    int D = (int)k.size(3);
    int L = (int)ring_length;
    int capacity = (int)cache_k.size(2);
    TORCH_CHECK(capacity == L + T, "cache capacity must equal ring_length + T");
    TORCH_CHECK(written.numel() == capacity, "written length must equal cache capacity");
    TORCH_CHECK(L % T == 0, "ring_length must be divisible by T");
    TORCH_CHECK((L / T) % pinned_dilation == 0, "ring frames must be divisible by pinned_dilation");

    int64_t bucket = (frame_idx + (pinned_dilation - 1)) / pinned_dilation;
    int64_t num_buckets = (L / T) / pinned_dilation;
    int base = (int)((bucket % num_buckets) * T);
    bool write_step = (frame_idx % pinned_dilation) == 0;

    auto mask_written = torch::empty_like(written);
    kv_cache_mask_kernel<<<div_up_i64(capacity, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        written.data_ptr<bool>(),
        mask_written.data_ptr<bool>(),
        capacity,
        T,
        base,
        write_step);
    check_last_cuda_error("kv_cache_mask");

    int64_t total = (int64_t)B * H * T * D;
    kv_cache_upsert_copy_f32_kernel<<<div_up_i64(total, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        cache_k.data_ptr<float>(),
        cache_v.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        written.data_ptr<bool>(),
        B,
        H,
        T,
        D,
        L,
        base,
        write_step,
        frozen);
    check_last_cuda_error("kv_cache_upsert_copy");

    return mask_written;
}

std::vector<torch::Tensor> cache_frame_indices_cuda(
        torch::Tensor written,
        int64_t tokens_per_frame,
        int64_t base,
        bool write_step) {
    WM_CHECK_CUDA(written);
    WM_CHECK_CONTIGUOUS(written);
    TORCH_CHECK(written.scalar_type() == at::ScalarType::Bool, "written must be bool");
    TORCH_CHECK(written.dim() == 1, "written must be 1D");
    TORCH_CHECK(tokens_per_frame > 0, "tokens_per_frame must be positive");

    int capacity = (int)written.numel();
    int T = (int)tokens_per_frame;
    TORCH_CHECK(capacity > 0, "written must not be empty");
    TORCH_CHECK(capacity % T == 0, "written length must be divisible by tokens_per_frame");
    TORCH_CHECK(base >= 0 && base + T <= capacity, "base frame slot out of range");
    TORCH_CHECK((base % T) == 0, "base must point to a frame slot boundary");

    auto indices = torch::empty({capacity}, written.options().dtype(torch::kLong));
    auto count = torch::empty({1}, written.options().dtype(torch::kInt32));
    cache_frame_indices_kernel<<<capacity / T, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        written.data_ptr<bool>(),
        indices.data_ptr<int64_t>(),
        count.data_ptr<int>(),
        capacity,
        T,
        (int)base,
        write_step);
    check_last_cuda_error("cache_frame_indices");
    return {indices, count};
}

torch::Tensor patchify_cuda(torch::Tensor x, torch::Tensor weight) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be [D,C,ph,pw]");
    TORCH_CHECK(x.size(1) == weight.size(1), "channel mismatch");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int D = (int)weight.size(0);
    int ph = (int)weight.size(2);
    int pw = (int)weight.size(3);
    TORCH_CHECK(H % ph == 0 && W % pw == 0, "H/W must be divisible by patch");
    int Hp = H / ph;
    int Wp = W / pw;

    auto tokens = torch::empty({B, Hp * Wp, D}, x.options());
    int64_t rows = (int64_t)B * Hp * Wp * D;
    patchify_f32_kernel<<<div_up_i64(rows, 1), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        tokens.data_ptr<float>(),
        B,
        C,
        H,
        W,
        D,
        ph,
        pw,
        Hp,
        Wp);
    check_last_cuda_error("patchify_f32");
    return tokens;
}

torch::Tensor patchify_cutlass_cuda(torch::Tensor x, torch::Tensor weight) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    TORCH_CHECK(x.dim() == 4, "x must be [B,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be [D,C,ph,pw]");
    TORCH_CHECK(x.size(1) == weight.size(1), "channel mismatch");

    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int D = (int)weight.size(0);
    int ph = (int)weight.size(2);
    int pw = (int)weight.size(3);
    TORCH_CHECK(H % ph == 0 && W % pw == 0, "H/W must be divisible by patch");
    int Hp = H / ph;
    int Wp = W / pw;
    int T = Hp * Wp;
    int patch_elems = C * ph * pw;

    auto rows = torch::empty({B * T, patch_elems}, x.options());
    auto tokens = torch::empty({B, T, D}, x.options());
    int64_t row_elems = (int64_t)B * T * patch_elems;
    patchify_im2row_f32_kernel<<<div_up_i64(row_elems, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        rows.data_ptr<float>(),
        B,
        C,
        H,
        W,
        ph,
        pw,
        Hp,
        Wp);
    check_last_cuda_error("patchify_im2row_f32");

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
        cutlass::gemm::GemmShape<128, 128, 8>,
        cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        1,
        1>;

    typename Gemm::Arguments args(
        {B * T, D, patch_elems},
        {rows.data_ptr<float>(), patch_elems},
        {weight.data_ptr<float>(), patch_elems},
        {tokens.data_ptr<float>(), D},
        {tokens.data_ptr<float>(), D},
        {1.0f, 0.0f});
    Gemm gemm;
    check_cutlass_status(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()), "patchify_cutlass");
    check_last_cuda_error("patchify_cutlass");
    return tokens;
}

torch::Tensor unpatchify_cuda(
        torch::Tensor tokens,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t channels,
        int64_t height,
        int64_t width,
        int64_t patch_h,
        int64_t patch_w) {
    WM_CHECK_CUDA(tokens);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(tokens);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(tokens);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(tokens.dim() == 3, "tokens must be [B,T,D]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [C*ph*pw,D]");
    TORCH_CHECK(bias.dim() == 1, "bias must be [C*ph*pw]");
    TORCH_CHECK(channels > 0 && height > 0 && width > 0 && patch_h > 0 && patch_w > 0, "invalid shape");
    TORCH_CHECK(height % patch_h == 0 && width % patch_w == 0, "height/width must be divisible by patch");

    int B = (int)tokens.size(0);
    int T = (int)tokens.size(1);
    int D = (int)tokens.size(2);
    int C = (int)channels;
    int H = (int)height;
    int W = (int)width;
    int ph = (int)patch_h;
    int pw = (int)patch_w;
    int Hp = H / ph;
    int Wp = W / pw;
    int out_dim = C * ph * pw;
    TORCH_CHECK(T == Hp * Wp, "T must equal patched spatial token count");
    TORCH_CHECK(weight.size(0) == out_dim && weight.size(1) == D, "weight shape mismatch");
    TORCH_CHECK(bias.numel() == out_dim, "bias shape mismatch");

    auto x = torch::empty({B, C, H, W}, tokens.options());
    int64_t rows = (int64_t)B * T * out_dim;
    unpatchify_f32_kernel<<<div_up_i64(rows, 1), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        tokens.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        x.data_ptr<float>(),
        B,
        T,
        D,
        C,
        H,
        W,
        ph,
        pw,
        Hp,
        Wp,
        out_dim);
    check_last_cuda_error("unpatchify_f32");
    return x;
}

torch::Tensor taehv_conv2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cout,Cin,K,K]");
    TORCH_CHECK(weight.size(1) == x.size(1), "input channel mismatch");
    TORCH_CHECK(weight.size(2) == weight.size(3), "kernel must be square");
    TORCH_CHECK(weight.size(2) == 1 || weight.size(2) == 3, "only 1x1 and 3x3 are supported");
    TORCH_CHECK(bias.numel() == weight.size(0), "bias length must equal Cout");

    int N = (int)x.size(0);
    int C_in = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int C_out = (int)weight.size(0);
    int K = (int)weight.size(2);
    auto out = torch::empty({N, C_out, H, W}, x.options());
    int64_t total = (int64_t)N * C_out * H * W;
    taehv_conv2d_nchw_f32_kernel<<<div_up_i64(total, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), N, C_in, C_out, H, W, K);
    check_last_cuda_error("taehv_conv2d_nchw_f32");
    return out;
}

torch::Tensor taehv_conv1x1_cutlass_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cout,Cin,1,1]");
    TORCH_CHECK(weight.size(1) == x.size(1), "input channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "weight must be 1x1");
    TORCH_CHECK(bias.numel() == weight.size(0), "bias length must equal Cout");

    int N = (int)x.size(0);
    int C_in = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int C_out = (int)weight.size(0);
    int spatial = H * W;
    auto out = torch::empty({N, C_out, H, W}, x.options());

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
    const float *x_ptr = x.data_ptr<float>();
    const float *w_ptr = weight.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();
    for (int n = 0; n < N; ++n) {
        const float *x_frame = x_ptr + (int64_t)n * C_in * spatial;
        float *out_frame = out_ptr + (int64_t)n * C_out * spatial;
        typename Gemm::Arguments args(
            {C_out, spatial, C_in},
            {w_ptr, C_in},
            {x_frame, spatial},
            {out_frame, spatial},
            {out_frame, spatial},
            {1.0f, 0.0f});
        check_cutlass_status(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()), "taehv_conv1x1_cutlass");
    }
    check_last_cuda_error("taehv_conv1x1_cutlass");

    taehv_add_bias_nchw_f32_kernel<<<div_up_i64((int64_t)N * C_out * H * W, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<float>(), bias.data_ptr<float>(), N, C_out, H, W);
    check_last_cuda_error("taehv_conv1x1_cutlass_bias");
    return out;
}

torch::Tensor taehv_conv3x3_cutlass_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cout,Cin,3,3]");
    TORCH_CHECK(weight.size(1) == x.size(1), "input channel mismatch");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "weight must be 3x3");
    TORCH_CHECK(bias.numel() == weight.size(0), "bias length must equal Cout");

    int N = (int)x.size(0);
    int C_in = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int C_out = (int)weight.size(0);
    int spatial = H * W;
    int k_elems = C_in * 9;
    auto cols = torch::empty({k_elems, spatial}, x.options());
    auto out = torch::empty({N, C_out, H, W}, x.options());

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
    const float *x_ptr = x.data_ptr<float>();
    const float *w_ptr = weight.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();
    float *cols_ptr = cols.data_ptr<float>();
    for (int n = 0; n < N; ++n) {
        taehv_im2col3x3_nchw_tile_f32_kernel<<<div_up_i64((int64_t)k_elems * spatial, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_ptr, cols_ptr, C_in, H, W, n, 0, spatial);
        check_last_cuda_error("taehv_im2col3x3_cutlass");

        float *out_frame = out_ptr + (int64_t)n * C_out * spatial;
        typename Gemm::Arguments args(
            {C_out, spatial, k_elems},
            {w_ptr, k_elems},
            {cols_ptr, spatial},
            {out_frame, spatial},
            {out_frame, spatial},
            {1.0f, 0.0f});
        check_cutlass_status(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()), "taehv_conv3x3_cutlass");
    }
    check_last_cuda_error("taehv_conv3x3_cutlass");

    taehv_add_bias_nchw_f32_kernel<<<div_up_i64((int64_t)N * C_out * H * W, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<float>(), bias.data_ptr<float>(), N, C_out, H, W);
    check_last_cuda_error("taehv_conv3x3_cutlass_bias");
    return out;
}

torch::Tensor taehv_conv3x3_cutlass_batched_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t tile_cols_arg) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be [Cout,Cin,3,3]");
    TORCH_CHECK(weight.size(1) == x.size(1), "input channel mismatch");
    TORCH_CHECK(weight.size(2) == 3 && weight.size(3) == 3, "weight must be 3x3");
    TORCH_CHECK(bias.numel() == weight.size(0), "bias length must equal Cout");
    TORCH_CHECK(tile_cols_arg > 0 && tile_cols_arg <= INT_MAX, "invalid tile_cols");

    int N = (int)x.size(0);
    int C_in = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int C_out = (int)weight.size(0);
    int spatial = H * W;
    int total_cols = N * spatial;
    int tile_cols_max = (int)tile_cols_arg;
    int k_elems = C_in * 9;
    auto cols = torch::empty({k_elems, tile_cols_max}, x.options());
    auto tile_out = torch::empty({C_out, tile_cols_max}, x.options());
    auto out = torch::empty({N, C_out, H, W}, x.options());

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
    const float *x_ptr = x.data_ptr<float>();
    const float *w_ptr = weight.data_ptr<float>();
    float *cols_ptr = cols.data_ptr<float>();
    float *tile_ptr = tile_out.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();
    for (int tile_start = 0; tile_start < total_cols; tile_start += tile_cols_max) {
        int tile_cols = total_cols - tile_start;
        if (tile_cols > tile_cols_max) tile_cols = tile_cols_max;
        taehv_im2col3x3_nchw_batch_tile_f32_kernel<<<div_up_i64((int64_t)k_elems * tile_cols, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            x_ptr, cols_ptr, N, C_in, H, W, tile_start, tile_cols);
        check_last_cuda_error("taehv_im2col3x3_batched_cutlass");

        typename Gemm::Arguments args(
            {C_out, tile_cols, k_elems},
            {w_ptr, k_elems},
            {cols_ptr, tile_cols},
            {tile_ptr, tile_cols},
            {tile_ptr, tile_cols},
            {1.0f, 0.0f});
        check_cutlass_status(gemm(args, nullptr, at::cuda::getCurrentCUDAStream()), "taehv_conv3x3_batched_cutlass");
        check_last_cuda_error("taehv_conv3x3_batched_cutlass");

        taehv_scatter_conv_tile_to_nchw_f32_kernel<<<div_up_i64((int64_t)C_out * tile_cols, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            tile_ptr, out_ptr, N, C_out, H, W, tile_start, tile_cols);
        check_last_cuda_error("taehv_conv3x3_batched_scatter");
    }

    taehv_add_bias_nchw_f32_kernel<<<div_up_i64((int64_t)N * C_out * H * W, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<float>(), bias.data_ptr<float>(), N, C_out, H, W);
    check_last_cuda_error("taehv_conv3x3_batched_cutlass_bias");
    return out;
}

torch::Tensor taehv_conv3x3_cutlass_implicit_nhwc_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CUDA(weight);
    WM_CHECK_CUDA(bias);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_CONTIGUOUS(weight);
    WM_CHECK_CONTIGUOUS(bias);
    WM_CHECK_F32(x);
    WM_CHECK_F32(weight);
    WM_CHECK_F32(bias);
    TORCH_CHECK(x.dim() == 4, "x must be NHWC [N,H,W,C]");
    TORCH_CHECK(weight.dim() == 4, "weight must be KRSC [Cout,3,3,Cin]");
    TORCH_CHECK(weight.size(1) == 3 && weight.size(2) == 3, "weight must be 3x3 KRSC");
    TORCH_CHECK(weight.size(3) == x.size(3), "input channel mismatch");
    TORCH_CHECK(bias.numel() == weight.size(0), "bias length must equal Cout");

    int N = (int)x.size(0);
    int H = (int)x.size(1);
    int W = (int)x.size(2);
    int C_in = (int)x.size(3);
    int C_out = (int)weight.size(0);
    auto out = torch::empty({N, H, W, C_out}, x.options());

    using Layout = cutlass::layout::TensorNHWC;
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        float,
        Layout,
        float,
        Layout,
        float,
        Layout,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 8>,
        cutlass::gemm::GemmShape<32, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kAnalytic,
        cutlass::conv::StrideSupport::kUnity,
        1,
        1>::Kernel;
    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    cutlass::Tensor4DCoord input_size(N, H, W, C_in);
    cutlass::Tensor4DCoord filter_size(C_out, 3, 3, C_in);
    cutlass::Tensor4DCoord padding(1, 1, 1, 1);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(N, H, W, C_out);
    cutlass::conv::Conv2dProblemSize problem(
        input_size,
        filter_size,
        padding,
        stride,
        dilation,
        output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        1);

    typename ImplicitGemm::Arguments args(
        problem,
        cutlass::TensorRef<float, Layout>(
            x.data_ptr<float>(),
            Layout::packed(input_size)),
        cutlass::TensorRef<float, Layout>(
            weight.data_ptr<float>(),
            Layout::packed(filter_size)),
        cutlass::TensorRef<float, Layout>(
            out.data_ptr<float>(),
            Layout::packed(output_size)),
        cutlass::TensorRef<float, Layout>(
            out.data_ptr<float>(),
            Layout::packed(output_size)),
        {1.0f, 0.0f});

    ImplicitGemm implicit_gemm;
    check_cutlass_status(implicit_gemm.can_implement(args), "taehv_conv3x3_cutlass_implicit_nhwc.can_implement");
    size_t workspace_size = implicit_gemm.get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size}, x.options().dtype(torch::kUInt8));
    void *workspace_ptr = workspace_size ? workspace.data_ptr<uint8_t>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    check_cutlass_status(implicit_gemm(args, workspace_ptr, stream), "taehv_conv3x3_cutlass_implicit_nhwc");
    check_last_cuda_error("taehv_conv3x3_cutlass_implicit_nhwc");

    taehv_add_bias_nhwc_f32_kernel<<<div_up_i64((int64_t)N * H * W * C_out, 256), 256, 0, stream>>>(
        out.data_ptr<float>(),
        bias.data_ptr<float>(),
        (int64_t)N * H * W * C_out,
        C_out);
    check_last_cuda_error("taehv_conv3x3_cutlass_implicit_nhwc_bias");
    return out;
}

torch::Tensor taehv_concat_past_cuda(torch::Tensor x) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_F32(x);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    auto out = torch::empty({N, C * 2, H, W}, x.options());
    int64_t total = (int64_t)N * C * 2 * H * W;
    taehv_concat_past_nchw_f32_kernel<<<div_up_i64(total, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W);
    check_last_cuda_error("taehv_concat_past_nchw_f32");
    return out;
}

torch::Tensor taehv_upsample2_cuda(torch::Tensor x) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_F32(x);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C,H,W]");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    auto out = torch::empty({N, C, H * 2, W * 2}, x.options());
    int64_t total = (int64_t)N * C * H * 2 * W * 2;
    taehv_upsample2_nchw_f32_kernel<<<div_up_i64(total, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W);
    check_last_cuda_error("taehv_upsample2_nchw_f32");
    return out;
}

torch::Tensor taehv_tgrow_reshape_cuda(torch::Tensor x, int64_t stride) {
    WM_CHECK_CUDA(x);
    WM_CHECK_CONTIGUOUS(x);
    WM_CHECK_F32(x);
    TORCH_CHECK(x.dim() == 4, "x must be [N,C*stride,H,W]");
    TORCH_CHECK(stride == 1 || stride == 2, "only stride 1/2 are supported");
    TORCH_CHECK(x.size(1) % stride == 0, "channel count must be divisible by stride");
    int N = (int)x.size(0);
    int C = (int)(x.size(1) / stride);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    auto out = torch::empty({N * stride, C, H, W}, x.options());
    int64_t total = (int64_t)N * stride * C * H * W;
    taehv_tgrow_reshape_f32_kernel<<<div_up_i64(total, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, (int)stride);
    check_last_cuda_error("taehv_tgrow_reshape_f32");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu", &silu_cuda, "WorldModel SiLU (CUDA, f32)");
    m.def("row_major_linear_fp16", &row_major_linear_fp16_cuda, "WorldModel row-major Linear using FP16 inputs/weights and FP32 output");
    m.def("row_major_linear_fp16_tensorop", &row_major_linear_fp16_tensorop_cuda, "WorldModel row-major Linear using CUTLASS FP16 tensor-op and FP32 output");
    m.def("rms_norm", &rms_norm_cuda, "WorldModel RMSNorm (CUDA, f32)", py::arg("x"), py::arg("eps") = 1.0e-6);
    m.def("ada_rms_norm", &ada_rms_norm_cuda, "WorldModel AdaRMSNorm (CUDA, f32)", py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("eps") = 1.0e-6);
    m.def("ortho_rope", &ortho_rope_cuda, "WorldModel OrthoRoPE (CUDA, f32)");
    m.def("qkv_rms_rope", &qkv_rms_rope_cuda, "WorldModel fused QKV split + RMSNorm + OrthoRoPE (CUDA, f32)", py::arg("qkv"), py::arg("x_pos"), py::arg("y_pos"), py::arg("t_pos"), py::arg("xy"), py::arg("inv_t"), py::arg("n_heads"), py::arg("n_kv_heads"), py::arg("width"), py::arg("height"), py::arg("eps") = 1.0e-6);
    m.def("masked_attention", &masked_attention_cuda, "WorldModel written-mask GQA attention (CUDA, f32)", py::arg("q"), py::arg("k"), py::arg("v"), py::arg("written"), py::arg("scale"));
    m.def("indexed_attention", &indexed_attention_cuda, "WorldModel indexed GQA attention (CUDA, f32)", py::arg("q"), py::arg("k"), py::arg("v"), py::arg("indices"), py::arg("scale"));
    m.def("indexed_attention_flash", &indexed_attention_flash_cuda, "WorldModel fused online-softmax indexed GQA attention (CUDA, f32)", py::arg("q"), py::arg("k"), py::arg("v"), py::arg("indices"), py::arg("scale"));
    m.def("kv_cache_upsert", &kv_cache_upsert_cuda, "WorldModel KV ring-cache upsert (CUDA, f32)", py::arg("cache_k"), py::arg("cache_v"), py::arg("written"), py::arg("k"), py::arg("v"), py::arg("frame_idx"), py::arg("ring_length"), py::arg("pinned_dilation"), py::arg("frozen"));
    m.def("cache_frame_indices", &cache_frame_indices_cuda, "WorldModel frame-slot cache index collection (CUDA)", py::arg("written"), py::arg("tokens_per_frame"), py::arg("base"), py::arg("write_step"));
    m.def("patchify", &patchify_cuda, "WorldModel patchify Conv2d + token layout (CUDA, f32)");
    m.def("patchify_cutlass", &patchify_cutlass_cuda, "WorldModel patchify im2row + CUTLASS GEMM (CUDA, f32)");
    m.def("unpatchify", &unpatchify_cuda, "WorldModel unpatchify Linear + image layout (CUDA, f32)", py::arg("tokens"), py::arg("weight"), py::arg("bias"), py::arg("channels"), py::arg("height"), py::arg("width"), py::arg("patch_h"), py::arg("patch_w"));
    m.def("taehv_conv2d", &taehv_conv2d_cuda, "TAEHV direct same-padding Conv2d (CUDA, f32)");
    m.def("taehv_conv1x1_cutlass", &taehv_conv1x1_cutlass_cuda, "TAEHV 1x1 Conv2d using per-frame CUTLASS GEMM (CUDA, f32)");
    m.def("taehv_conv3x3_cutlass", &taehv_conv3x3_cutlass_cuda, "TAEHV 3x3 Conv2d using im2col tile + CUTLASS GEMM (CUDA, f32)");
    m.def("taehv_conv3x3_cutlass_batched", &taehv_conv3x3_cutlass_batched_cuda, "TAEHV 3x3 Conv2d using batch-spatial im2col tile + CUTLASS GEMM (CUDA, f32)", py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("tile_cols"));
    m.def("taehv_conv3x3_cutlass_implicit_nhwc", &taehv_conv3x3_cutlass_implicit_nhwc_cuda, "TAEHV 3x3 Conv2d using CUTLASS implicit-GEMM NHWC/KRSC probe (CUDA, f32)");
    m.def("taehv_concat_past", &taehv_concat_past_cuda, "TAEHV MemBlock current+past concat (CUDA, f32)");
    m.def("taehv_upsample2", &taehv_upsample2_cuda, "TAEHV nearest upsample x2 (CUDA, f32)");
    m.def("taehv_tgrow_reshape", &taehv_tgrow_reshape_cuda, "TAEHV TGrow channel-to-time reshape (CUDA, f32)");
}
