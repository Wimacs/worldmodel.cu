#include "world_cuda.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef WORLD_USE_CUDNN
#include <cudnn.h>
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

#define WORLD_ATTN_D64_K_BLOCK 64
#define WORLD_ATTN_D64_FLASH_WARPS 16

#define CUDA_OK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

#define CUBLAS_OK(expr) do { \
    cublasStatus_t _s = (expr); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        return 1; \
    } \
} while (0)

#ifdef WORLD_USE_CUDNN
#define CUDNN_OK(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(_s)); \
        return 1; \
    } \
} while (0)
#endif

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

__global__ static void add_bias_silu_f32_kernel(const float *x, const float *bias, float *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = wm_silu(x[i] + bias[i]);
}

__global__ static void add_channel_silu_inplace_f32_kernel(float *x, const float *bias, int rows, int D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)rows * D;
    if (i < n) x[i] = wm_silu(x[i] + bias[i % D]);
}

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

__global__ static void collect_cache_frame_indices_kernel(
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

__global__ static void gather_indexed_kv_d64_f32_kernel(
        const float *__restrict__ cache_k,
        const float *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        float *__restrict__ k_compact,
        float *__restrict__ v_compact,
        int Hq,
        int Hkv,
        int Nkv,
        int Tk) {
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

__global__ static void gather_indexed_kv_hkv_d64_f32_kernel(
        const float *__restrict__ cache_k,
        const float *__restrict__ cache_v,
        const int64_t *__restrict__ indices,
        float *__restrict__ k_compact,
        float *__restrict__ v_compact,
        int Hkv,
        int Nkv,
        int Tk) {
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

__global__ static void taehv_repeat_latent4_kernel(
        const float *latent,
        float *out,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t frame_elems = (int64_t)C * H * W;
    int64_t total = 4 * frame_elems;
    if (i >= total) return;
    out[i] = latent[i % frame_elems];
}

__global__ static void taehv_clamp_kernel(float *x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / 3.0f) * 3.0f;
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
        out[i] = x[(((int64_t)n - 1) * C * H + c * H + h) * W + w];
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

__global__ static void taehv_pixel_shuffle_last4_u8_kernel(
        const float *in,
        unsigned char *rgb,
        int N,
        int H,
        int W) {
    int frames = 4;
    int H2 = H * 2;
    int W2 = W * 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)frames * H2 * W2 * 3;
    if (i >= total) return;

    int c = (int)(i % 3);
    int64_t q = i / 3;
    int ox = (int)(q % W2);
    q /= W2;
    int oy = (int)(q % H2);
    int f = (int)(q / H2);
    int n = N - frames + f;
    int ic = c * 4 + (oy & 1) * 2 + (ox & 1);
    float v = in[((int64_t)n * 12 * H + ic * H + oy / 2) * W + ox / 2];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    int u = (int)floorf(v * 255.0f + 0.5f);
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    rgb[i] = (unsigned char)u;
}

__global__ static void taehv_repeat_latent4_clamp_nhwc_h_kernel(
        const float *latent,
        __half *out,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)4 * H * W * C;
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

__global__ static void taehv_add_bias_nhwc_h_kernel(__half *x, const __half *bias, int N, int H, int W, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H * W * C;
    if (i >= total) return;
    int c = (int)(i % C);
    float v = __half2float(x[i]) + __half2float(bias[c]);
    x[i] = __float2half_rn(v);
}

__global__ static void taehv_concat_past_nhwc_h_kernel(
        const __half *x,
        __half *out,
        int N,
        int C,
        int H,
        int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H * W * (2 * C);
    if (i >= total) return;
    int c2 = (int)(i % (2 * C));
    int64_t q = i / (2 * C);
    int w = (int)(q % W);
    q /= W;
    int h = (int)(q % H);
    int n = (int)(q / H);
    if (c2 < C) {
        out[i] = x[(((int64_t)n * H + h) * W + w) * C + c2];
    } else if (n == 0) {
        out[i] = __float2half_rn(0.0f);
    } else {
        int c = c2 - C;
        out[i] = x[((((int64_t)n - 1) * H + h) * W + w) * C + c];
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

__global__ static void taehv_pixel_shuffle_last4_u8_nhwc_h_kernel(
        const __half *in,
        unsigned char *rgb,
        int N,
        int H,
        int W) {
    int frames = 4;
    int H2 = H * 2;
    int W2 = W * 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)frames * H2 * W2 * 3;
    if (i >= total) return;

    int c = (int)(i % 3);
    int64_t q = i / 3;
    int ox = (int)(q % W2);
    q /= W2;
    int oy = (int)(q % H2);
    int f = (int)(q / H2);
    int n = N - frames + f;
    int ic = c * 4 + (oy & 1) * 2 + (ox & 1);
    float v = __half2float(in[(((int64_t)n * H + oy / 2) * W + ox / 2) * 12 + ic]);
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
        cublasHandle_t handle,
        const float *x_rm,
        const float *w_rm,
        float *y_rm,
        int m,
        int k,
        int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major y[m,n] = x[m,k] * w[n,k]^T.
    // Interpreted by cuBLAS as column-major:
    // Y_col[n,m] = W_col[k,n]^T * X_col[k,m].
    CUBLAS_OK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        w_rm,
        k,
        x_rm,
        k,
        &beta,
        y_rm,
        n));
    return 0;
}

static int row_major_linear_fp16_weight(
        cublasHandle_t handle,
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

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_OK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        w_rm_h,
        CUDA_R_16F,
        k,
        x_half_tmp,
        CUDA_R_16F,
        k,
        &beta,
        y_rm,
        CUDA_R_32F,
        n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

typedef struct {
    float *cond_bias;
    float *cond_proj_weight;
    float *qkv_proj_weight;
    __half *qkv_proj_weight_h;
    float *out_proj_weight;
    __half *out_proj_weight_h;
    float v_lamb;
    float *ctrl_fc1_x_weight;
    __half *ctrl_fc1_x_weight_h;
    float *ctrl_fc1_c_weight;
    float *ctrl_fc2_weight;
    __half *ctrl_fc2_weight_h;
    float *dit_mlp_fc1_weight;
    __half *dit_mlp_fc1_weight_h;
    float *dit_mlp_fc2_weight;
    __half *dit_mlp_fc2_weight_h;
    int has_ctrl;
} DeviceWorldLayerWeights;

typedef struct {
    float *k;
    float *v;
    bool *written;
    int64_t *indices;
    int *index_count;
    int ring_length;
    int capacity;
    int pinned_dilation;
    int is_global;
} DeviceWorldLayerCache;

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
        cudaFree(layers[i].out_proj_weight);
        cudaFree(layers[i].out_proj_weight_h);
        cudaFree(layers[i].ctrl_fc1_x_weight);
        cudaFree(layers[i].ctrl_fc1_x_weight_h);
        cudaFree(layers[i].ctrl_fc1_c_weight);
        cudaFree(layers[i].ctrl_fc2_weight);
        cudaFree(layers[i].ctrl_fc2_weight_h);
        cudaFree(layers[i].dit_mlp_fc1_weight);
        cudaFree(layers[i].dit_mlp_fc1_weight_h);
        cudaFree(layers[i].dit_mlp_fc2_weight);
        cudaFree(layers[i].dit_mlp_fc2_weight_h);
    }
    free(layers);
}

static void free_device_world_caches(DeviceWorldLayerCache *caches, int n_layers) {
    if (!caches) return;
    for (int i = 0; i < n_layers; ++i) {
        cudaFree(caches[i].k);
        cudaFree(caches[i].v);
        cudaFree(caches[i].written);
        cudaFree(caches[i].indices);
        cudaFree(caches[i].index_count);
    }
    free(caches);
}

static int alloc_device_world_caches(
        DeviceWorldLayerCache **dst_caches,
        const WorldConfig *cfg,
        int n_layers,
        int T,
        int n_kv_heads,
        int d_head) {
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
        if (c->ring_length % T != 0 || ((c->ring_length / T) % c->pinned_dilation) != 0) goto fail;

        size_t kv_elems = (size_t)n_kv_heads * c->capacity * d_head;
        if (cudaMalloc((void **)&c->k, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->v, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->written, (size_t)c->capacity * sizeof(bool)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->indices, (size_t)c->capacity * sizeof(int64_t)) != cudaSuccess) goto fail;
        if (cudaMalloc((void **)&c->index_count, sizeof(int)) != cudaSuccess) goto fail;
        if (cudaMemset(c->k, 0, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
        if (cudaMemset(c->v, 0, kv_elems * sizeof(float)) != cudaSuccess) goto fail;
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

static int copy_world_layers_to_device(
        DeviceWorldLayerWeights **dst_layers,
        const WorldLayerWeights *src_layers,
        int n_layers,
        int D,
        int kv_dim,
        int mlp_hidden) {
    *dst_layers = NULL;
    DeviceWorldLayerWeights *dst = (DeviceWorldLayerWeights *)calloc((size_t)n_layers, sizeof(*dst));
    if (!dst) return 1;

    for (int i = 0; i < n_layers; ++i) {
        const WorldLayerWeights *src = &src_layers[i];
        DeviceWorldLayerWeights *dl = &dst[i];
        dl->has_ctrl = src->has_ctrl;
        dl->v_lamb = src->v_lamb ? src->v_lamb[0] : 0.0f;
        if (copy_f32_to_device(&dl->cond_bias, src->cond_bias, (size_t)D)) goto fail;
        if (copy_cond_proj_to_device(&dl->cond_proj_weight, src, D)) goto fail;
        if (copy_qkv_proj_to_device(&dl->qkv_proj_weight, src, D, kv_dim)) goto fail;
        if (copy_qkv_proj_to_half_device(&dl->qkv_proj_weight_h, src, D, kv_dim)) goto fail;
        if (copy_f32_to_device(&dl->out_proj_weight, src->out_proj_weight, (size_t)D * D)) goto fail;
        if (copy_f32_to_half_device(&dl->out_proj_weight_h, src->out_proj_weight, (size_t)D * D)) goto fail;
        if (src->has_ctrl) {
            if (copy_f32_to_device(&dl->ctrl_fc1_x_weight, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_half_device(&dl->ctrl_fc1_x_weight_h, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_device(&dl->ctrl_fc1_c_weight, src->ctrl_fc1_c_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_device(&dl->ctrl_fc2_weight, src->ctrl_fc2_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_half_device(&dl->ctrl_fc2_weight_h, src->ctrl_fc2_weight, (size_t)D * D)) goto fail;
        }
        if (copy_f32_to_device(&dl->dit_mlp_fc1_weight, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
        if (copy_f32_to_half_device(&dl->dit_mlp_fc1_weight_h, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
        if (copy_f32_to_device(&dl->dit_mlp_fc2_weight, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
        if (copy_f32_to_half_device(&dl->dit_mlp_fc2_weight_h, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
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
    __half *bias_h;
    int out_c;
    int in_c;
    int kernel;
    int has_bias;
#ifdef WORLD_USE_CUDNN
    cudnnTensorDescriptor_t in_desc;
    cudnnTensorDescriptor_t out_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_bytes;
    int plan_ready;
    int plan_n;
    int plan_h;
    int plan_w;
    cudnnTensorDescriptor_t h_in_desc;
    cudnnTensorDescriptor_t h_out_desc;
    cudnnFilterDescriptor_t h_filter_desc;
    cudnnConvolutionDescriptor_t h_conv_desc;
    cudnnConvolutionFwdAlgo_t h_algo;
    size_t h_workspace_bytes;
    int h_plan_ready;
    int h_plan_n;
    int h_plan_h;
    int h_plan_w;
#endif
} DeviceVaeConvWeight;

typedef struct {
    DeviceVaeConvWeight convs[WORLD_VAE_DECODER_CONV_COUNT];
    float *buf0;
    float *buf1;
    float *buf2;
    __half *hbuf0;
    __half *hbuf1;
    __half *hbuf2;
    unsigned char *d_rgb;
    unsigned char *h_rgb;
    size_t max_elems;
    size_t rgb_elems;
#ifdef WORLD_USE_CUDNN
    cudnnHandle_t cudnn;
    void *cudnn_workspace;
    size_t cudnn_workspace_bytes;
#endif
    int out_w;
    int out_h;
    int H_pre_shuffle;
    int W_pre_shuffle;
    int fp16_nhwc_enabled;
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

static int taehv_copy_weights(DeviceVaeConvWeight dev[WORLD_VAE_DECODER_CONV_COUNT], const WorldVaeDecoderWeights *host) {
    memset(dev, 0, WORLD_VAE_DECODER_CONV_COUNT * sizeof(dev[0]));
    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
        const WorldVaeConvWeight *src = &host->convs[i];
        DeviceVaeConvWeight *dst = &dev[i];
        dst->out_c = src->out_c;
        dst->in_c = src->in_c;
        dst->kernel = src->kernel;
        dst->has_bias = src->has_bias;
        size_t w_elems = (size_t)src->out_c * src->in_c * src->kernel * src->kernel;
        CUDA_OK(cudaMalloc((void **)&dst->weight, w_elems * sizeof(float)));
        CUDA_OK(cudaMemcpy(dst->weight, src->weight, w_elems * sizeof(float), cudaMemcpyHostToDevice));
        if (copy_f32_to_half_device(&dst->weight_h, src->weight, w_elems)) return 1;
        if (src->has_bias) {
            CUDA_OK(cudaMalloc((void **)&dst->bias, (size_t)src->out_c * sizeof(float)));
            CUDA_OK(cudaMemcpy(dst->bias, src->bias, (size_t)src->out_c * sizeof(float), cudaMemcpyHostToDevice));
            if (copy_f32_to_half_device(&dst->bias_h, src->bias, (size_t)src->out_c)) return 1;
        }
    }
    return 0;
}

static void taehv_free_weights(DeviceVaeConvWeight dev[WORLD_VAE_DECODER_CONV_COUNT]) {
    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
#ifdef WORLD_USE_CUDNN
        if (dev[i].bias_desc) cudnnDestroyTensorDescriptor(dev[i].bias_desc);
        if (dev[i].conv_desc) cudnnDestroyConvolutionDescriptor(dev[i].conv_desc);
        if (dev[i].filter_desc) cudnnDestroyFilterDescriptor(dev[i].filter_desc);
        if (dev[i].out_desc) cudnnDestroyTensorDescriptor(dev[i].out_desc);
        if (dev[i].in_desc) cudnnDestroyTensorDescriptor(dev[i].in_desc);
        if (dev[i].h_conv_desc) cudnnDestroyConvolutionDescriptor(dev[i].h_conv_desc);
        if (dev[i].h_filter_desc) cudnnDestroyFilterDescriptor(dev[i].h_filter_desc);
        if (dev[i].h_out_desc) cudnnDestroyTensorDescriptor(dev[i].h_out_desc);
        if (dev[i].h_in_desc) cudnnDestroyTensorDescriptor(dev[i].h_in_desc);
#endif
        cudaFree(dev[i].weight);
        cudaFree(dev[i].bias);
        cudaFree(dev[i].weight_h);
        cudaFree(dev[i].bias_h);
        dev[i].weight = NULL;
        dev[i].bias = NULL;
        dev[i].weight_h = NULL;
        dev[i].bias_h = NULL;
    }
}

static void taehv_decoder_free(DeviceVaeDecoder *dec) {
    if (!dec) return;
#ifdef WORLD_USE_CUDNN
    cudaFree(dec->cudnn_workspace);
    if (dec->cudnn) cudnnDestroy(dec->cudnn);
#endif
    taehv_free_weights(dec->convs);
    cudaFree(dec->buf0);
    cudaFree(dec->buf1);
    cudaFree(dec->buf2);
    cudaFree(dec->hbuf0);
    cudaFree(dec->hbuf1);
    cudaFree(dec->hbuf2);
    cudaFree(dec->d_rgb);
    cudaFreeHost(dec->h_rgb);
    memset(dec, 0, sizeof(*dec));
}

#ifdef WORLD_USE_CUDNN
static int taehv_prepare_conv_plan(DeviceVaeDecoder *dec, DeviceVaeConvWeight *conv, int N, int H, int W) {
    if (!dec || !dec->cudnn || !conv) return 1;
    size_t workspace_bytes = 0;

#define VAE_CUDNN_PLAN_OK(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(_s)); \
        return 1; \
    } \
} while (0)

    VAE_CUDNN_PLAN_OK(cudnnCreateTensorDescriptor(&conv->in_desc));
    VAE_CUDNN_PLAN_OK(cudnnCreateTensorDescriptor(&conv->out_desc));
    VAE_CUDNN_PLAN_OK(cudnnCreateFilterDescriptor(&conv->filter_desc));
    VAE_CUDNN_PLAN_OK(cudnnCreateConvolutionDescriptor(&conv->conv_desc));
    VAE_CUDNN_PLAN_OK(cudnnSetTensor4dDescriptor(
        conv->in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, conv->in_c, H, W));
    VAE_CUDNN_PLAN_OK(cudnnSetTensor4dDescriptor(
        conv->out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, conv->out_c, H, W));
    VAE_CUDNN_PLAN_OK(cudnnSetFilter4dDescriptor(
        conv->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv->out_c, conv->in_c, conv->kernel, conv->kernel));
    VAE_CUDNN_PLAN_OK(cudnnSetConvolution2dDescriptor(
        conv->conv_desc, conv->kernel / 2, conv->kernel / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    VAE_CUDNN_PLAN_OK(cudnnSetConvolutionMathType(conv->conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    conv->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    {
        int algo_count = 0;
        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t perf[16];
        if (cudnnGetConvolutionForwardAlgorithmMaxCount(dec->cudnn, &algo_count) == CUDNN_STATUS_SUCCESS) {
            if (algo_count > (int)(sizeof(perf) / sizeof(perf[0]))) algo_count = (int)(sizeof(perf) / sizeof(perf[0]));
            if (algo_count > 0 &&
                cudnnGetConvolutionForwardAlgorithm_v7(
                    dec->cudnn, conv->in_desc, conv->filter_desc, conv->conv_desc, conv->out_desc,
                    algo_count, &returned, perf) == CUDNN_STATUS_SUCCESS) {
                for (int i = 0; i < returned; ++i) {
                    if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= dec->cudnn_workspace_bytes) {
                        conv->algo = perf[i].algo;
                        break;
                    }
                }
            }
        }
    }
    if (cudnnGetConvolutionForwardWorkspaceSize(
                dec->cudnn, conv->in_desc, conv->filter_desc, conv->conv_desc, conv->out_desc,
                conv->algo, &workspace_bytes) != CUDNN_STATUS_SUCCESS ||
            workspace_bytes > dec->cudnn_workspace_bytes) {
        conv->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        workspace_bytes = 0;
    }
    conv->workspace_bytes = workspace_bytes;

    if (conv->has_bias) {
        VAE_CUDNN_PLAN_OK(cudnnCreateTensorDescriptor(&conv->bias_desc));
        VAE_CUDNN_PLAN_OK(cudnnSetTensor4dDescriptor(
            conv->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv->out_c, 1, 1));
    }
    conv->plan_ready = 1;
    conv->plan_n = N;
    conv->plan_h = H;
    conv->plan_w = W;
#undef VAE_CUDNN_PLAN_OK
    return 0;
}

static int taehv_prepare_conv_plan_h_nhwc(DeviceVaeDecoder *dec, DeviceVaeConvWeight *conv, int N, int H, int W) {
    if (!dec || !dec->cudnn || !conv || !conv->weight_h) return 1;
    size_t workspace_bytes = 0;

#define VAE_CUDNN_PLAN_H_OK(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN half NHWC error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(_s)); \
        return 1; \
    } \
} while (0)

    VAE_CUDNN_PLAN_H_OK(cudnnCreateTensorDescriptor(&conv->h_in_desc));
    VAE_CUDNN_PLAN_H_OK(cudnnCreateTensorDescriptor(&conv->h_out_desc));
    VAE_CUDNN_PLAN_H_OK(cudnnCreateFilterDescriptor(&conv->h_filter_desc));
    VAE_CUDNN_PLAN_H_OK(cudnnCreateConvolutionDescriptor(&conv->h_conv_desc));
    VAE_CUDNN_PLAN_H_OK(cudnnSetTensor4dDescriptor(
        conv->h_in_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, conv->in_c, H, W));
    VAE_CUDNN_PLAN_H_OK(cudnnSetTensor4dDescriptor(
        conv->h_out_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, conv->out_c, H, W));
    VAE_CUDNN_PLAN_H_OK(cudnnSetFilter4dDescriptor(
        conv->h_filter_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, conv->out_c, conv->in_c, conv->kernel, conv->kernel));
    VAE_CUDNN_PLAN_H_OK(cudnnSetConvolution2dDescriptor(
        conv->h_conv_desc, conv->kernel / 2, conv->kernel / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    VAE_CUDNN_PLAN_H_OK(cudnnSetConvolutionMathType(conv->h_conv_desc, CUDNN_TENSOR_OP_MATH));

    conv->h_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    {
        int algo_count = 0;
        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t perf[16];
        if (cudnnGetConvolutionForwardAlgorithmMaxCount(dec->cudnn, &algo_count) == CUDNN_STATUS_SUCCESS) {
            if (algo_count > (int)(sizeof(perf) / sizeof(perf[0]))) algo_count = (int)(sizeof(perf) / sizeof(perf[0]));
            if (algo_count > 0 &&
                cudnnGetConvolutionForwardAlgorithm_v7(
                    dec->cudnn, conv->h_in_desc, conv->h_filter_desc, conv->h_conv_desc, conv->h_out_desc,
                    algo_count, &returned, perf) == CUDNN_STATUS_SUCCESS) {
                for (int i = 0; i < returned; ++i) {
                    if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= dec->cudnn_workspace_bytes) {
                        conv->h_algo = perf[i].algo;
                        break;
                    }
                }
            }
        }
    }
    if (cudnnGetConvolutionForwardWorkspaceSize(
                dec->cudnn, conv->h_in_desc, conv->h_filter_desc, conv->h_conv_desc, conv->h_out_desc,
                conv->h_algo, &workspace_bytes) != CUDNN_STATUS_SUCCESS ||
            workspace_bytes > dec->cudnn_workspace_bytes) {
        conv->h_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        workspace_bytes = 0;
    }
    conv->h_workspace_bytes = workspace_bytes;
    conv->h_plan_ready = 1;
    conv->h_plan_n = N;
    conv->h_plan_h = H;
    conv->h_plan_w = W;
#undef VAE_CUDNN_PLAN_H_OK
    return 0;
}

static int taehv_prepare_conv_plans(DeviceVaeDecoder *dec, const WorldConfig *cfg) {
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int N = 4;
    int H = H0;
    int W = W0;

#define VAE_PLAN(idx) do { \
    if (taehv_prepare_conv_plan(dec, &dec->convs[(idx)], N, H, W)) return 1; \
} while (0)
#define VAE_PLAN_MEMBLOCK(a, b, c) do { \
    VAE_PLAN(a); \
    VAE_PLAN(b); \
    VAE_PLAN(c); \
} while (0)
#define VAE_PLAN_UPSAMPLE2() do { \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_PLAN_TGROW(idx, stride_value) do { \
    int _stride = (stride_value); \
    VAE_PLAN(idx); \
    N *= _stride; \
} while (0)

    VAE_PLAN(WORLD_VAE_DEC_CONV_IN);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_PLAN_UPSAMPLE2();
    VAE_PLAN_TGROW(WORLD_VAE_DEC_TGROW7, 1);

    VAE_PLAN(WORLD_VAE_DEC_CONV8);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_PLAN_UPSAMPLE2();
    VAE_PLAN_TGROW(WORLD_VAE_DEC_TGROW13, 2);

    VAE_PLAN(WORLD_VAE_DEC_CONV14);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
    VAE_PLAN_MEMBLOCK(WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
    VAE_PLAN_UPSAMPLE2();
    VAE_PLAN_TGROW(WORLD_VAE_DEC_TGROW19, 2);

    VAE_PLAN(WORLD_VAE_DEC_CONV20);
    VAE_PLAN(WORLD_VAE_DEC_CONV_OUT);

#undef VAE_PLAN
#undef VAE_PLAN_MEMBLOCK
#undef VAE_PLAN_UPSAMPLE2
#undef VAE_PLAN_TGROW
    return 0;
}

static int taehv_prepare_conv_plans_h_nhwc(DeviceVaeDecoder *dec, const WorldConfig *cfg) {
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int N = 4;
    int H = H0;
    int W = W0;

#define VAE_PLAN_H(idx) do { \
    if (taehv_prepare_conv_plan_h_nhwc(dec, &dec->convs[(idx)], N, H, W)) return 1; \
} while (0)
#define VAE_PLAN_H_MEMBLOCK(a, b, c) do { \
    VAE_PLAN_H(a); \
    VAE_PLAN_H(b); \
    VAE_PLAN_H(c); \
} while (0)
#define VAE_PLAN_H_UPSAMPLE2() do { \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_PLAN_H_TGROW(idx, stride_value) do { \
    int _stride = (stride_value); \
    VAE_PLAN_H(idx); \
    N *= _stride; \
} while (0)

    VAE_PLAN_H(WORLD_VAE_DEC_CONV_IN);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_PLAN_H_UPSAMPLE2();
    VAE_PLAN_H_TGROW(WORLD_VAE_DEC_TGROW7, 1);

    VAE_PLAN_H(WORLD_VAE_DEC_CONV8);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_PLAN_H_UPSAMPLE2();
    VAE_PLAN_H_TGROW(WORLD_VAE_DEC_TGROW13, 2);

    VAE_PLAN_H(WORLD_VAE_DEC_CONV14);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
    VAE_PLAN_H_MEMBLOCK(WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
    VAE_PLAN_H_UPSAMPLE2();
    VAE_PLAN_H_TGROW(WORLD_VAE_DEC_TGROW19, 2);

    VAE_PLAN_H(WORLD_VAE_DEC_CONV20);
    VAE_PLAN_H(WORLD_VAE_DEC_CONV_OUT);

#undef VAE_PLAN_H
#undef VAE_PLAN_H_MEMBLOCK
#undef VAE_PLAN_H_UPSAMPLE2
#undef VAE_PLAN_H_TGROW
    return 0;
}
#endif

static int taehv_decoder_init(DeviceVaeDecoder *dec, const WorldConfig *cfg, const WorldVaeDecoderWeights *host) {
    memset(dec, 0, sizeof(*dec));
    if (!host) return 0;
    const char *vae_fp16_env = getenv("WORLD_VAE_FP16_NHWC");

    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    dec->H_pre_shuffle = H0 * 8;
    dec->W_pre_shuffle = W0 * 8;
    dec->out_h = dec->H_pre_shuffle * 2;
    dec->out_w = dec->W_pre_shuffle * 2;
    dec->max_elems = (size_t)16 * 64 * dec->H_pre_shuffle * dec->W_pre_shuffle;
    dec->rgb_elems = (size_t)4 * dec->out_h * dec->out_w * 3;

#define VAE_INIT_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto fail; \
    } \
} while (0)

    if (taehv_copy_weights(dec->convs, host)) goto fail;
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->buf0, dec->max_elems * sizeof(float)));
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->buf1, dec->max_elems * sizeof(float)));
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->buf2, dec->max_elems * sizeof(float)));
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->d_rgb, dec->rgb_elems));
#ifdef WORLD_USE_CUDNN
    if (cudnnCreate(&dec->cudnn) != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "failed to create cuDNN handle\n");
        goto fail;
    }
    dec->cudnn_workspace_bytes = 128ull * 1024ull * 1024ull;
    VAE_INIT_CUDA(cudaMalloc((void **)&dec->cudnn_workspace, dec->cudnn_workspace_bytes));
    if (taehv_prepare_conv_plans(dec, cfg)) goto fail;
    dec->fp16_nhwc_enabled = vae_fp16_env ? vae_fp16_env[0] != '0' : 1;
    if (dec->fp16_nhwc_enabled) {
        cudaError_t h0 = cudaMalloc((void **)&dec->hbuf0, dec->max_elems * sizeof(__half));
        cudaError_t h1 = h0 == cudaSuccess ? cudaMalloc((void **)&dec->hbuf1, dec->max_elems * sizeof(__half)) : h0;
        cudaError_t h2 = h1 == cudaSuccess ? cudaMalloc((void **)&dec->hbuf2, dec->max_elems * sizeof(__half)) : h1;
        if (h2 != cudaSuccess) {
            fprintf(stderr, "VAE FP16/NHWC scratch unavailable, falling back to F32/NCHW: %s\n", cudaGetErrorString(h2));
            cudaFree(dec->hbuf0);
            cudaFree(dec->hbuf1);
            cudaFree(dec->hbuf2);
            dec->hbuf0 = NULL;
            dec->hbuf1 = NULL;
            dec->hbuf2 = NULL;
            dec->fp16_nhwc_enabled = 0;
        } else if (taehv_prepare_conv_plans_h_nhwc(dec, cfg)) {
            fprintf(stderr, "VAE FP16/NHWC cuDNN plan unavailable, falling back to F32/NCHW\n");
            dec->fp16_nhwc_enabled = 0;
        }
    }
#endif
    VAE_INIT_CUDA(cudaMallocHost((void **)&dec->h_rgb, dec->rgb_elems));

    fprintf(stderr, "VAE decoder init: RGB %dx%d, scratch %.2f MiB x3%s%s, pinned RGB host buffer\n",
            dec->out_w, dec->out_h, (double)(dec->max_elems * sizeof(float)) / (1024.0 * 1024.0),
#ifdef WORLD_USE_CUDNN
            ", cuDNN conv plans enabled",
            dec->fp16_nhwc_enabled ? ", FP16/NHWC enabled" : ""
#else
            "",
            ""
#endif
    );
#undef VAE_INIT_CUDA
    return 0;

fail:
#undef VAE_INIT_CUDA
    taehv_decoder_free(dec);
    return 1;
}

static int taehv_run_conv(DeviceVaeDecoder *dec, const float *in, float *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
#ifdef WORLD_USE_CUDNN
    if (dec && dec->cudnn && conv->plan_ready && conv->plan_n == N && conv->plan_h == H && conv->plan_w == W) {
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
#define CUDNN_CONV_OK(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(_s)); \
        return 1; \
    } \
} while (0)
            CUDNN_CONV_OK(cudnnConvolutionForward(
                dec->cudnn,
                &alpha,
                conv->in_desc,
                in,
                conv->filter_desc,
                conv->weight,
                conv->conv_desc,
                conv->algo,
                dec->cudnn_workspace,
                conv->workspace_bytes,
                &beta,
                conv->out_desc,
                out));
            if (conv->has_bias) {
                CUDNN_CONV_OK(cudnnAddTensor(dec->cudnn, &alpha, conv->bias_desc, conv->bias, &alpha, conv->out_desc, out));
            }
#undef CUDNN_CONV_OK
        }
        return 0;
    }
#endif
    int64_t total = (int64_t)N * conv->out_c * H * W;
    taehv_conv2d_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, conv->weight, conv->bias, out, N, conv->in_c, conv->out_c, H, W, conv->kernel, conv->has_bias);
    CUDA_OK(cudaGetLastError());
    return 0;
}

static int taehv_run_conv_h_nhwc(DeviceVaeDecoder *dec, const __half *in, __half *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
#ifdef WORLD_USE_CUDNN
    if (!dec || !dec->cudnn || !conv || !conv->h_plan_ready ||
        conv->h_plan_n != N || conv->h_plan_h != H || conv->h_plan_w != W) {
        return 1;
    }
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
#define CUDNN_CONV_H_OK(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN half NHWC error %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(_s)); \
        return 1; \
    } \
} while (0)
        CUDNN_CONV_H_OK(cudnnConvolutionForward(
            dec->cudnn,
            &alpha,
            conv->h_in_desc,
            in,
            conv->h_filter_desc,
            conv->weight_h,
            conv->h_conv_desc,
            conv->h_algo,
            dec->cudnn_workspace,
            conv->h_workspace_bytes,
            &beta,
            conv->h_out_desc,
            out));
#undef CUDNN_CONV_H_OK
    }
    if (conv->has_bias) {
        taehv_add_bias_nhwc_h_kernel<<<div_up_i64((int64_t)N * H * W * conv->out_c, 256), 256>>>(out, conv->bias_h, N, H, W, conv->out_c);
        CUDA_OK(cudaGetLastError());
    }
    return 0;
#else
    (void)dec;
    (void)in;
    (void)out;
    (void)conv;
    (void)N;
    (void)H;
    (void)W;
    return 1;
#endif
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

static int taehv_run_memblock(
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
    taehv_concat_past_nchw_kernel<<<div_up_i64(elems * 2, 256), 256>>>(cur, aux, N, C, H, W);
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

static int taehv_run_memblock_h_nhwc(
        DeviceVaeDecoder *dec,
        __half **cur_io,
        __half *buf0,
        __half *buf1,
        __half *buf2,
        const DeviceVaeConvWeight *conv0,
        const DeviceVaeConvWeight *conv2,
        const DeviceVaeConvWeight *conv4,
        int N,
        int C,
        int H,
        int W) {
    __half *cur = *cur_io;
    __half *tmp = NULL;
    __half *aux = NULL;
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux);

    int64_t elems = (int64_t)N * H * W * C;
    taehv_concat_past_nhwc_h_kernel<<<div_up_i64(elems * 2, 256), 256>>>(cur, aux, N, C, H, W);
    CUDA_OK(cudaGetLastError());
    if (taehv_run_conv_h_nhwc(dec, aux, tmp, conv0, N, H, W)) return 1;
    if (taehv_run_relu_h(tmp, elems)) return 1;
    if (taehv_run_conv_h_nhwc(dec, tmp, aux, conv2, N, H, W)) return 1;
    if (taehv_run_relu_h(aux, elems)) return 1;
    if (taehv_run_conv_h_nhwc(dec, aux, tmp, conv4, N, H, W)) return 1;
    taehv_add_relu_nhwc_h_kernel<<<div_up_i64(elems, 256), 256>>>(cur, tmp, aux, elems);
    CUDA_OK(cudaGetLastError());
    *cur_io = aux;
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

    int C_latent = cfg->channels;
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int H = H0;
    int W = W0;
    int N = 4;
    int C = C_latent;
    int rc = 1;

    __half *buf0 = dec->hbuf0;
    __half *buf1 = dec->hbuf1;
    __half *buf2 = dec->hbuf2;
    __half *cur = buf0;
    __half *tmp = NULL;
    __half *aux = NULL;

    fprintf(stderr, "VAE decode FP16/NHWC: latent [%d,%d,%d] -> RGB %dx%d\n", C_latent, H0, W0, dec->out_w, dec->out_h);

    taehv_repeat_latent4_clamp_nhwc_h_kernel<<<div_up_i64((int64_t)4 * H * W * C, 256), 256>>>(d_latent, cur, C, H, W);
    CUDA_OK(cudaGetLastError());

#define VAE_CONV_TO_H(idx, out_c) do { \
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv_h_nhwc(dec, cur, tmp, &dec->convs[(idx)], N, H, W)) goto cleanup; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_RELU_H() do { \
    if (taehv_run_relu_h(cur, (int64_t)N * H * W * C)) goto cleanup; \
} while (0)
#define VAE_MEMBLOCK_H(a, b, c) do { \
    if (taehv_run_memblock_h_nhwc(dec, &cur, buf0, buf1, buf2, &dec->convs[(a)], &dec->convs[(b)], &dec->convs[(c)], N, C, H, W)) goto cleanup; \
} while (0)
#define VAE_UPSAMPLE2_H() do { \
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux); \
    taehv_upsample2_nhwc_h_kernel<<<div_up_i64((int64_t)N * (H * 2) * (W * 2) * C, 256), 256>>>(cur, tmp, N, C, H, W); \
    CUDA_OK(cudaGetLastError()); \
    cur = tmp; \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_TGROW_H(idx, stride_value) do { \
    int _stride = (stride_value); \
    taehv_pick_scratch_h(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv_h_nhwc(dec, cur, tmp, &dec->convs[(idx)], N, H, W)) goto cleanup; \
    if (_stride == 1) { \
        cur = tmp; \
    } else { \
        taehv_tgrow_reshape_nhwc_h_kernel<<<div_up_i64((int64_t)N * _stride * H * W * C, 256), 256>>>(tmp, cur, N, C, H, W, _stride); \
        CUDA_OK(cudaGetLastError()); \
        N *= _stride; \
    } \
} while (0)

    VAE_CONV_TO_H(WORLD_VAE_DEC_CONV_IN, 256);
    VAE_RELU_H();
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_UPSAMPLE2_H();
    VAE_TGROW_H(WORLD_VAE_DEC_TGROW7, 1);

    VAE_CONV_TO_H(WORLD_VAE_DEC_CONV8, 128);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_UPSAMPLE2_H();
    VAE_TGROW_H(WORLD_VAE_DEC_TGROW13, 2);

    VAE_CONV_TO_H(WORLD_VAE_DEC_CONV14, 64);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
    VAE_MEMBLOCK_H(WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
    VAE_UPSAMPLE2_H();
    VAE_TGROW_H(WORLD_VAE_DEC_TGROW19, 2);

    VAE_CONV_TO_H(WORLD_VAE_DEC_CONV20, 64);
    VAE_RELU_H();
    VAE_CONV_TO_H(WORLD_VAE_DEC_CONV_OUT, 12);

    taehv_pixel_shuffle_last4_u8_nhwc_h_kernel<<<div_up_i64((int64_t)dec->rgb_elems, 256), 256>>>(cur, dec->d_rgb, N, H, W);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(dec->h_rgb, dec->d_rgb, dec->rgb_elems, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());
    if (rgb_out) *rgb_out = dec->h_rgb;
    if (frame_count_out) *frame_count_out = 4;
    if (width_out) *width_out = dec->out_w;
    if (height_out) *height_out = dec->out_h;
    rc = 0;

cleanup:
#undef VAE_CONV_TO_H
#undef VAE_RELU_H
#undef VAE_MEMBLOCK_H
#undef VAE_UPSAMPLE2_H
#undef VAE_TGROW_H
    return rc;
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

    int C_latent = cfg->channels;
    int H0 = cfg->height * cfg->patch_h;
    int W0 = cfg->width * cfg->patch_w;
    int H = H0;
    int W = W0;
    int N = 4;
    int C = C_latent;
    int rc = 1;

    float *buf0 = dec->buf0;
    float *buf1 = dec->buf1;
    float *buf2 = dec->buf2;
    float *cur = buf0;
    float *tmp = NULL;
    float *aux = NULL;

    fprintf(stderr, "VAE decode: latent [%d,%d,%d] -> RGB %dx%d\n", C_latent, H0, W0, dec->out_w, dec->out_h);

    taehv_repeat_latent4_kernel<<<div_up_i64((int64_t)4 * C * H * W, 256), 256>>>(d_latent, cur, C, H, W);
    CUDA_OK(cudaGetLastError());
    taehv_clamp_kernel<<<div_up_i64((int64_t)N * C * H * W, 256), 256>>>(cur, (int64_t)N * C * H * W);
    CUDA_OK(cudaGetLastError());

#define VAE_CONV_TO(idx, out_c) do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv(dec, cur, tmp, &dec->convs[(idx)], N, H, W)) goto cleanup; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_RELU() do { \
    if (taehv_run_relu(cur, (int64_t)N * C * H * W)) goto cleanup; \
} while (0)
#define VAE_MEMBLOCK(a, b, c) do { \
    if (taehv_run_memblock(dec, &cur, buf0, buf1, buf2, &dec->convs[(a)], &dec->convs[(b)], &dec->convs[(c)], N, C, H, W)) goto cleanup; \
} while (0)
#define VAE_UPSAMPLE2() do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    taehv_upsample2_nchw_kernel<<<div_up_i64((int64_t)N * C * (H * 2) * (W * 2), 256), 256>>>(cur, tmp, N, C, H, W); \
    CUDA_OK(cudaGetLastError()); \
    cur = tmp; \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_TGROW(idx, stride_value) do { \
    int _stride = (stride_value); \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv(dec, cur, tmp, &dec->convs[(idx)], N, H, W)) goto cleanup; \
    if (_stride == 1) { \
        cur = tmp; \
    } else { \
        taehv_tgrow_reshape_kernel<<<div_up_i64((int64_t)N * _stride * C * H * W, 256), 256>>>(tmp, cur, N, C, H, W, _stride); \
        CUDA_OK(cudaGetLastError()); \
        N *= _stride; \
    } \
} while (0)

    VAE_CONV_TO(WORLD_VAE_DEC_CONV_IN, 256);
    VAE_RELU();
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_UPSAMPLE2();
    VAE_TGROW(WORLD_VAE_DEC_TGROW7, 1);

    VAE_CONV_TO(WORLD_VAE_DEC_CONV8, 128);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_UPSAMPLE2();
    VAE_TGROW(WORLD_VAE_DEC_TGROW13, 2);

    VAE_CONV_TO(WORLD_VAE_DEC_CONV14, 64);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
    VAE_UPSAMPLE2();
    VAE_TGROW(WORLD_VAE_DEC_TGROW19, 2);

    VAE_CONV_TO(WORLD_VAE_DEC_CONV20, 64);
    VAE_RELU();
    VAE_CONV_TO(WORLD_VAE_DEC_CONV_OUT, 12);

    taehv_pixel_shuffle_last4_u8_kernel<<<div_up_i64((int64_t)dec->rgb_elems, 256), 256>>>(cur, dec->d_rgb, N, H, W);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(dec->h_rgb, dec->d_rgb, dec->rgb_elems, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());
    if (rgb_out) *rgb_out = dec->h_rgb;
    if (frame_count_out) *frame_count_out = 4;
    if (width_out) *width_out = dec->out_w;
    if (height_out) *height_out = dec->out_h;
    rc = 0;

cleanup:
#undef VAE_CONV_TO
#undef VAE_RELU
#undef VAE_MEMBLOCK
#undef VAE_UPSAMPLE2
#undef VAE_TGROW
    return rc;
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
    size_t latent_elems;
    size_t token_elems;
    size_t kv_rope_elems;
    size_t q_rope_elems;
    size_t linear_half_elems;
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
    float *d_attn_scores;
    float *d_attn_k_compact;
    float *d_attn_v_compact;
    float *d_attn_out;
    const float **h_attn_qk_a;
    const float **h_attn_qk_b;
    float **h_attn_qk_c;
    const float **h_attn_av_a;
    const float **h_attn_av_b;
    float **h_attn_av_c;
    const float **d_attn_qk_a;
    const float **d_attn_qk_b;
    float **d_attn_qk_c;
    const float **d_attn_av_a;
    const float **d_attn_av_b;
    float **d_attn_av_c;
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
    int attn_cublas_enabled;
    int attn_cublas_gqa_enabled;
    int attn_flash_enabled;
    int attn_cublas_max_tokens;
    DeviceWorldLayerWeights *d_layers;
    DeviceWorldLayerCache *d_caches;
    cublasHandle_t handle;
    cudaEvent_t ev_step_start;
    cudaEvent_t ev_after_setup;
    cudaEvent_t ev_after_transformer;
    cudaEvent_t ev_after_vae;
    DeviceVaeDecoder d_vae;
};

static int visible_cache_tokens_pinned1_host(const DeviceWorldLayerCache *cache, int T, int frame_idx) {
    int ring_slots = cache->ring_length / T;
    int slots = frame_idx + 1;
    if (slots < 1) slots = 1;
    if (slots > ring_slots) slots = ring_slots;
    return slots * T;
}

static int indexed_attention_cache_d64_cublas(
        WorldCudaRuntime *rt,
        const DeviceWorldLayerCache *cache,
        int Nkv,
        float scale) {
    if (!rt || !cache || Nkv <= 0 || Nkv > rt->attn_cublas_max_tokens) return 1;
    int Hq = rt->cfg.n_heads;
    int Hkv = rt->cfg.n_kv_heads;
    int Tq = rt->T;
    int group = Hq / Hkv;
    if (group <= 0 || Hq % Hkv != 0) return 1;

    int64_t compact_elems = (int64_t)Hq * Nkv * 64;
    gather_indexed_kv_d64_f32_kernel<<<div_up_i64(compact_elems, 256), 256>>>(
        cache->k, cache->v, cache->indices,
        rt->d_attn_k_compact, rt->d_attn_v_compact,
        Hq, Hkv, Nkv, cache->capacity);
    CUDA_OK(cudaGetLastError());

    const float alpha_qk = scale;
    const float alpha_av = 1.0f;
    const float beta = 0.0f;
    CUBLAS_OK(cublasSgemmStridedBatched(
        rt->handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        Nkv,
        Tq,
        64,
        &alpha_qk,
        rt->d_attn_k_compact,
        64,
        (long long)Nkv * 64,
        rt->d_q,
        64,
        (long long)Tq * 64,
        &beta,
        rt->d_attn_scores,
        Nkv,
        (long long)Tq * Nkv,
        Hq));

    softmax_rows_inplace_f32_kernel<<<Hq * Tq, 256>>>(rt->d_attn_scores, Hq * Tq, Nkv);
    CUDA_OK(cudaGetLastError());

    CUBLAS_OK(cublasSgemmStridedBatched(
        rt->handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        64,
        Tq,
        Nkv,
        &alpha_av,
        rt->d_attn_v_compact,
        64,
        (long long)Nkv * 64,
        rt->d_attn_scores,
        Nkv,
        (long long)Tq * Nkv,
        &beta,
        rt->d_attn,
        Hq * 64,
        64,
        Hq));
    return 0;
}

static int upload_attention_gqa_pointers(WorldCudaRuntime *rt, int Nkv) {
    if (!rt || Nkv <= 0) return 1;
    int Hq = rt->cfg.n_heads;
    int Hkv = rt->cfg.n_kv_heads;
    int Tq = rt->T;
    int group = Hq / Hkv;
    if (group <= 0 || Hq % Hkv != 0) return 1;
    for (int hq = 0; hq < Hq; ++hq) {
        int hk = hq / group;
        rt->h_attn_qk_a[hq] = rt->d_attn_k_compact + (int64_t)hk * Nkv * 64;
        rt->h_attn_qk_b[hq] = rt->d_q + (int64_t)hq * Tq * 64;
        rt->h_attn_qk_c[hq] = rt->d_attn_scores + (int64_t)hq * Tq * Nkv;
        rt->h_attn_av_a[hq] = rt->d_attn_v_compact + (int64_t)hk * Nkv * 64;
        rt->h_attn_av_b[hq] = rt->d_attn_scores + (int64_t)hq * Tq * Nkv;
        rt->h_attn_av_c[hq] = rt->d_attn + (int64_t)hq * 64;
    }
    size_t ptr_bytes = (size_t)Hq * sizeof(float *);
    CUDA_OK(cudaMemcpyAsync(rt->d_attn_qk_a, rt->h_attn_qk_a, ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyAsync(rt->d_attn_qk_b, rt->h_attn_qk_b, ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyAsync(rt->d_attn_qk_c, rt->h_attn_qk_c, ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyAsync(rt->d_attn_av_a, rt->h_attn_av_a, ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyAsync(rt->d_attn_av_b, rt->h_attn_av_b, ptr_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyAsync(rt->d_attn_av_c, rt->h_attn_av_c, ptr_bytes, cudaMemcpyHostToDevice));
    return 0;
}

static int indexed_attention_cache_d64_cublas_gqa(
        WorldCudaRuntime *rt,
        const DeviceWorldLayerCache *cache,
        int Nkv,
        float scale) {
    if (!rt || !cache || Nkv <= 0 || Nkv > rt->attn_cublas_max_tokens) return 1;
    int Hq = rt->cfg.n_heads;
    int Hkv = rt->cfg.n_kv_heads;
    int Tq = rt->T;
    int group = Hq / Hkv;
    if (group <= 0 || Hq % Hkv != 0) return 1;

    int64_t compact_elems = (int64_t)Hkv * Nkv * 64;
    gather_indexed_kv_hkv_d64_f32_kernel<<<div_up_i64(compact_elems, 256), 256>>>(
        cache->k, cache->v, cache->indices,
        rt->d_attn_k_compact, rt->d_attn_v_compact,
        Hkv, Nkv, cache->capacity);
    CUDA_OK(cudaGetLastError());
    if (upload_attention_gqa_pointers(rt, Nkv)) return 1;

    const float alpha_qk = scale;
    const float alpha_av = 1.0f;
    const float beta = 0.0f;
    CUBLAS_OK(cublasSgemmBatched(
        rt->handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        Nkv,
        Tq,
        64,
        &alpha_qk,
        rt->d_attn_qk_a,
        64,
        rt->d_attn_qk_b,
        64,
        &beta,
        rt->d_attn_qk_c,
        Nkv,
        Hq));

    softmax_rows_inplace_f32_kernel<<<Hq * Tq, 256>>>(rt->d_attn_scores, Hq * Tq, Nkv);
    CUDA_OK(cudaGetLastError());

    CUBLAS_OK(cublasSgemmBatched(
        rt->handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        64,
        Tq,
        Nkv,
        &alpha_av,
        rt->d_attn_av_a,
        64,
        rt->d_attn_av_b,
        Nkv,
        &beta,
        rt->d_attn_av_c,
        Hq * 64,
        Hq));
    return 0;
}

static int precompute_runtime_layer_mods(WorldCudaRuntime *rt) {
    if (!rt || !rt->d_layer_mod_table || !rt->d_out_mod_table) return 1;
    const WorldConfig *cfg = &rt->cfg;
    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= rt->steps_to_run;
        float sigma_step = is_cache_pass ? 0.0f : cfg->scheduler_sigmas[pass_idx];

        fill_noise_embedding(rt->h_noise, sigma_step);
        CUDA_OK(cudaMemcpy(rt->d_noise, rt->h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
        if (row_major_linear(rt->handle, rt->d_noise, rt->d_denoise_fc1, rt->d_noise_hidden, 1, 512, rt->mlp_hidden)) return 1;
        silu_f32_kernel<<<div_up_i64(rt->mlp_hidden, 256), 256>>>(rt->d_noise_hidden, rt->d_noise_hidden, rt->mlp_hidden);
        CUDA_OK(cudaGetLastError());
        if (row_major_linear(rt->handle, rt->d_noise_hidden, rt->d_denoise_fc2, rt->d_cond, 1, rt->mlp_hidden, rt->D)) return 1;
        silu_f32_kernel<<<div_up_i64(rt->D, 256), 256>>>(rt->d_cond, rt->d_cond_act, rt->D);
        CUDA_OK(cudaGetLastError());
        if (row_major_linear(
                    rt->handle,
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
            if (row_major_linear(rt->handle, rt->d_cond_act, lw->cond_proj_weight, dst, 1, rt->D, 6 * rt->D)) return 1;
        }
    }
    CUDA_OK(cudaGetLastError());
    return 0;
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

extern "C" void world_cuda_runtime_destroy(WorldCudaRuntime *rt) {
    if (!rt) return;
    taehv_decoder_free(&rt->d_vae);
    if (rt->ev_after_vae) cudaEventDestroy(rt->ev_after_vae);
    if (rt->ev_after_transformer) cudaEventDestroy(rt->ev_after_transformer);
    if (rt->ev_after_setup) cudaEventDestroy(rt->ev_after_setup);
    if (rt->ev_step_start) cudaEventDestroy(rt->ev_step_start);
    if (rt->handle) cublasDestroy(rt->handle);
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
    cudaFree(rt->d_attn_scores);
    cudaFree(rt->d_attn_k_compact);
    cudaFree(rt->d_attn_v_compact);
    cudaFree(rt->d_attn_out);
    free(rt->h_attn_qk_a);
    free(rt->h_attn_qk_b);
    free(rt->h_attn_qk_c);
    free(rt->h_attn_av_a);
    free(rt->h_attn_av_b);
    free(rt->h_attn_av_c);
    cudaFree(rt->d_attn_qk_a);
    cudaFree(rt->d_attn_qk_b);
    cudaFree(rt->d_attn_qk_c);
    cudaFree(rt->d_attn_av_a);
    cudaFree(rt->d_attn_av_b);
    cudaFree(rt->d_attn_av_c);
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
    size_t qkv_token_elems = rt->token_elems + 2 * ((size_t)rt->T * rt->kv_dim);
    size_t patch_weight_elems = (size_t)rt->D * rt->C * rt->ph * rt->pw;
    size_t patch_row_elems = (size_t)rt->T * rt->C * rt->ph * rt->pw;
    size_t out_norm_weight_elems = (size_t)2 * rt->D * rt->D;
    size_t unpatch_weight_elems = (size_t)rt->D * rt->C * rt->ph * rt->pw;
    size_t layer_mod_table_elems = (size_t)rt->total_passes * layers_to_run * 6 * rt->D;
    size_t out_mod_table_elems = (size_t)rt->total_passes * 2 * rt->D;
    const char *cublas_attn_env = NULL;
    const char *cublas_attn_gqa_env = NULL;
    const char *flash_attn_env = NULL;
    int cublas_attn_max_tokens = 0;
    int cublas_attn_supported = 1;

#define RT_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto fail; \
    } \
} while (0)
#define RT_CUBLAS(expr) do { \
    cublasStatus_t _s = (expr); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
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

    if (copy_f32_to_device(&rt->d_patch, weights->patchify_weight, patch_weight_elems)) goto fail;
    if (copy_f32_to_device(&rt->d_denoise_fc1, weights->denoise_fc1_weight, (size_t)rt->mlp_hidden * 512)) goto fail;
    if (copy_f32_to_device(&rt->d_denoise_fc2, weights->denoise_fc2_weight, (size_t)rt->D * rt->mlp_hidden)) goto fail;
    if (copy_f32_to_device(&rt->d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)rt->mlp_hidden * rt->ctrl_dim)) goto fail;
    if (copy_f32_to_device(&rt->d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)rt->D * rt->mlp_hidden)) goto fail;
    if (copy_f32_to_device(&rt->d_out_norm_w, weights->out_norm_fc_weight, out_norm_weight_elems)) goto fail;
    if (copy_f32_to_device(&rt->d_unpatch_w, weights->unpatchify_weight, unpatch_weight_elems)) goto fail;
    if (copy_f32_to_device(&rt->d_unpatch_b, weights->unpatchify_bias, (size_t)rt->C)) goto fail;
    if (copy_world_layers_to_device(&rt->d_layers, weights->layers, layers_to_run, rt->D, rt->kv_dim, rt->mlp_hidden)) goto fail;
    if (alloc_device_world_caches(&rt->d_caches, cfg, layers_to_run, rt->T, cfg->n_kv_heads, rt->d_head)) goto fail;
    cublas_attn_env = getenv("WORLD_CUBLAS_ATTN");
    cublas_attn_gqa_env = getenv("WORLD_CUBLAS_ATTN_GQA");
    flash_attn_env = getenv("WORLD_FLASH_ATTN");
    rt->attn_cublas_gqa_enabled = cublas_attn_gqa_env ? cublas_attn_gqa_env[0] != '0' : 1;
    rt->attn_flash_enabled = flash_attn_env ? flash_attn_env[0] != '0' : 0;
    if (rt->attn_flash_enabled) {
        fprintf(stderr, "fused flash-like attention enabled by WORLD_FLASH_ATTN=1\n");
    }
    if ((!cublas_attn_env || cublas_attn_env[0] != '0') && rt->d_head == 64 && cfg->n_heads % cfg->n_kv_heads == 0) {
        cublas_attn_max_tokens = 0;
        cublas_attn_supported = 1;
        for (int i = 0; i < layers_to_run; ++i) {
            DeviceWorldLayerCache *cache = &rt->d_caches[i];
            if (cache->pinned_dilation != 1) cublas_attn_supported = 0;
            if (cache->ring_length > cublas_attn_max_tokens) cublas_attn_max_tokens = cache->ring_length;
        }
        if (cublas_attn_supported && cublas_attn_max_tokens > 0 && cublas_attn_max_tokens <= 8192) {
            rt->attn_cublas_enabled = 1;
            rt->attn_cublas_max_tokens = cublas_attn_max_tokens;
            RT_CUDA(cudaMalloc((void **)&rt->d_attn_scores, (size_t)cfg->n_heads * rt->T * cublas_attn_max_tokens * sizeof(float)));
            RT_CUDA(cudaMalloc((void **)&rt->d_attn_k_compact, (size_t)cfg->n_heads * cublas_attn_max_tokens * 64 * sizeof(float)));
            RT_CUDA(cudaMalloc((void **)&rt->d_attn_v_compact, (size_t)cfg->n_heads * cublas_attn_max_tokens * 64 * sizeof(float)));
            if (rt->attn_cublas_gqa_enabled) {
                size_t ptr_bytes = (size_t)cfg->n_heads * sizeof(float *);
                rt->h_attn_qk_a = (const float **)malloc(ptr_bytes);
                rt->h_attn_qk_b = (const float **)malloc(ptr_bytes);
                rt->h_attn_qk_c = (float **)malloc(ptr_bytes);
                rt->h_attn_av_a = (const float **)malloc(ptr_bytes);
                rt->h_attn_av_b = (const float **)malloc(ptr_bytes);
                rt->h_attn_av_c = (float **)malloc(ptr_bytes);
                if (!rt->h_attn_qk_a || !rt->h_attn_qk_b || !rt->h_attn_qk_c ||
                    !rt->h_attn_av_a || !rt->h_attn_av_b || !rt->h_attn_av_c) {
                    fprintf(stderr, "runtime attention pointer host allocation failed\n");
                    goto fail;
                }
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_qk_a, ptr_bytes));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_qk_b, ptr_bytes));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_qk_c, ptr_bytes));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_av_a, ptr_bytes));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_av_b, ptr_bytes));
                RT_CUDA(cudaMalloc((void **)&rt->d_attn_av_c, ptr_bytes));
            }
            fprintf(stderr, "cuBLAS attention enabled%s: max_tokens=%d score_scratch=%.2f MiB\n",
                    rt->attn_cublas_gqa_enabled ? " (GQA pointer batched)" : "",
                    cublas_attn_max_tokens,
                    (double)((size_t)cfg->n_heads * rt->T * cublas_attn_max_tokens * sizeof(float)) / (1024.0 * 1024.0));
        } else {
            fprintf(stderr, "cuBLAS attention disabled: pinned1=%d max_tokens=%d\n",
                    cublas_attn_supported, cublas_attn_max_tokens);
        }
    }
    RT_CUDA(cudaMemcpy(rt->d_xy_table, rt->h_xy, (size_t)rt->d_xy * sizeof(float), cudaMemcpyHostToDevice));
    RT_CUDA(cudaMemcpy(rt->d_inv_t, rt->h_inv_t, (size_t)rt->d_t * sizeof(float), cudaMemcpyHostToDevice));
    RT_CUBLAS(cublasCreate(&rt->handle));
    RT_CUBLAS(cublasSetMathMode(rt->handle, CUBLAS_TF32_TENSOR_OP_MATH));
    if (precompute_runtime_layer_mods(rt)) goto fail;
    RT_CUDA(cudaEventCreate(&rt->ev_step_start));
    RT_CUDA(cudaEventCreate(&rt->ev_after_setup));
    RT_CUDA(cudaEventCreate(&rt->ev_after_transformer));
    RT_CUDA(cudaEventCreate(&rt->ev_after_vae));
    if (taehv_decoder_init(&rt->d_vae, cfg, vae)) goto fail;
    RT_CUDA(cudaDeviceSynchronize());

    *out = rt;
#undef RT_CUDA
#undef RT_CUBLAS
    return 0;

fail:
#undef RT_CUDA
#undef RT_CUBLAS
    world_cuda_runtime_destroy(rt);
    return 1;
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

#define STEP_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)
#define STEP_LINEAR(x, w, y, m, k, n) do { \
    if (row_major_linear(rt->handle, (x), (w), (y), (m), (k), (n))) return 1; \
} while (0)
#define STEP_LINEAR_FAST(x, w, wh, y, m, k, n) do { \
    if (use_fp16_gemm && (wh) && (m) > 1) { \
        if (row_major_linear_fp16_weight(rt->handle, (x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
    } else { \
        STEP_LINEAR((x), (w), (y), (m), (k), (n)); \
    } \
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
        patchify_im2row_f32_kernel<<<div_up_i64((int64_t)rt->T * patch_elems, 256), 256>>>(
            rt->d_latent, rt->d_patch_rows, rt->C, rt->H, rt->W, rt->ph, rt->pw, cfg->height, cfg->width);
        STEP_CUDA(cudaGetLastError());
        STEP_LINEAR(rt->d_patch_rows, rt->d_patch, rt->d_tokens, rt->T, patch_elems, rt->D);
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

            ada_rms_norm_single_f32_kernel<<<rt->T, 256>>>(d_tokens_cur, d_s0, d_b0, rt->d_norm, rt->T, rt->D, rt->rms_eps);
            STEP_CUDA(cudaGetLastError());
            STEP_LINEAR_FAST(rt->d_norm, lw->qkv_proj_weight, lw->qkv_proj_weight_h,
                             rt->d_qkv_raw, rt->T, rt->D, rt->D + 2 * rt->kv_dim);
            float *d_v_cur = (cfg->value_residual && layer_idx == 0) ? rt->d_v_first : rt->d_v;
            {
                dim3 grid(rt->T, cfg->n_heads + 2 * cfg->n_kv_heads);
                size_t smem = (size_t)(rt->d_head + 256) * sizeof(float);
                qkv_fused_rms_rope_f32_kernel<<<grid, 256, smem>>>(
                    rt->d_qkv_raw, rt->d_q, rt->d_k, d_v_cur,
                    rt->d_x_pos, rt->d_y_pos, rt->d_t_pos, rt->d_xy_table, rt->d_inv_t,
                    rt->T, cfg->n_heads, cfg->n_kv_heads, rt->d_head, cfg->width, cfg->height, rt->rms_eps);
            }
            STEP_CUDA(cudaGetLastError());
            if (cfg->value_residual && layer_idx != 0) {
                lerp_inplace_f32_kernel<<<div_up_i64((int64_t)rt->kv_rope_elems, 256), 256>>>(
                    rt->d_v, rt->d_v_first, lw->v_lamb, (int64_t)rt->kv_rope_elems);
                STEP_CUDA(cudaGetLastError());
            }

            int bucket = (current_frame_idx + (cache->pinned_dilation - 1)) / cache->pinned_dilation;
            int num_buckets = (cache->ring_length / rt->T) / cache->pinned_dilation;
            int base = (bucket % num_buckets) * rt->T;
            bool write_step = (current_frame_idx % cache->pinned_dilation) == 0;
            kv_cache_upsert_copy_f32_kernel<<<div_up_i64((int64_t)cfg->n_kv_heads * rt->T * rt->d_head, 256), 256>>>(
                cache->k, cache->v, rt->d_k, d_v_cur, cache->written,
                cfg->n_kv_heads, rt->T, rt->d_head, cache->ring_length, base, write_step, (bool)frozen_pass);
            STEP_CUDA(cudaGetLastError());
            collect_cache_frame_indices_kernel<<<cache->capacity / rt->T, 256>>>(
                cache->written, cache->indices, cache->index_count,
                cache->capacity, rt->T, base, write_step);
            STEP_CUDA(cudaGetLastError());
            if (rt->d_head == 64) {
                int cublas_attn_ok = 0;
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
                    cublas_attn_ok = 1;
                } else if (rt->attn_cublas_enabled && cache->pinned_dilation == 1) {
                    int n_tokens = visible_cache_tokens_pinned1_host(cache, rt->T, current_frame_idx);
                    if (n_tokens > 0 && n_tokens <= rt->attn_cublas_max_tokens) {
                        if (rt->attn_cublas_gqa_enabled) {
                            if (indexed_attention_cache_d64_cublas_gqa(rt, cache, n_tokens, 1.0f / 8.0f)) return 1;
                        } else {
                            if (indexed_attention_cache_d64_cublas(rt, cache, n_tokens, 1.0f / 8.0f)) return 1;
                        }
                        cublas_attn_ok = 1;
                    }
                }
                if (!cublas_attn_ok) {
                    indexed_attention_cache_d64_warp_f32_kernel<<<div_up_i64((int64_t)cfg->n_heads * rt->T, 4), 128>>>(
                        rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                        cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                        1.0f / 8.0f);
                }
            } else {
                indexed_attention_cache_f32_kernel<<<cfg->n_heads * rt->T, 256>>>(
                    rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                    cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity, rt->d_head,
                    1.0f / sqrtf((float)rt->d_head));
            }
            STEP_CUDA(cudaGetLastError());
            STEP_LINEAR_FAST(rt->d_attn, lw->out_proj_weight, lw->out_proj_weight_h,
                             rt->d_attn_out, rt->T, rt->D, rt->D);
            gated_residual_add_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                d_tokens_cur, rt->d_attn_out, d_g0, rt->d_tokens_after_attn, rt->T, rt->D);
            STEP_CUDA(cudaGetLastError());

            float *d_tokens_ctrl = rt->d_tokens_after_attn;
            if (lw->has_ctrl) {
                rms_norm_rows_f32_kernel<<<rt->T, 256>>>(rt->d_tokens_after_attn, rt->d_ctrl_norm, rt->T, rt->D, rt->rms_eps);
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
                d_tokens_ctrl = rt->d_tokens_after_ctrl;
            }

            ada_rms_norm_single_f32_kernel<<<rt->T, 256>>>(d_tokens_ctrl, d_s1, d_b1, rt->d_mlp_in, rt->T, rt->D, rt->rms_eps);
            STEP_CUDA(cudaGetLastError());
            STEP_LINEAR_FAST(rt->d_mlp_in, lw->dit_mlp_fc1_weight, lw->dit_mlp_fc1_weight_h,
                             rt->d_mlp_hidden, rt->T, rt->D, rt->mlp_hidden);
            silu_f32_kernel<<<div_up_i64((int64_t)rt->T * rt->mlp_hidden, 256), 256>>>(
                rt->d_mlp_hidden, rt->d_mlp_hidden, (int64_t)rt->T * rt->mlp_hidden);
            STEP_CUDA(cudaGetLastError());
            STEP_LINEAR_FAST(rt->d_mlp_hidden, lw->dit_mlp_fc2_weight, lw->dit_mlp_fc2_weight_h,
                             rt->d_mlp_out, rt->T, rt->mlp_hidden, rt->D);
            gated_residual_add_f32_kernel<<<div_up_i64((int64_t)rt->token_elems, 256), 256>>>(
                d_tokens_ctrl, rt->d_mlp_out, d_g1, d_tokens_next, rt->T, rt->D);
            STEP_CUDA(cudaGetLastError());

            float *d_swap = d_tokens_cur;
            d_tokens_cur = d_tokens_next;
            d_tokens_next = d_swap;
        }

        if (is_cache_pass) continue;

        float *d_out_mod = rt->d_out_mod_table + (int64_t)table_pass_idx * 2 * rt->D;
        out_norm_silu_f32_kernel<<<rt->T, 256>>>(d_tokens_cur, d_out_mod, rt->d_final_tokens, rt->T, rt->D, rt->rms_eps);
        STEP_CUDA(cudaGetLastError());
        unpatchify_orig_f32_kernel<<<rt->T * (rt->C * rt->ph * rt->pw), 256>>>(
            rt->d_final_tokens, rt->d_unpatch_w, rt->d_unpatch_b, rt->d_latent_out,
            rt->T, rt->D, rt->C, rt->H, rt->W, rt->ph, rt->pw, cfg->width, rt->C * rt->ph * rt->pw);
        STEP_CUDA(cudaGetLastError());
        latent_update_f32_kernel<<<div_up_i64((int64_t)rt->latent_elems, 256), 256>>>(
            rt->d_latent, rt->d_latent_out, dsigma, (int64_t)rt->latent_elems);
        STEP_CUDA(cudaGetLastError());
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
#undef STEP_CUDA
#undef STEP_LINEAR
#undef STEP_LINEAR_FAST
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

    cublasHandle_t handle = NULL;
    CUBLAS_OK(cublasCreate(&handle));
    if (row_major_linear(handle, d_tokens, d_q_weight, d_q, T, D, D)) return 1;
    CUBLAS_OK(cublasDestroy(handle));

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
    cublasHandle_t handle = NULL;

#define TRY_CUDA(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto cleanup_device; \
    } \
} while (0)
#define TRY_CUBLAS(expr) do { \
    cublasStatus_t _s = (expr); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        goto cleanup_device; \
    } \
} while (0)
#define TRY_LINEAR(x, w, y, m, k, n) do { \
    if (row_major_linear(handle, (x), (w), (y), (m), (k), (n))) goto cleanup_device; \
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

    TRY_CUBLAS(cublasCreate(&handle));

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
    if (handle) cublasDestroy(handle);
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
#undef TRY_CUBLAS
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
    cublasHandle_t handle = NULL;
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
#define TRY_CUBLAS2(expr) do { \
    cublasStatus_t _s = (expr); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        goto cleanup_device; \
    } \
} while (0)
#define TRY_LINEAR2(x, w, y, m, k, n) do { \
    if (row_major_linear(handle, (x), (w), (y), (m), (k), (n))) goto cleanup_device; \
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
    if (copy_world_layers_to_device(&d_layers, weights->layers, layers_to_run, D, kv_dim, mlp_hidden)) goto cleanup_device;
    if (alloc_device_world_caches(&d_caches, cfg, layers_to_run, T, cfg->n_kv_heads, d_head)) goto cleanup_device;

    TRY_CUDA2(cudaMemcpy(d_xy_table, h_xy, (size_t)d_xy * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA2(cudaMemcpy(d_inv_t, h_inv_t, (size_t)d_t * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUBLAS2(cublasCreate(&handle));
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
                    cache->written, cache->indices, cache->index_count,
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
    if (handle) cublasDestroy(handle);
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
#undef TRY_CUBLAS2
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
