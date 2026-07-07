#include "world_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <math.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

static int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__device__ __forceinline__ float wm_silu(float x) {
    return x / (1.0f + expf(-x));
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

__global__ static void kv_cache_mask_kernel(
        const bool *written,
        bool *mask_written,
        int capacity,
        int T,
        int base,
        bool write_step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    bool value = written[i];
    if (write_step && i >= base && i < base + T) value = false;
    mask_written[i] = value;
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

__global__ static void collect_cache_indices_kernel(
        const bool *mask_written,
        int64_t *indices,
        int *count,
        int capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int n = 0;
    for (int i = 0; i < capacity; ++i) {
        if (mask_written[i]) indices[n++] = (int64_t)i;
    }
    *count = n;
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

static int copy_f32_to_device(float **dst, const float *src, size_t n) {
    *dst = NULL;
    CUDA_OK(cudaMalloc((void **)dst, n * sizeof(float)));
    CUDA_OK(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

typedef struct {
    float *cond_bias;
    float *cond_proj_weight;
    float *qkv_proj_weight;
    float *out_proj_weight;
    float v_lamb;
    float *ctrl_fc1_x_weight;
    float *ctrl_fc1_c_weight;
    float *ctrl_fc2_weight;
    float *dit_mlp_fc1_weight;
    float *dit_mlp_fc2_weight;
    int has_ctrl;
} DeviceWorldLayerWeights;

typedef struct {
    float *k;
    float *v;
    bool *written;
    bool *mask_written;
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
        cudaFree(layers[i].out_proj_weight);
        cudaFree(layers[i].ctrl_fc1_x_weight);
        cudaFree(layers[i].ctrl_fc1_c_weight);
        cudaFree(layers[i].ctrl_fc2_weight);
        cudaFree(layers[i].dit_mlp_fc1_weight);
        cudaFree(layers[i].dit_mlp_fc2_weight);
    }
    free(layers);
}

static void free_device_world_caches(DeviceWorldLayerCache *caches, int n_layers) {
    if (!caches) return;
    for (int i = 0; i < n_layers; ++i) {
        cudaFree(caches[i].k);
        cudaFree(caches[i].v);
        cudaFree(caches[i].written);
        cudaFree(caches[i].mask_written);
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
        if (cudaMalloc((void **)&c->mask_written, (size_t)c->capacity * sizeof(bool)) != cudaSuccess) goto fail;
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
        if (copy_f32_to_device(&dl->out_proj_weight, src->out_proj_weight, (size_t)D * D)) goto fail;
        if (src->has_ctrl) {
            if (copy_f32_to_device(&dl->ctrl_fc1_x_weight, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_device(&dl->ctrl_fc1_c_weight, src->ctrl_fc1_c_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_device(&dl->ctrl_fc2_weight, src->ctrl_fc2_weight, (size_t)D * D)) goto fail;
        }
        if (copy_f32_to_device(&dl->dit_mlp_fc1_weight, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
        if (copy_f32_to_device(&dl->dit_mlp_fc2_weight, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
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
    int out_c;
    int in_c;
    int kernel;
    int has_bias;
} DeviceVaeConvWeight;

typedef struct {
    DeviceVaeConvWeight convs[WORLD_VAE_DECODER_CONV_COUNT];
    float *buf0;
    float *buf1;
    float *buf2;
    unsigned char *d_rgb;
    unsigned char *h_rgb;
    size_t max_elems;
    size_t rgb_elems;
    int out_w;
    int out_h;
    int H_pre_shuffle;
    int W_pre_shuffle;
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
        if (src->has_bias) {
            CUDA_OK(cudaMalloc((void **)&dst->bias, (size_t)src->out_c * sizeof(float)));
            CUDA_OK(cudaMemcpy(dst->bias, src->bias, (size_t)src->out_c * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    return 0;
}

static void taehv_free_weights(DeviceVaeConvWeight dev[WORLD_VAE_DECODER_CONV_COUNT]) {
    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
        cudaFree(dev[i].weight);
        cudaFree(dev[i].bias);
        dev[i].weight = NULL;
        dev[i].bias = NULL;
    }
}

static void taehv_decoder_free(DeviceVaeDecoder *dec) {
    if (!dec) return;
    taehv_free_weights(dec->convs);
    cudaFree(dec->buf0);
    cudaFree(dec->buf1);
    cudaFree(dec->buf2);
    cudaFree(dec->d_rgb);
    free(dec->h_rgb);
    memset(dec, 0, sizeof(*dec));
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
    dec->h_rgb = (unsigned char *)malloc(dec->rgb_elems);
    if (!dec->h_rgb) {
        fprintf(stderr, "failed to allocate VAE RGB host buffer\n");
        goto fail;
    }

    fprintf(stderr, "VAE decoder init: RGB %dx%d, scratch %.2f MiB x3\n",
            dec->out_w, dec->out_h, (double)(dec->max_elems * sizeof(float)) / (1024.0 * 1024.0));
#undef VAE_INIT_CUDA
    return 0;

fail:
#undef VAE_INIT_CUDA
    taehv_decoder_free(dec);
    return 1;
}

static int taehv_run_conv(const float *in, float *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
    int64_t total = (int64_t)N * conv->out_c * H * W;
    taehv_conv2d_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, conv->weight, conv->bias, out, N, conv->in_c, conv->out_c, H, W, conv->kernel, conv->has_bias);
    CUDA_OK(cudaGetLastError());
    return 0;
}

static int taehv_run_relu(float *x, int64_t n) {
    taehv_relu_kernel<<<div_up_i64(n, 256), 256>>>(x, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

static int taehv_run_memblock(
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
    if (taehv_run_conv(aux, tmp, conv0, N, H, W)) return 1;
    if (taehv_run_relu(tmp, elems)) return 1;
    if (taehv_run_conv(tmp, aux, conv2, N, H, W)) return 1;
    if (taehv_run_relu(aux, elems)) return 1;
    if (taehv_run_conv(aux, tmp, conv4, N, H, W)) return 1;
    taehv_add_relu_kernel<<<div_up_i64(elems, 256), 256>>>(cur, tmp, aux, elems);
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

static int world_cuda_decode_vae_to_ppm(
        const WorldConfig *cfg,
        DeviceVaeDecoder *dec,
        const float *d_latent,
        const char *out_path,
        int frame_offset) {
    if (!dec || !dec->buf0 || !out_path || !out_path[0]) return 0;

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

    fprintf(stderr, "VAE decode: latent [%d,%d,%d] -> RGB %dx%d PPM\n", C_latent, H0, W0, dec->out_w, dec->out_h);

    taehv_repeat_latent4_kernel<<<div_up_i64((int64_t)4 * C * H * W, 256), 256>>>(d_latent, cur, C, H, W);
    CUDA_OK(cudaGetLastError());
    taehv_clamp_kernel<<<div_up_i64((int64_t)N * C * H * W, 256), 256>>>(cur, (int64_t)N * C * H * W);
    CUDA_OK(cudaGetLastError());

#define VAE_CONV_TO(idx, out_c) do { \
    taehv_pick_scratch(cur, buf0, buf1, buf2, &tmp, &aux); \
    if (taehv_run_conv(cur, tmp, &dec->convs[(idx)], N, H, W)) goto cleanup; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_RELU() do { \
    if (taehv_run_relu(cur, (int64_t)N * C * H * W)) goto cleanup; \
} while (0)
#define VAE_MEMBLOCK(a, b, c) do { \
    if (taehv_run_memblock(&cur, buf0, buf1, buf2, &dec->convs[(a)], &dec->convs[(b)], &dec->convs[(c)], N, C, H, W)) goto cleanup; \
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
    if (taehv_run_conv(cur, tmp, &dec->convs[(idx)], N, H, W)) goto cleanup; \
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
    if (taehv_write_ppm_frames(out_path, dec->h_rgb, 4, dec->out_w, dec->out_h, frame_offset)) goto cleanup;
    rc = 0;

cleanup:
#undef VAE_CONV_TO
#undef VAE_RELU
#undef VAE_MEMBLOCK
#undef VAE_UPSAMPLE2
#undef VAE_TGROW
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
                kv_cache_mask_kernel<<<div_up_i64(cache->capacity, 256), 256>>>(
                    cache->written, cache->mask_written, cache->capacity, T, base, write_step);
                TRY_CUDA2(cudaGetLastError());
                kv_cache_upsert_copy_f32_kernel<<<div_up_i64((int64_t)cfg->n_kv_heads * T * d_head, 256), 256>>>(
                    cache->k, cache->v, d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, T, d_head, cache->ring_length, base, write_step, (bool)frozen_pass);
                TRY_CUDA2(cudaGetLastError());
                collect_cache_indices_kernel<<<1, 1>>>(
                    cache->mask_written, cache->indices, cache->index_count, cache->capacity);
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
