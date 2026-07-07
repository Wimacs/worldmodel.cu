#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#define WM_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define WM_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define WM_CHECK_F32(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")

static int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

static void check_last_cuda_error(const char *name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu", &silu_cuda, "WorldModel SiLU (CUDA, f32)");
    m.def("rms_norm", &rms_norm_cuda, "WorldModel RMSNorm (CUDA, f32)", py::arg("x"), py::arg("eps") = 1.0e-6);
    m.def("ada_rms_norm", &ada_rms_norm_cuda, "WorldModel AdaRMSNorm (CUDA, f32)", py::arg("x"), py::arg("scale"), py::arg("bias"), py::arg("eps") = 1.0e-6);
    m.def("ortho_rope", &ortho_rope_cuda, "WorldModel OrthoRoPE (CUDA, f32)");
    m.def("qkv_rms_rope", &qkv_rms_rope_cuda, "WorldModel fused QKV split + RMSNorm + OrthoRoPE (CUDA, f32)", py::arg("qkv"), py::arg("x_pos"), py::arg("y_pos"), py::arg("t_pos"), py::arg("xy"), py::arg("inv_t"), py::arg("n_heads"), py::arg("n_kv_heads"), py::arg("width"), py::arg("height"), py::arg("eps") = 1.0e-6);
    m.def("masked_attention", &masked_attention_cuda, "WorldModel written-mask GQA attention (CUDA, f32)", py::arg("q"), py::arg("k"), py::arg("v"), py::arg("written"), py::arg("scale"));
    m.def("indexed_attention", &indexed_attention_cuda, "WorldModel indexed GQA attention (CUDA, f32)", py::arg("q"), py::arg("k"), py::arg("v"), py::arg("indices"), py::arg("scale"));
    m.def("kv_cache_upsert", &kv_cache_upsert_cuda, "WorldModel KV ring-cache upsert (CUDA, f32)", py::arg("cache_k"), py::arg("cache_v"), py::arg("written"), py::arg("k"), py::arg("v"), py::arg("frame_idx"), py::arg("ring_length"), py::arg("pinned_dilation"), py::arg("frozen"));
    m.def("patchify", &patchify_cuda, "WorldModel patchify Conv2d + token layout (CUDA, f32)");
    m.def("unpatchify", &unpatchify_cuda, "WorldModel unpatchify Linear + image layout (CUDA, f32)", py::arg("tokens"), py::arg("weight"), py::arg("bias"), py::arg("channels"), py::arg("height"), py::arg("width"), py::arg("patch_h"), py::arg("patch_w"));
}
