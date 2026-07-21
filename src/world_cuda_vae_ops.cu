#include "world_cuda_vae_ops.cuh"

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

#include <stdio.h>

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

int wm_cuda_vae_copy_latent_clamp_f32(
        const float *latent, float *out, int64_t n) {
    if (!latent || !out || n <= 0) return 1;
    taehv_copy_latent_clamp_kernel<<<div_up_i64(n, 256), 256>>>(latent, out, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_copy_latent_clamp_nhwc_f16(
        const float *latent, __half *out, int channels, int height, int width) {
    if (!latent || !out || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)channels * height * width;
    taehv_copy_latent_clamp_nhwc_h_kernel<<<div_up_i64(total, 256), 256>>>(
        latent, out, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_relu_f32(float *x, int64_t n) {
    if (!x || n <= 0) return 1;
    taehv_relu_kernel<<<div_up_i64(n, 256), 256>>>(x, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_relu_f16(__half *x, int64_t n) {
    if (!x || n <= 0) return 1;
    taehv_relu_nhwc_h_kernel<<<div_up_i64(n, 256), 256>>>(x, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_add_relu_f32(
        const float *a, const float *b, float *out, int64_t n) {
    if (!a || !b || !out || n <= 0) return 1;
    taehv_add_relu_kernel<<<div_up_i64(n, 256), 256>>>(a, b, out, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_add_relu_f16(
        const __half *a, const __half *b, __half *out, int64_t n) {
    if (!a || !b || !out || n <= 0) return 1;
    taehv_add_relu_nhwc_h_kernel<<<div_up_i64(n, 256), 256>>>(a, b, out, n);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_conv_direct_nchw_f32(
        const float *in,
        float *out,
        const WmCudaVaeConvDesc *conv,
        int batch,
        int height,
        int width) {
    if (!in || !out || !conv || !conv->weight || batch <= 0 || height <= 0 || width <= 0 ||
            conv->in_c <= 0 || conv->out_c <= 0 || conv->kernel <= 0 ||
            (conv->has_bias && !conv->bias)) return 1;
    int64_t total = (int64_t)batch * conv->out_c * height * width;
    taehv_conv2d_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, conv->weight, conv->bias, out,
        batch, conv->in_c, conv->out_c, height, width, conv->kernel, conv->has_bias);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_conv_stride2_nchw_f32(
        const float *in,
        float *out,
        const WmCudaVaeConvDesc *conv,
        int batch,
        int height,
        int width) {
    if (!in || !out || !conv || !conv->weight || conv->kernel != 3 ||
            batch <= 0 || height <= 0 || width <= 0 ||
            (conv->has_bias && !conv->bias)) return 1;
    int out_h = (height + 1) / 2;
    int out_w = (width + 1) / 2;
    int64_t total = (int64_t)batch * conv->out_c * out_h * out_w;
    taehv_conv2d_stride2_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, conv->weight, conv->bias, out,
        batch, conv->in_c, conv->out_c, height, width, conv->kernel, conv->has_bias);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_conv_gemm_f32(
        const float *weight,
        const float *cols,
        float *out,
        int out_channels,
        int column_count,
        int reduction_size,
        int out_stride) {
    if (!weight || !cols || !out || out_channels <= 0 || column_count <= 0 ||
            reduction_size <= 0 || out_stride < column_count) {
        return 1;
    }
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
        2>;
    typename Gemm::Arguments args(
        {out_channels, column_count, reduction_size},
        {weight, reduction_size},
        {cols, column_count},
        {out, out_stride},
        {out, out_stride},
        {1.0f, 0.0f});
    Gemm gemm;
    CUTLASS_OK(gemm(args));
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_conv_nhwc_f16(
        const __half *in,
        __half *out,
        const WmCudaVaeConvDesc *conv,
        int batch,
        int height,
        int width) {
    if (!in || !out || !conv || !conv->weight_krsc_h || batch <= 0 || height <= 0 || width <= 0 ||
            conv->in_c <= 0 || conv->out_c <= 0 ||
            (conv->kernel != 1 && conv->kernel != 3)) return 1;

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

    int kernel = conv->kernel;
    int pad = kernel / 2;
    cutlass::Tensor4DCoord input_size(batch, height, width, conv->in_c);
    cutlass::Tensor4DCoord filter_size(conv->out_c, kernel, kernel, conv->in_c);
    cutlass::Tensor4DCoord padding(pad, pad, pad, pad);
    cutlass::MatrixCoord stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(batch, height, width, conv->out_c);
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
    if (workspace_size > 0) CUDA_OK(cudaMalloc(&workspace, workspace_size));
    cutlass::Status status = implicit_gemm(args, workspace, 0);
    cudaFree(workspace);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS error %s:%d: %s\n", __FILE__, __LINE__, cutlassGetStatusString(status));
        return 1;
    }
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_add_bias_nchw_f32(
        float *out, const float *bias, int batch, int channels, int height, int width) {
    if (!out || !bias || batch <= 0 || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)batch * channels * height * width;
    taehv_add_bias_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        out, bias, batch, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_add_bias_nhwc_f16(
        __half *out, const __half *bias, int64_t n, int channels) {
    if (!out || !bias || n <= 0 || channels <= 0) return 1;
    taehv_add_bias_nhwc_h_kernel<<<div_up_i64(n, 256), 256>>>(out, bias, n, channels);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_im2col3x3_nchw_tile_f32(
        const float *in,
        float *cols,
        int channels,
        int height,
        int width,
        int frame,
        int tile_start,
        int tile_cols) {
    if (!in || !cols || channels <= 0 || height <= 0 || width <= 0 ||
            frame < 0 || tile_start < 0 || tile_cols <= 0) return 1;
    int64_t total = (int64_t)channels * 9 * tile_cols;
    taehv_im2col3x3_nchw_tile_kernel<<<div_up_i64(total, 256), 256>>>(
        in, cols, channels, height, width, frame, tile_start, tile_cols);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_im2col3x3_nchw_batch_tile_f32(
        const float *in,
        float *cols,
        int batch,
        int channels,
        int height,
        int width,
        int tile_start,
        int tile_cols) {
    if (!in || !cols || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
            tile_start < 0 || tile_cols <= 0) return 1;
    int64_t total = (int64_t)channels * 9 * tile_cols;
    taehv_im2col3x3_nchw_batch_tile_kernel<<<div_up_i64(total, 256), 256>>>(
        in, cols, batch, channels, height, width, tile_start, tile_cols);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_scatter_conv_tile_nchw_f32(
        const float *tile,
        float *out,
        int batch,
        int channels,
        int height,
        int width,
        int tile_start,
        int tile_cols) {
    if (!tile || !out || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
            tile_start < 0 || tile_cols <= 0) return 1;
    int64_t total = (int64_t)channels * tile_cols;
    taehv_scatter_conv_tile_to_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        tile, out, batch, channels, height, width, tile_start, tile_cols);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_concat_memory_nchw_f32(
        const float *cur, const float *mem, float *out, int channels, int height, int width) {
    if (!cur || !mem || !out || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)2 * channels * height * width;
    taehv_concat_memory_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        cur, mem, out, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_concat_memory_nhwc_f16(
        const __half *cur, const __half *mem, __half *out, int channels, int height, int width) {
    if (!cur || !mem || !out || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)2 * channels * height * width;
    taehv_concat_memory_nhwc_h_kernel<<<div_up_i64(total, 256), 256>>>(
        cur, mem, out, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_concat_past_nchw_f32(
        const float *in, float *out, int batch, int channels, int height, int width) {
    if (!in || !out || batch <= 0 || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)batch * 2 * channels * height * width;
    taehv_concat_past_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, out, batch, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_upsample2_nchw_f32(
        const float *in, float *out, int batch, int channels, int height, int width) {
    if (!in || !out || batch <= 0 || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)batch * channels * (height * 2) * (width * 2);
    taehv_upsample2_nchw_kernel<<<div_up_i64(total, 256), 256>>>(
        in, out, batch, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_upsample2_nhwc_f16(
        const __half *in, __half *out, int batch, int channels, int height, int width) {
    if (!in || !out || batch <= 0 || channels <= 0 || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)batch * (height * 2) * (width * 2) * channels;
    taehv_upsample2_nhwc_h_kernel<<<div_up_i64(total, 256), 256>>>(
        in, out, batch, channels, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_tgrow_reshape_nchw_f32(
        const float *in,
        float *out,
        int batch,
        int channels,
        int height,
        int width,
        int stride) {
    if (!in || !out || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || stride <= 0) {
        return 1;
    }
    int64_t total = (int64_t)batch * stride * channels * height * width;
    taehv_tgrow_reshape_kernel<<<div_up_i64(total, 256), 256>>>(
        in, out, batch, channels, height, width, stride);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_tgrow_reshape_nhwc_f16(
        const __half *in,
        __half *out,
        int batch,
        int channels,
        int height,
        int width,
        int stride) {
    if (!in || !out || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || stride <= 0) {
        return 1;
    }
    int64_t total = (int64_t)batch * stride * height * width * channels;
    taehv_tgrow_reshape_nhwc_h_kernel<<<div_up_i64(total, 256), 256>>>(
        in, out, batch, channels, height, width, stride);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_pixel_shuffle_u8_nchw_f32(
        const float *in, unsigned char *rgb, int height, int width) {
    if (!in || !rgb || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)(height * 2) * (width * 2) * 3;
    taehv_pixel_shuffle_one_u8_kernel<<<div_up_i64(total, 256), 256>>>(
        in, rgb, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}

int wm_cuda_vae_pixel_shuffle_u8_nhwc_f16(
        const __half *in, unsigned char *rgb, int height, int width) {
    if (!in || !rgb || height <= 0 || width <= 0) return 1;
    int64_t total = (int64_t)(height * 2) * (width * 2) * 3;
    taehv_pixel_shuffle_one_u8_nhwc_h_kernel<<<div_up_i64(total, 256), 256>>>(
        in, rgb, height, width);
    CUDA_OK(cudaGetLastError());
    return 0;
}
