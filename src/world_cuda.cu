#include "world_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static uint32_t lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static void fill_latent(float *x, int n, unsigned int seed) {
    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < n; ++i) {
        uint32_t r = lcg_next(&s);
        float u = (float)((r >> 8) & 0x00FFFFFFu) / 16777216.0f;
        x[i] = 2.0f * u - 1.0f;
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
    fill_latent(h_latent, (int)latent_elems, seed);

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
