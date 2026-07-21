#include "world_cuda_vae.cuh"
#include "world_cuda_vae_ops.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_OK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)


static int copy_f32_to_half_device(__half **dst, const float *src, size_t n) {
    *dst = NULL;
    __half *host = (__half *)malloc(n * sizeof(__half));
    if (!host) {
        fprintf(stderr, "failed to allocate half conversion buffer\n");
        return 1;
    }
    for (size_t i = 0; i < n; ++i) host[i] = __float2half(src[i]);
    cudaError_t err = cudaMalloc((void **)dst, n * sizeof(__half));
    if (err == cudaSuccess) {
        err = cudaMemcpy(*dst, host, n * sizeof(__half), cudaMemcpyHostToDevice);
    }
    free(host);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error while copying FP16 weights: %s\n", cudaGetErrorString(err));
        cudaFree(*dst);
        *dst = NULL;
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

typedef WmCudaVaeConvDesc DeviceVaeConvWeight;

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
        cudaFree(dev[i].weight_krsc_h);
        cudaFree(dev[i].bias_h);
        dev[i].weight = NULL;
        dev[i].bias = NULL;
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

static int taehv_run_conv1x1_gemm_nchw(
        DeviceVaeDecoder *dec,
        const float *in,
        float *out,
        const DeviceVaeConvWeight *conv,
        int N,
        int H,
        int W) {
    int spatial = H * W;
    for (int n = 0; n < N; ++n) {
        const float *in_frame = in + (int64_t)n * conv->in_c * spatial;
        float *out_frame = out + (int64_t)n * conv->out_c * spatial;
        if (taehv_profile_begin(dec)) return 1;
        if (wm_cuda_vae_conv_gemm_f32(
                    conv->weight, in_frame, out_frame,
                    conv->out_c, spatial, conv->in_c, spatial)) return 1;
        if (taehv_profile_accum(dec, &dec->prof_1x1_gemm_ms)) return 1;
        if (dec && dec->profile_enabled) dec->prof_1x1_gemm_launches++;
    }

    if (conv->has_bias) {
        if (taehv_profile_begin(dec)) return 1;
        if (wm_cuda_vae_add_bias_nchw_f32(out, conv->bias, N, conv->out_c, H, W)) return 1;
        if (taehv_profile_accum(dec, &dec->prof_1x1_bias_ms)) return 1;
    }
    if (dec && dec->profile_enabled) dec->prof_1x1_calls++;
    return 0;
}

static int taehv_run_conv3x3_gemm_nchw(
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
    if (dec->conv3x3_batch_cols_enabled) {
        int total_cols = N * spatial;
        for (int tile_start = 0; tile_start < total_cols; tile_start += dec->conv3x3_tile_cols) {
            int tile_cols = total_cols - tile_start;
            if (tile_cols > dec->conv3x3_tile_cols) tile_cols = dec->conv3x3_tile_cols;
            if (taehv_profile_begin(dec)) return 1;
            if (wm_cuda_vae_im2col3x3_nchw_batch_tile_f32(
                        in, dec->conv3x3_cols, N, conv->in_c, H, W,
                        tile_start, tile_cols)) return 1;
            if (taehv_profile_accum(dec, &dec->prof_3x3_im2col_ms)) return 1;

            if (taehv_profile_begin(dec)) return 1;
            if (wm_cuda_vae_conv_gemm_f32(
                        conv->weight, dec->conv3x3_cols, dec->conv3x3_out_tile,
                        conv->out_c, tile_cols, k_elems, tile_cols)) return 1;
            if (taehv_profile_accum(dec, &dec->prof_3x3_gemm_ms)) return 1;

            if (taehv_profile_begin(dec)) return 1;
            if (wm_cuda_vae_scatter_conv_tile_nchw_f32(
                        dec->conv3x3_out_tile, out, N, conv->out_c, H, W,
                        tile_start, tile_cols)) return 1;
            if (taehv_profile_accum(dec, &dec->prof_3x3_scatter_ms)) return 1;
            if (dec->profile_enabled) dec->prof_3x3_tiles++;
        }
        if (conv->has_bias) {
            if (taehv_profile_begin(dec)) return 1;
            if (wm_cuda_vae_add_bias_nchw_f32(out, conv->bias, N, conv->out_c, H, W)) return 1;
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
            if (wm_cuda_vae_im2col3x3_nchw_tile_f32(
                        in, dec->conv3x3_cols, conv->in_c, H, W, n,
                        tile_start, tile_cols)) return 1;
            if (taehv_profile_accum(dec, &dec->prof_3x3_im2col_ms)) return 1;

            if (taehv_profile_begin(dec)) return 1;
            if (wm_cuda_vae_conv_gemm_f32(
                        conv->weight, dec->conv3x3_cols, out_frame + tile_start,
                        conv->out_c, tile_cols, k_elems, spatial)) return 1;
            if (taehv_profile_accum(dec, &dec->prof_3x3_gemm_ms)) return 1;
            if (dec->profile_enabled) dec->prof_3x3_tiles++;
        }
    }

    if (conv->has_bias) {
        if (taehv_profile_begin(dec)) return 1;
        if (wm_cuda_vae_add_bias_nchw_f32(out, conv->bias, N, conv->out_c, H, W)) return 1;
        if (taehv_profile_accum(dec, &dec->prof_3x3_bias_ms)) return 1;
    }
    if (dec->profile_enabled) dec->prof_3x3_calls++;
    return 0;
}

static int taehv_run_conv(DeviceVaeDecoder *dec, const float *in, float *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
    if (dec && dec->cutlass_1x1_enabled && conv->kernel == 1) {
        return taehv_run_conv1x1_gemm_nchw(dec, in, out, conv, N, H, W);
    }
    if (dec && dec->cutlass_3x3_enabled && conv->kernel == 3) {
        return taehv_run_conv3x3_gemm_nchw(dec, in, out, conv, N, H, W);
    }
    if (taehv_profile_begin(dec)) return 1;
    if (wm_cuda_vae_conv_direct_nchw_f32(in, out, conv, N, H, W)) return 1;
    if (taehv_profile_accum(dec, &dec->prof_direct_ms)) return 1;
    if (dec && dec->profile_enabled) dec->prof_direct_calls++;
    return 0;
}

static int taehv_run_conv_h_nhwc(DeviceVaeDecoder *dec, const __half *in, __half *out, const DeviceVaeConvWeight *conv, int N, int H, int W) {
    if (!in || !out || !conv || !conv->weight_krsc_h) return 1;
    if (conv->has_bias && !conv->bias_h) return 1;
    if (conv->kernel != 1 && conv->kernel != 3) return 1;
    if (taehv_profile_begin(dec)) {
        return 1;
    }
    if (wm_cuda_vae_conv_nhwc_f16(in, out, conv, N, H, W)) return 1;
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
        if (wm_cuda_vae_add_bias_nhwc_f16(out, conv->bias_h, total, conv->out_c)) return 1;
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
    return wm_cuda_vae_conv_stride2_nchw_f32(in, out, conv, N, H, W);
}

static int taehv_run_relu(float *x, int64_t n) {
    return wm_cuda_vae_relu_f32(x, n);
}

static int taehv_run_relu_h(__half *x, int64_t n) {
    return wm_cuda_vae_relu_f16(x, n);
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
    if (wm_cuda_vae_concat_memory_nchw_f32(cur, mem, aux, C, H, W)) return 1;
    CUDA_OK(cudaMemcpy(mem, cur, elems * sizeof(float), cudaMemcpyDeviceToDevice));
    if (taehv_run_conv(dec, aux, tmp, conv0, 1, H, W)) return 1;
    if (taehv_run_relu(tmp, elems)) return 1;
    if (taehv_run_conv(dec, tmp, aux, conv2, 1, H, W)) return 1;
    if (taehv_run_relu(aux, elems)) return 1;
    if (taehv_run_conv(dec, aux, tmp, conv4, 1, H, W)) return 1;
    if (wm_cuda_vae_add_relu_f32(cur, tmp, aux, elems)) return 1;
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
    if (wm_cuda_vae_concat_memory_nhwc_f16(cur, mem, aux, C, H, W)) return 1;
    CUDA_OK(cudaMemcpy(mem, cur, elems * sizeof(__half), cudaMemcpyDeviceToDevice));
    if (taehv_run_conv_h_nhwc(dec, aux, tmp, conv0, 1, H, W)) return 1;
    if (taehv_run_relu_h(tmp, elems)) return 1;
    if (taehv_run_conv_h_nhwc(dec, tmp, aux, conv2, 1, H, W)) return 1;
    if (taehv_run_relu_h(aux, elems)) return 1;
    if (taehv_run_conv_h_nhwc(dec, aux, tmp, conv4, 1, H, W)) return 1;
    if (wm_cuda_vae_add_relu_f16(cur, tmp, aux, elems)) return 1;
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
    if (wm_cuda_vae_concat_past_nchw_f32(cur, aux, N, C, H, W)) return 1;
    if (taehv_run_conv(dec, aux, tmp, conv0, N, H, W)) return 1;
    if (taehv_run_relu(tmp, elems)) return 1;
    if (taehv_run_conv(dec, tmp, aux, conv2, N, H, W)) return 1;
    if (taehv_run_relu(aux, elems)) return 1;
    if (taehv_run_conv(dec, aux, tmp, conv4, N, H, W)) return 1;
    if (wm_cuda_vae_add_relu_f32(cur, tmp, aux, elems)) return 1;
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

    if (wm_cuda_vae_copy_latent_clamp_f32(d_latent, cur, (int64_t)C * H * W)) return 1;

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
    if (wm_cuda_vae_upsample2_nchw_f32(cur, tmp, 1, C, H, W)) return 1; \
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
    if (wm_cuda_vae_tgrow_reshape_nchw_f32(
                tmp, dec->stream_branch0, 1, C, H, W, 2)) return 1;

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
        if (wm_cuda_vae_tgrow_reshape_nchw_f32(
                    tmp, dec->stream_branch1, 1, C, H, W, 2)) return 1;

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
                if (wm_cuda_vae_pixel_shuffle_u8_nchw_f32(cur, frame_rgb, H, W)) return 1;
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

    if (wm_cuda_vae_copy_latent_clamp_nhwc_f16(d_latent, cur, C, H, W)) return 1;

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
    if (wm_cuda_vae_upsample2_nhwc_f16(cur, tmp, 1, C, H, W)) return 1; \
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
    if (wm_cuda_vae_tgrow_reshape_nhwc_f16(
                tmp, dec->hstream_branch0, 1, C, H, W, 2)) return 1;

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
        if (wm_cuda_vae_tgrow_reshape_nhwc_f16(
                    tmp, dec->hstream_branch1, 1, C, H, W, 2)) return 1;

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
                if (wm_cuda_vae_pixel_shuffle_u8_nhwc_f16(cur, frame_rgb, H, W)) return 1;
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

struct WorldCudaVae {
    WorldConfig cfg;
    DeviceVaeDecoder decoder;
};

int wm_cuda_vae_create(
        WorldCudaVae **out,
        const WorldConfig *cfg,
        const WorldVaeDecoderWeights *decoder) {
    if (!out || !cfg) return 1;
    *out = NULL;
    WorldCudaVae *vae = (WorldCudaVae *)calloc(1, sizeof(*vae));
    if (!vae) return 1;
    vae->cfg = *cfg;
    if (taehv_decoder_init(&vae->decoder, cfg, decoder)) {
        free(vae);
        return 1;
    }
    *out = vae;
    return 0;
}

void wm_cuda_vae_destroy(WorldCudaVae *vae) {
    if (!vae) return;
    taehv_decoder_free(&vae->decoder);
    free(vae);
}

int wm_cuda_vae_reset(WorldCudaVae *vae) {
    if (!vae) return 1;
    for (int i = 0; i < WORLD_VAE_STREAM_MEM_COUNT; ++i) {
        size_t elems = taehv_stream_mem_elems(&vae->cfg, i);
        if (vae->decoder.stream_mem[i]) {
            CUDA_OK(cudaMemset(vae->decoder.stream_mem[i], 0, elems * sizeof(float)));
        }
        if (vae->decoder.hstream_mem[i]) {
            CUDA_OK(cudaMemset(vae->decoder.hstream_mem[i], 0, elems * sizeof(__half)));
        }
    }
    vae->decoder.stream_started_f32 = 0;
    vae->decoder.stream_started_h = 0;
    return 0;
}

int wm_cuda_vae_init_encoder(
        WorldCudaVae *vae,
        const WorldVaeEncoderWeights *encoder) {
    if (!vae || !encoder || !vae->decoder.buf0) return 1;
    vae->decoder.encoder_enabled = 0;
    taehv_free_weights(vae->decoder.encoder_convs, WORLD_VAE_ENCODER_CONV_COUNT);
    if (taehv_copy_weights(
                vae->decoder.encoder_convs,
                encoder->convs,
                WORLD_VAE_ENCODER_CONV_COUNT,
                0)) {
        taehv_free_weights(vae->decoder.encoder_convs, WORLD_VAE_ENCODER_CONV_COUNT);
        return 1;
    }
    vae->decoder.encoder_enabled = 1;
    return 0;
}

int wm_cuda_vae_encode_rgb(
        WorldCudaVae *vae,
        const float *rgb,
        int width,
        int height,
        float *latent_out) {
    if (!vae) return 1;
    return taehv_encode_image_rgb(
        &vae->cfg, &vae->decoder, rgb, width, height, latent_out);
}

int wm_cuda_vae_decode_rgb(
        WorldCudaVae *vae,
        const float *device_latent,
        const unsigned char **rgb_out,
        int *frame_count_out,
        int *width_out,
        int *height_out) {
    if (!vae) return 1;
    return world_cuda_decode_vae_to_rgb(
        &vae->cfg,
        &vae->decoder,
        device_latent,
        rgb_out,
        frame_count_out,
        width_out,
        height_out);
}
