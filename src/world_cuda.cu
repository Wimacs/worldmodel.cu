#include "world_cuda.h"
#include "world_cuda_internal.cuh"
#include "world_cuda_ops.cuh"
#include "world_cuda_vae.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>


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

void wm_cuda_fill_latent(float *x, int n, unsigned int seed, int noise_mode) {
    uint32_t s = seed ? seed : 1u;
    if (noise_mode == WORLD_NOISE_NORMAL) {
        fill_latent_normal(x, n, &s);
    } else {
        fill_latent_uniform(x, n, &s);
    }
}

int wm_cuda_copy_f32_to_device(float **dst, const float *src, size_t n) {
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

void wm_cuda_free_device_world_layers(DeviceWorldLayerWeights *layers, int n_layers) {
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

void wm_cuda_free_device_world_caches(DeviceWorldLayerCache *caches, int n_layers) {
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

int wm_cuda_alloc_device_world_caches(
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
        if (wm_cuda_init_cache_written(c->written, c->ring_length, T)) goto fail;
        if (cudaGetLastError() != cudaSuccess) goto fail;
    }

    *dst_caches = caches;
    return 0;

fail:
    wm_cuda_free_device_world_caches(caches, n_layers);
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
    int rc = wm_cuda_copy_f32_to_device(dst, host, total);
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
    int rc = wm_cuda_copy_f32_to_device(dst, host, total);
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

int wm_cuda_copy_world_layers_to_device(
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
        if (wm_cuda_copy_f32_to_device(&dl->cond_bias, src->cond_bias, (size_t)D)) goto fail;
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
            if (wm_cuda_copy_f32_to_device(&dl->out_proj_weight, src->out_proj_weight, (size_t)D * D)) goto fail;
            if (copy_f32_to_half_device(&dl->out_proj_weight_h, src->out_proj_weight, (size_t)D * D)) goto fail;
        }
        if (src->has_ctrl) {
            if (wm_cuda_copy_f32_to_device(&dl->ctrl_fc1_c_weight, src->ctrl_fc1_c_weight, (size_t)D * D)) goto fail;
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
                if (wm_cuda_copy_f32_to_device(&dl->ctrl_fc1_x_weight, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
                if (copy_f32_to_half_device(&dl->ctrl_fc1_x_weight_h, src->ctrl_fc1_x_weight, (size_t)D * D)) goto fail;
                if (wm_cuda_copy_f32_to_device(&dl->ctrl_fc2_weight, src->ctrl_fc2_weight, (size_t)D * D)) goto fail;
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
            if (wm_cuda_copy_f32_to_device(&dl->dit_mlp_fc1_weight, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
            if (copy_f32_to_half_device(&dl->dit_mlp_fc1_weight_h, src->dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto fail;
            if (wm_cuda_copy_f32_to_device(&dl->dit_mlp_fc2_weight, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
            if (copy_f32_to_half_device(&dl->dit_mlp_fc2_weight_h, src->dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto fail;
        }
    }

    *dst_layers = dst;
    return 0;

fail:
    wm_cuda_free_device_world_layers(dst, n_layers);
    return 1;
}

void wm_cuda_fill_noise_embedding(float *emb, float sigma) {
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

void wm_cuda_fill_positions(int64_t *x_pos, int64_t *y_pos, int64_t *t_pos, int T, int width, int frame_timestamp) {
    for (int i = 0; i < T; ++i) {
        y_pos[i] = i / width;
        x_pos[i] = i - (int)y_pos[i] * width;
        t_pos[i] = frame_timestamp;
    }
}

void wm_cuda_fill_rope_tables(float *xy, float *inv_t, int d_head, int height, int width) {
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
    WorldCudaVae *vae;
};

static int precompute_runtime_layer_mods(WorldCudaRuntime *rt) {
    if (!rt || !rt->d_layer_mod_table || !rt->d_out_mod_table) return 1;
    const WorldConfig *cfg = &rt->cfg;
    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= rt->steps_to_run;
        float sigma_step = is_cache_pass ? 0.0f : cfg->scheduler_sigmas[pass_idx];

        wm_cuda_fill_noise_embedding(rt->h_noise, sigma_step);
        CUDA_OK(cudaMemcpy(rt->d_noise, rt->h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
        if (wm_cuda_linear_f32(rt->d_noise, rt->d_denoise_fc1, rt->d_noise_hidden, 1, 512, rt->mlp_hidden)) return 1;
        if (wm_cuda_silu_f32(rt->d_noise_hidden, rt->d_noise_hidden, rt->mlp_hidden)) return 1;
        CUDA_OK(cudaGetLastError());
        if (wm_cuda_linear_f32(rt->d_noise_hidden, rt->d_denoise_fc2, rt->d_cond, 1, rt->mlp_hidden, rt->D)) return 1;
        if (wm_cuda_silu_f32(rt->d_cond, rt->d_cond_act, rt->D)) return 1;
        CUDA_OK(cudaGetLastError());
        if (wm_cuda_linear_f32(
                    rt->d_cond_act,
                    rt->d_out_norm_w,
                    rt->d_out_mod_table + (int64_t)pass_idx * 2 * rt->D,
                    1,
                    rt->D,
                    2 * rt->D)) return 1;

        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            const DeviceWorldLayerWeights *lw = &rt->d_layers[layer_idx];
            float *dst = rt->d_layer_mod_table + ((int64_t)pass_idx * rt->layers_to_run + layer_idx) * (6 * rt->D);
            if (wm_cuda_add_bias_silu_f32(rt->d_cond, lw->cond_bias, rt->d_cond_act, rt->D)) return 1;
            CUDA_OK(cudaGetLastError());
            if (wm_cuda_linear_f32(rt->d_cond_act, lw->cond_proj_weight, dst, 1, rt->D, 6 * rt->D)) return 1;
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
            if (wm_cuda_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->qkv_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D + 2 * rt->kv_dim)) return 1;
            checked_qkv = 1;
        }
        if (!checked_out && lw->out_proj_weight_i8) {
            if (wm_cuda_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->out_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            checked_out = 1;
        }
        if (!checked_ctrl && lw->ctrl_fc1_x_weight_i8 && lw->ctrl_fc2_weight_i8) {
            if (wm_cuda_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->ctrl_fc1_x_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            if (wm_cuda_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->ctrl_fc2_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            checked_ctrl = 1;
        }
        if (!checked_mlp && lw->dit_mlp_fc1_weight_i8 && lw->dit_mlp_fc2_weight_i8) {
            if (wm_cuda_gemm_i8_i32_can_implement(
                    rt->d_w8a8_x,
                    lw->dit_mlp_fc1_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->mlp_hidden)) return 1;
            if (wm_cuda_gemm_i8_i32_can_implement(
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
    wm_cuda_vae_destroy(rt->vae);
    if (rt->prof_stop) cudaEventDestroy(rt->prof_stop);
    if (rt->prof_start) cudaEventDestroy(rt->prof_start);
    if (rt->ev_after_vae) cudaEventDestroy(rt->ev_after_vae);
    if (rt->ev_after_transformer) cudaEventDestroy(rt->ev_after_transformer);
    if (rt->ev_after_setup) cudaEventDestroy(rt->ev_after_setup);
    if (rt->ev_step_start) cudaEventDestroy(rt->ev_step_start);
    wm_cuda_free_device_world_layers(rt->d_layers, rt->layers_to_run);
    wm_cuda_free_device_world_caches(rt->d_caches, rt->layers_to_run);
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
        const WorldModelWeights *weights,
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
    wm_cuda_fill_rope_tables(rt->h_xy, rt->h_inv_t, rt->d_head, cfg->height, cfg->width);

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
            rt->splitk_workspace_bytes = wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel_workspace_size(
                rt->T, rt->mlp_hidden, rt->D, rt->mlp_fc2_splitk_slices);
        } else {
            rt->splitk_workspace_bytes = wm_cuda_linear_fp16_weight_tensorop_splitk_workspace_size(
                rt->T, rt->mlp_hidden, rt->D, rt->mlp_fc2_splitk_slices);
        }
        if (rt->splitk_workspace_bytes > 0) {
            RT_CUDA(cudaMalloc(&rt->d_splitk_workspace, rt->splitk_workspace_bytes));
        }
        fprintf(stderr,
                "MLP fc2 split-K workspace: %.2f MiB\n",
                (double)rt->splitk_workspace_bytes / (1024.0 * 1024.0));
    }

    if (wm_cuda_copy_f32_to_device(&rt->d_patch, weights->patchify_weight, patch_weight_elems)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_denoise_fc1, weights->denoise_fc1_weight, (size_t)rt->mlp_hidden * 512)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_denoise_fc2, weights->denoise_fc2_weight, (size_t)rt->D * rt->mlp_hidden)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)rt->mlp_hidden * rt->ctrl_dim)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)rt->D * rt->mlp_hidden)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_out_norm_w, weights->out_norm_fc_weight, out_norm_weight_elems)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_unpatch_w, weights->unpatchify_weight, unpatch_weight_elems)) goto fail;
    if (wm_cuda_copy_f32_to_device(&rt->d_unpatch_b, weights->unpatchify_bias, (size_t)rt->C)) goto fail;
    if (wm_cuda_copy_world_layers_to_device(
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
        if (!wm_cuda_has_cutlass_fmha() &&
                (rt->attn_cutlass_fmha_enabled || rt->attn_sparse_fmha_enabled)) {
            fprintf(stderr, "CUTLASS FMHA options ignored because the example headers are unavailable\n");
            rt->attn_cutlass_fmha_enabled = 0;
            rt->attn_sparse_fmha_enabled = 0;
        }
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
    if (wm_cuda_alloc_device_world_caches(&rt->d_caches, cfg, layers_to_run, rt->T, cfg->n_kv_heads, rt->d_head, rt->attn_half_cache_enabled)) goto fail;
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
    if (wm_cuda_vae_create(&rt->vae, cfg, vae)) goto fail;
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
    if (!rt || !encoder) return 1;
    CUDA_OK(cudaDeviceSynchronize());
    if (wm_cuda_vae_init_encoder(rt->vae, encoder)) return 1;
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
    if (wm_cuda_vae_encode_rgb(rt->vae, rgb, width, height, latent_out)) return 1;
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
        if (wm_cuda_init_cache_written(
            cache->written, cache->ring_length, rt->T)) return 1;
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemset(cache->index_count, 0, sizeof(int)));
        CUDA_OK(cudaMemset(cache->indices, 0, (size_t)cache->capacity * sizeof(int64_t)));
        CUDA_OK(cudaMemset(cache->block_ids, 0,
                    (size_t)div_up_i64(cache->capacity, 128) * sizeof(int32_t)));
        memset(cache->h_slot_written, 0, (size_t)cache->slot_count);
        cache->h_slot_written[cache->slot_count - 1] = 1;
    }
    if (wm_cuda_vae_reset(rt->vae)) return 1;
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
    if (wm_cuda_linear_f32((x), (w), (y), (m), (k), (n))) return 1; \
} while (0)
#define STEP_LINEAR_FAST(x, w, wh, y, m, k, n) do { \
    if (use_fp16_gemm && (wh) && (m) > 1) { \
        if (use_fp16_tensorop) { \
            if (wm_cuda_should_use_m64n64_tensorop(rt->fp16_gemm_m64n64_enabled, (m), (k), (n))) { \
                if (wm_cuda_linear_fp16_weight_tensorop_m64n64((x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
            } else { \
                if (wm_cuda_linear_fp16_weight_tensorop((x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
            } \
        } else { \
            if (wm_cuda_linear_fp16_weight_simt((x), rt->d_linear_half, (wh), (y), (m), (k), (n))) return 1; \
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
    if (wm_cuda_silu_f32(rt->d_ctrl_emb_hidden, rt->d_ctrl_emb_hidden, rt->mlp_hidden)) return 1;
    STEP_CUDA(cudaGetLastError());
    STEP_LINEAR(rt->d_ctrl_emb_hidden, rt->d_ctrl_emb_fc2_w, rt->d_ctrl_emb, 1, rt->mlp_hidden, rt->D);
    if (wm_cuda_rms_norm_rows_f32(rt->d_ctrl_emb, rt->d_ctrl_emb_norm, 1, rt->D, rt->rms_eps)) return 1;
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
        wm_cuda_fill_latent(rt->h_latent, (int)rt->latent_elems, rt->seed + (unsigned int)rt->frame_ordinal, rt->noise_mode);
    }
    wm_cuda_fill_positions(rt->h_x_pos, rt->h_y_pos, rt->h_t_pos, rt->T, cfg->width, frame_timestamp);
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
        if (wm_cuda_patchify_im2row_f32(
            rt->d_latent, rt->d_patch_rows, rt->C, rt->H, rt->W, rt->ph, rt->pw, cfg->height, cfg->width)) return 1;
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
                wm_cuda_should_use_m64n64_tensorop(
                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->D + 2 * rt->kv_dim);
            STEP_PROFILE_BEGIN();
            if (qkv_w8a8) {
                if (wm_cuda_rms_norm_quantize_rows_i8(
                    d_tokens_cur,
                    d_s0,
                    d_b0,
                    rt->d_w8a8_x,
                    rt->d_w8a8_x_scales,
                    rt->T,
                    rt->D,
                    rt->rms_eps)) return 1;
            } else if (qkv_half_boundary) {
                if (wm_cuda_ada_rms_norm_f16(
                    d_tokens_cur, d_s0, d_b0, rt->d_linear_half, rt->T, rt->D, rt->rms_eps)) return 1;
            } else {
                if (wm_cuda_ada_rms_norm_f32(
                    d_tokens_cur, d_s0, d_b0, rt->d_norm, rt->T, rt->D, rt->rms_eps)) return 1;
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_norm_ms, prof_norm_calls);
            STEP_PROFILE_BEGIN();
            if (qkv_w8a8) {
                if (wm_cuda_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->qkv_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D + 2 * rt->kv_dim)) return 1;
            } else if (qkv_half_boundary) {
                if (wm_cuda_linear_fp16_input_weight_tensorop_m64n64(
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
                if (qkv_w8a8) {
                    if (wm_cuda_qkv_fused_rms_rope_i32_dequant(
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
                        rt->rms_eps)) return 1;
                } else {
                    if (wm_cuda_qkv_fused_rms_rope_f32(
                        rt->d_qkv_raw, rt->d_q, rt->d_k, d_v_cur,
                        rt->d_x_pos, rt->d_y_pos, rt->d_t_pos, rt->d_xy_table, rt->d_inv_t,
                        rt->T, cfg->n_heads, cfg->n_kv_heads, rt->d_head, cfg->width, cfg->height, rt->rms_eps)) return 1;
                }
            }
            STEP_CUDA(cudaGetLastError());
            if (cfg->value_residual && layer_idx != 0) {
                if (wm_cuda_lerp_inplace_f32(
                    rt->d_v, rt->d_v_first, lw->v_lamb, (int64_t)rt->kv_rope_elems)) return 1;
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
                if (wm_cuda_kv_cache_upsert_copy_f16(
                    cache->k_h, cache->v_h, rt->d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, rt->T, rt->d_head, cache->ring_length, base, write_step, (bool)frozen_pass)) return 1;
            } else {
                if (wm_cuda_kv_cache_upsert_copy_f32(
                    cache->k, cache->v, rt->d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, rt->T, rt->d_head, cache->ring_length, base, write_step, (bool)frozen_pass)) return 1;
            }
            STEP_CUDA(cudaGetLastError());
            if (wm_cuda_collect_cache_frame_indices(
                cache->written, cache->indices, cache->block_ids, cache->index_count,
                cache->capacity, rt->T, base, write_step)) return 1;
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
                                wm_cuda_should_use_m64n64_tensorop(
                                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->D);
                            cutlass_rc = wm_cuda_attention_d64_sparse_fmha_f16_kv(
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
                                wm_cuda_should_use_m64n64_tensorop(
                                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->D);
                            cutlass_rc = wm_cuda_attention_d64_fmha_f16_kv(
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
                            cutlass_rc = wm_cuda_attention_d64_cutlass_grouped_f16_kv(
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
                            cutlass_rc = wm_cuda_attention_d64_cutlass_f16_kv(
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
                        if (wm_cuda_indexed_attention_d64_flash_f16_kv(
                            rt->d_q, cache->k_h, cache->v_h, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f)) return 1;
                    } else {
                        if (wm_cuda_indexed_attention_d64_warp_f16_kv(
                            rt->d_q, cache->k_h, cache->v_h, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f)) return 1;
                    }
                    attn_done = 1;
                }
                if (rt->attn_flash_enabled && cfg->n_heads % cfg->n_kv_heads == 0 && (cfg->n_heads / cfg->n_kv_heads) <= WORLD_ATTN_D64_FLASH_WARPS) {
                    if (wm_cuda_indexed_attention_d64_flash_f32(
                        rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                        cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                        1.0f / 8.0f)) return 1;
                    attn_done = 1;
                }
                if (!attn_done) {
                    if (rt->attn_q4_shared_enabled) {
                        if (wm_cuda_indexed_attention_d64_q4_shared_f32(
                            rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f)) return 1;
                    } else {
                        if (wm_cuda_indexed_attention_d64_warp_f32(
                            rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                            cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity,
                            1.0f / 8.0f)) return 1;
                    }
                }
            } else {
                if (wm_cuda_indexed_attention_f32(
                    rt->d_q, cache->k, cache->v, cache->indices, cache->index_count, rt->d_attn,
                    cfg->n_heads, cfg->n_kv_heads, rt->T, cache->capacity, rt->d_head,
                    1.0f / sqrtf((float)rt->d_head))) return 1;
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_attn_ms, prof_attn_calls);
            STEP_PROFILE_BEGIN();
            if (out_w8a8) {
                if (wm_cuda_quantize_rows_f32_i8(
                    rt->d_attn,
                    rt->d_w8a8_x,
                    rt->d_w8a8_x_scales,
                    rt->T,
                    rt->D)) return 1;
                STEP_CUDA(cudaGetLastError());
                if (wm_cuda_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->out_proj_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->D,
                    rt->D)) return 1;
            } else if (attn_output_half_ready) {
                if (wm_cuda_linear_fp16_input_weight_tensorop_m64n64(
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
                if (wm_cuda_dequant_gated_residual_f32(
                    rt->d_w8a8_acc,
                    rt->d_w8a8_x_scales,
                    lw->out_proj_weight_i8_scales,
                    d_tokens_cur,
                    d_g0,
                    rt->d_tokens_after_attn,
                    rt->T,
                    rt->D)) return 1;
            } else {
                if (wm_cuda_gated_residual_add_f32(
                    d_tokens_cur, rt->d_attn_out, d_g0, rt->d_tokens_after_attn, rt->T, rt->D)) return 1;
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
                    if (wm_cuda_rms_norm_quantize_rows_i8(
                        rt->d_tokens_after_attn,
                        NULL,
                        NULL,
                        rt->d_w8a8_x,
                        rt->d_w8a8_x_scales,
                        rt->T,
                        rt->D,
                        rt->rms_eps)) return 1;
                    STEP_CUDA(cudaGetLastError());
                    if (wm_cuda_gemm_i8_i32(
                        rt->d_w8a8_x,
                        lw->ctrl_fc1_x_weight_i8,
                        rt->d_w8a8_acc,
                        rt->T,
                        rt->D,
                        rt->D)) return 1;
                    if (wm_cuda_dequant_silu_quantize_rows_i8(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->ctrl_fc1_x_weight_i8_scales,
                        rt->d_ctrl_cond_by_layer + (size_t)layer_idx * rt->D,
                        rt->d_w8a8_x,
                        rt->d_w8a8_x_scales,
                        rt->T,
                        rt->D)) return 1;
                    STEP_CUDA(cudaGetLastError());
                    if (wm_cuda_gemm_i8_i32(
                        rt->d_w8a8_x,
                        lw->ctrl_fc2_weight_i8,
                        rt->d_w8a8_acc,
                        rt->T,
                        rt->D,
                        rt->D)) return 1;
                    if (wm_cuda_dequant_add_residual_f32(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->ctrl_fc2_weight_i8_scales,
                        rt->d_tokens_after_attn,
                        rt->d_tokens_after_ctrl,
                        rt->T,
                        rt->D)) return 1;
                    STEP_CUDA(cudaGetLastError());
                } else {
                    if (wm_cuda_rms_norm_rows_f32(
                        rt->d_tokens_after_attn, rt->d_ctrl_norm, rt->T, rt->D, rt->rms_eps)) return 1;
                    STEP_CUDA(cudaGetLastError());
                    STEP_LINEAR_FAST(rt->d_ctrl_norm, lw->ctrl_fc1_x_weight, lw->ctrl_fc1_x_weight_h,
                                     rt->d_ctrl_hidden, rt->T, rt->D, rt->D);
                    if (wm_cuda_add_channel_silu_inplace_f32(
                        rt->d_ctrl_hidden, rt->d_ctrl_cond_by_layer + (size_t)layer_idx * rt->D, rt->T, rt->D)) return 1;
                    STEP_CUDA(cudaGetLastError());
                    STEP_LINEAR_FAST(rt->d_ctrl_hidden, lw->ctrl_fc2_weight, lw->ctrl_fc2_weight_h,
                                     rt->d_ctrl_out, rt->T, rt->D, rt->D);
                    if (wm_cuda_add_f32(
                        rt->d_tokens_after_attn, rt->d_ctrl_out, rt->d_tokens_after_ctrl, rt->token_elems)) return 1;
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
                wm_cuda_should_use_m64n64_tensorop(
                    rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->mlp_hidden);
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                if (wm_cuda_rms_norm_quantize_rows_i8(
                    d_tokens_ctrl,
                    d_s1,
                    d_b1,
                    rt->d_w8a8_x,
                    rt->d_w8a8_x_scales,
                    rt->T,
                    rt->D,
                    rt->rms_eps)) return 1;
            } else if (mlp_fc1_half_boundary) {
                if (wm_cuda_ada_rms_norm_f16(
                    d_tokens_ctrl, d_s1, d_b1, rt->d_linear_half, rt->T, rt->D, rt->rms_eps)) return 1;
            } else {
                if (wm_cuda_ada_rms_norm_f32(
                    d_tokens_ctrl, d_s1, d_b1, rt->d_mlp_in, rt->T, rt->D, rt->rms_eps)) return 1;
            }
            STEP_CUDA(cudaGetLastError());
            STEP_PROFILE_ACCUM(prof_norm_ms, prof_norm_calls);
            int mlp_hidden_half_ready = 0;
            __half *d_mlp_hidden_half_cur = NULL;
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                if (wm_cuda_gemm_i8_i32(
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
                    wm_cuda_should_use_m64n64_tensorop(rt->fp16_gemm_m64n64_enabled, rt->T, rt->D, rt->mlp_hidden)) {
                if (mlp_fc1_half_boundary) {
                    if (wm_cuda_linear_fp16_input_weight_tensorop_m64n64_silu_half(
                                rt->d_linear_half,
                                lw->dit_mlp_fc1_weight_h,
                                rt->d_mlp_hidden_half,
                                rt->T,
                                rt->D,
                                rt->mlp_hidden)) return 1;
                } else {
                    if (wm_cuda_linear_fp16_weight_tensorop_m64n64_silu_half(
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
                    if (wm_cuda_dequant_silu_quantize_rows_i8(
                        rt->d_w8a8_acc,
                        rt->d_w8a8_x_scales,
                        lw->dit_mlp_fc1_weight_i8_scales,
                        NULL,
                        rt->d_w8a8_x,
                        rt->d_w8a8_x_scales,
                        rt->T,
                        rt->mlp_hidden)) return 1;
                } else if (use_fp16_gemm && use_fp16_tensorop && lw->dit_mlp_fc2_weight_h && rt->mlp_fc2_splitk_slices > 1) {
                    if (wm_cuda_silu_f32_to_f16(
                        rt->d_mlp_hidden, rt->d_mlp_hidden_half, (int64_t)rt->T * rt->mlp_hidden)) return 1;
                    mlp_hidden_half_ready = 1;
                    d_mlp_hidden_half_cur = rt->d_mlp_hidden_half;
                } else {
                    if (wm_cuda_silu_f32(
                        rt->d_mlp_hidden, rt->d_mlp_hidden, (int64_t)rt->T * rt->mlp_hidden)) return 1;
                }
                STEP_CUDA(cudaGetLastError());
                STEP_PROFILE_ACCUM(prof_mlp_silu_ms, prof_mlp_silu_calls);
            }
            STEP_PROFILE_BEGIN();
            if (mlp_w8a8) {
                if (wm_cuda_gemm_i8_i32(
                    rt->d_w8a8_x,
                    lw->dit_mlp_fc2_weight_i8,
                    rt->d_w8a8_acc,
                    rt->T,
                    rt->mlp_hidden,
                    rt->D)) return 1;
            } else if (mlp_hidden_half_ready) {
                if (!d_mlp_hidden_half_cur) return 1;
                if (rt->mlp_fc2_splitk_parallel_enabled) {
                    if (wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel(
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
                    if (wm_cuda_linear_fp16_input_weight_tensorop_splitk(
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
                    wm_cuda_should_use_m64n64_tensorop(rt->fp16_gemm_m64n64_enabled, rt->T, rt->mlp_hidden, rt->D)) {
                if (wm_cuda_linear_fp16_weight_tensorop_m64n64(
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
                if (wm_cuda_dequant_gated_residual_f32(
                    rt->d_w8a8_acc,
                    rt->d_w8a8_x_scales,
                    lw->dit_mlp_fc2_weight_i8_scales,
                    d_tokens_ctrl,
                    d_g1,
                    d_tokens_next,
                    rt->T,
                    rt->D)) return 1;
            } else {
                if (wm_cuda_gated_residual_add_f32(
                    d_tokens_ctrl, rt->d_mlp_out, d_g1, d_tokens_next, rt->T, rt->D)) return 1;
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
        if (wm_cuda_out_norm_silu_f32(d_tokens_cur, d_out_mod, rt->d_final_tokens, rt->T, rt->D, rt->rms_eps)) return 1;
        STEP_CUDA(cudaGetLastError());
        if (wm_cuda_unpatchify_f32(
            rt->d_final_tokens, rt->d_unpatch_w, rt->d_unpatch_b, rt->d_latent_out,
            rt->T, rt->D, rt->C, rt->H, rt->W, rt->ph, rt->pw, cfg->width, rt->C * rt->ph * rt->pw)) return 1;
        STEP_CUDA(cudaGetLastError());
        if (wm_cuda_latent_update_f32(
            rt->d_latent, rt->d_latent_out, dsigma, (int64_t)rt->latent_elems)) return 1;
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
    if (wm_cuda_vae_decode_rgb(
            rt->vae,
            rt->d_latent,
            rgb_out,
            frames_out,
            width_out,
            height_out)) return 1;
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
