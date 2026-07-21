#include "world_cuda_probe.h"

#include "world_cuda_internal.cuh"
#include "world_cuda_ops.cuh"
#include "world_cuda_vae.cuh"

#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int dump_cache_written_counts(
        const char *prefix,
        const DeviceWorldLayerCache *caches,
        int n_layers) {
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
        cudaError_t err = cudaMemcpy(
            written,
            cache->written,
            (size_t)cache->capacity * sizeof(bool),
            cudaMemcpyDeviceToHost);
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

static int probe_write_ppm(
        const char *path,
        const unsigned char *rgb,
        int width,
        int height) {
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

static int probe_make_frame_path(
        char *out,
        size_t out_size,
        const char *path,
        int frame_idx) {
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    if (backslash && (!slash || backslash > slash)) slash = backslash;
    const char *name = slash ? slash + 1 : path;
    const char *dot = strrchr(name, '.');
    int stem_len = dot ? (int)(dot - path) : (int)strlen(path);
    const char *ext = dot ? dot : "";
    int n = snprintf(out, out_size, "%.*s.%d%s", stem_len, path, frame_idx, ext);
    return n < 0 || (size_t)n >= out_size;
}

static int probe_write_ppm_frames(
        const char *path,
        const unsigned char *rgb,
        int frame_count,
        int width,
        int height,
        int frame_offset) {
    size_t frame_bytes = (size_t)width * height * 3;
    if (frame_offset == 0) {
        if (probe_write_ppm(path, rgb, width, height)) return 1;
        fprintf(stderr, "wrote RGB image: %s\n", path);
    }
    for (int i = 0; i < frame_count; ++i) {
        char frame_path[4096];
        int global_frame = frame_offset + i;
        if (probe_make_frame_path(frame_path, sizeof(frame_path), path, global_frame)) {
            fprintf(stderr, "output frame path too long for %s frame %d\n", path, global_frame);
            return 1;
        }
        if (probe_write_ppm(
                frame_path,
                rgb + (size_t)i * frame_bytes,
                width,
                height)) return 1;
        fprintf(stderr, "wrote RGB frame %d: %s\n", global_frame, frame_path);
    }
    return 0;
}

static int probe_vae_decode_ppm(
        WorldCudaVae *vae,
        const float *device_latent,
        const char *out_path,
        int frame_offset) {
    const unsigned char *rgb = NULL;
    int frames = 0;
    int width = 0;
    int height = 0;
    if (!vae || !device_latent || !out_path || !out_path[0]) return 1;
    if (wm_cuda_vae_decode_rgb(
            vae,
            device_latent,
            &rgb,
            &frames,
            &width,
            &height)) return 1;
    return probe_write_ppm_frames(
        out_path, rgb, frames, width, height, frame_offset);
}

extern "C" int world_cuda_vae_decode_probe(
        const WorldConfig *cfg,
        const float *latent,
        const WorldVaeDecoderWeights *vae,
        const char *out_path) {
    if (!cfg || !latent || !vae || !out_path || !out_path[0]) return 1;
    WorldCudaVae *device_vae = NULL;
    float *device_latent = NULL;
    int rc = 1;
    size_t latent_elems = (size_t)cfg->channels *
                          (size_t)(cfg->height * cfg->patch_h) *
                          (size_t)(cfg->width * cfg->patch_w);
    if (cudaMalloc((void **)&device_latent, latent_elems * sizeof(float)) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(
            device_latent,
            latent,
            latent_elems * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (wm_cuda_vae_create(&device_vae, cfg, vae)) goto cleanup;
    if (probe_vae_decode_ppm(device_vae, device_latent, out_path, 0)) goto cleanup;
    rc = 0;

cleanup:
    wm_cuda_vae_destroy(device_vae);
    cudaFree(device_latent);
    return rc;
}

extern "C" int world_cuda_vae_decode_sequence_probe(
        const WorldConfig *cfg,
        const float *latents,
        int latent_count,
        const WorldVaeDecoderWeights *vae,
        const char *out_path) {
    if (!cfg || !latents || latent_count <= 0 || !vae || !out_path || !out_path[0]) return 1;
    WorldCudaVae *device_vae = NULL;
    float *device_latent = NULL;
    int rc = 1;
    size_t latent_elems = (size_t)cfg->channels *
                          (size_t)(cfg->height * cfg->patch_h) *
                          (size_t)(cfg->width * cfg->patch_w);
    if (cudaMalloc((void **)&device_latent, latent_elems * sizeof(float)) != cudaSuccess) goto cleanup;
    if (wm_cuda_vae_create(&device_vae, cfg, vae)) goto cleanup;
    for (int i = 0; i < latent_count; ++i) {
        const float *src = latents + (size_t)i * latent_elems;
        if (cudaMemcpy(
                device_latent,
                src,
                latent_elems * sizeof(float),
                cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
        if (probe_vae_decode_ppm(device_vae, device_latent, out_path, i * 4)) goto cleanup;
    }
    rc = 0;

cleanup:
    wm_cuda_vae_destroy(device_vae);
    cudaFree(device_latent);
    return rc;
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
        wm_cuda_fill_latent(h_latent, (int)latent_elems, seed, noise_mode);
    }
    wm_cuda_fill_noise_embedding(h_noise, sigma);
    wm_cuda_fill_positions(h_x_pos, h_y_pos, h_t_pos, T, cfg->width, 0);
    wm_cuda_fill_rope_tables(h_xy, h_inv_t, d_head, cfg->height, cfg->width);

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
    if (wm_cuda_linear_f32((x), (w), (y), (m), (k), (n))) goto cleanup_device; \
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

    if (wm_cuda_copy_f32_to_device(&d_patch, weights->patchify_weight, patch_weight_elems)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_denoise_fc1, weights->denoise_fc1_weight, (size_t)mlp_hidden * 512)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_denoise_fc2, weights->denoise_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)mlp_hidden * ctrl_dim)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_cond_bias, weights->layer0_cond_bias, (size_t)D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_cond_s_w, weights->layer0_attn_cond_s_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_cond_b_w, weights->layer0_attn_cond_b_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_cond_g_w, weights->layer0_attn_cond_g_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_q_w, weights->layer0_q_proj_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_k_w, weights->layer0_k_proj_weight, (size_t)kv_dim * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_v_w, weights->layer0_v_proj_weight, (size_t)kv_dim * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_out_w, weights->layer0_out_proj_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_mlp_cond_s_w, weights->layer0_mlp_cond_s_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_mlp_cond_b_w, weights->layer0_mlp_cond_b_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_mlp_cond_g_w, weights->layer0_mlp_cond_g_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_fc1_x_w, weights->layer0_ctrl_fc1_x_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_fc1_c_w, weights->layer0_ctrl_fc1_c_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_fc2_w, weights->layer0_ctrl_fc2_weight, (size_t)D * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_dit_mlp_fc1_w, weights->layer0_dit_mlp_fc1_weight, (size_t)mlp_hidden * D)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_dit_mlp_fc2_w, weights->layer0_dit_mlp_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;

    TRY_CUDA(cudaMemcpy(d_latent, h_latent, latent_elems * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_noise, h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_control_input, weights->control_input, (size_t)ctrl_dim * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_xy_table, h_xy, (size_t)d_xy * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_inv_t, h_inv_t, (size_t)d_t * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_x_pos, h_x_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_y_pos, h_y_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));
    TRY_CUDA(cudaMemcpy(d_t_pos, h_t_pos, (size_t)T * sizeof(int64_t), cudaMemcpyHostToDevice));

    TRY_LINEAR(d_control_input, d_ctrl_emb_fc1_w, d_ctrl_emb_hidden, 1, ctrl_dim, mlp_hidden);
    if (wm_cuda_silu_f32(d_ctrl_emb_hidden, d_ctrl_emb_hidden, mlp_hidden)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_emb_hidden, d_ctrl_emb_fc2_w, d_ctrl_emb, 1, mlp_hidden, D);
    if (wm_cuda_rms_norm_rows_f32(d_ctrl_emb, d_ctrl_emb_norm, 1, D, 1.0e-6f)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_emb_norm, d_ctrl_fc1_c_w, d_ctrl_cond, 1, D, D);

    TRY_LINEAR(d_noise, d_denoise_fc1, d_noise_hidden, 1, 512, mlp_hidden);
    if (wm_cuda_silu_f32(d_noise_hidden, d_noise_hidden, mlp_hidden)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_noise_hidden, d_denoise_fc2, d_cond, 1, mlp_hidden, D);

    if (wm_cuda_add_bias_silu_f32(d_cond, d_cond_bias, d_cond_act, D)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_cond_act, d_cond_s_w, d_s0, 1, D, D);
    TRY_LINEAR(d_cond_act, d_cond_b_w, d_b0, 1, D, D);
    TRY_LINEAR(d_cond_act, d_cond_g_w, d_g0, 1, D, D);
    TRY_LINEAR(d_cond_act, d_mlp_cond_s_w, d_s1, 1, D, D);
    TRY_LINEAR(d_cond_act, d_mlp_cond_b_w, d_b1, 1, D, D);
    TRY_LINEAR(d_cond_act, d_mlp_cond_g_w, d_g1, 1, D, D);

    if (wm_cuda_patchify_f32(d_latent, d_patch, d_tokens, C, H, W, D, ph, pw, cfg->height, cfg->width)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    if (wm_cuda_ada_rms_norm_f32(d_tokens, d_s0, d_b0, d_norm, T, D, 1.0e-6f)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());

    TRY_LINEAR(d_norm, d_q_w, d_q_raw, T, D, D);
    TRY_LINEAR(d_norm, d_k_w, d_k_raw, T, D, kv_dim);
    TRY_LINEAR(d_norm, d_v_w, d_v_raw, T, D, kv_dim);

    {
        if (wm_cuda_qkv_separate_rms_rope_f32(
            d_q_raw, d_k_raw, d_v_raw,
            d_q, d_k, d_v,
            d_x_pos, d_y_pos, d_t_pos, d_xy_table, d_inv_t,
            T, cfg->n_heads, cfg->n_kv_heads, d_head, cfg->width, cfg->height, 1.0e-6f)) goto cleanup_device;
    }
    TRY_CUDA(cudaGetLastError());

    if (wm_cuda_current_frame_attention_f32(
        d_q, d_k, d_v, d_attn,
        cfg->n_heads, cfg->n_kv_heads, T, d_head, 1.0f / sqrtf((float)d_head))) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_attn, d_out_w, d_attn_out, T, D, D);
    if (wm_cuda_gated_residual_add_f32(
        d_tokens, d_attn_out, d_g0, d_tokens_after_attn, T, D)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());

    if (wm_cuda_rms_norm_rows_f32(d_tokens_after_attn, d_ctrl_norm, T, D, 1.0e-6f)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_norm, d_ctrl_fc1_x_w, d_ctrl_hidden, T, D, D);
    if (wm_cuda_add_channel_silu_inplace_f32(d_ctrl_hidden, d_ctrl_cond, T, D)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_ctrl_hidden, d_ctrl_fc2_w, d_ctrl_out, T, D, D);
    if (wm_cuda_add_f32(
        d_tokens_after_attn, d_ctrl_out, d_tokens_after_ctrl, token_elems)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());

    if (wm_cuda_ada_rms_norm_f32(d_tokens_after_ctrl, d_s1, d_b1, d_mlp_in, T, D, 1.0e-6f)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_mlp_in, d_dit_mlp_fc1_w, d_mlp_hidden, T, D, mlp_hidden);
    if (wm_cuda_silu_f32(
        d_mlp_hidden, d_mlp_hidden, (int64_t)T * mlp_hidden)) goto cleanup_device;
    TRY_CUDA(cudaGetLastError());
    TRY_LINEAR(d_mlp_hidden, d_dit_mlp_fc2_w, d_mlp_out, T, mlp_hidden, D);
    if (wm_cuda_gated_residual_add_f32(
        d_tokens_after_ctrl, d_mlp_out, d_g1, d_tokens_after_mlp, T, D)) goto cleanup_device;
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
        const WorldModelWeights *weights,
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
    wm_cuda_fill_rope_tables(h_xy, h_inv_t, d_head, cfg->height, cfg->width);

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
    WorldCudaVae *probe_vae = NULL;

#define TRY_CUDA2(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        goto cleanup_device; \
    } \
} while (0)
#define TRY_LINEAR2(x, w, y, m, k, n) do { \
    if (wm_cuda_linear_f32((x), (w), (y), (m), (k), (n))) goto cleanup_device; \
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

    if (wm_cuda_copy_f32_to_device(&d_patch, weights->patchify_weight, patch_weight_elems)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_denoise_fc1, weights->denoise_fc1_weight, (size_t)mlp_hidden * 512)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_denoise_fc2, weights->denoise_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_emb_fc1_w, weights->ctrl_emb_fc1_weight, (size_t)mlp_hidden * ctrl_dim)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_ctrl_emb_fc2_w, weights->ctrl_emb_fc2_weight, (size_t)D * mlp_hidden)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_out_norm_w, weights->out_norm_fc_weight, out_norm_weight_elems)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_unpatch_w, weights->unpatchify_weight, unpatch_weight_elems)) goto cleanup_device;
    if (wm_cuda_copy_f32_to_device(&d_unpatch_b, weights->unpatchify_bias, (size_t)C)) goto cleanup_device;
    if (wm_cuda_copy_world_layers_to_device(
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
    if (wm_cuda_alloc_device_world_caches(&d_caches, cfg, layers_to_run, T, cfg->n_kv_heads, d_head, 0)) goto cleanup_device;

    TRY_CUDA2(cudaMemcpy(d_xy_table, h_xy, (size_t)d_xy * sizeof(float), cudaMemcpyHostToDevice));
    TRY_CUDA2(cudaMemcpy(d_inv_t, h_inv_t, (size_t)d_t * sizeof(float), cudaMemcpyHostToDevice));
    if (vae && out_path && out_path[0]) {
        if (wm_cuda_vae_create(&probe_vae, cfg, vae)) goto cleanup_device;
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
        if (wm_cuda_silu_f32(d_ctrl_emb_hidden, d_ctrl_emb_hidden, mlp_hidden)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());
        TRY_LINEAR2(d_ctrl_emb_hidden, d_ctrl_emb_fc2_w, d_ctrl_emb, 1, mlp_hidden, D);
        if (wm_cuda_rms_norm_rows_f32(d_ctrl_emb, d_ctrl_emb_norm, 1, D, 1.0e-6f)) goto cleanup_device;
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
            wm_cuda_fill_latent(h_latent, (int)latent_elems, seed + (unsigned int)frame_ordinal, noise_mode);
        }
        wm_cuda_fill_positions(h_x_pos, h_y_pos, h_t_pos, T, cfg->width, frame_timestamp);
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

        wm_cuda_fill_noise_embedding(h_noise, sigma_step);
        TRY_CUDA2(cudaMemcpy(d_noise, h_noise, 512 * sizeof(float), cudaMemcpyHostToDevice));
        TRY_LINEAR2(d_noise, d_denoise_fc1, d_noise_hidden, 1, 512, mlp_hidden);
        if (wm_cuda_silu_f32(d_noise_hidden, d_noise_hidden, mlp_hidden)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());
        TRY_LINEAR2(d_noise_hidden, d_denoise_fc2, d_cond, 1, mlp_hidden, D);

        if (wm_cuda_patchify_f32(d_latent, d_patch, d_tokens, C, H, W, D, ph, pw, cfg->height, cfg->width)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());
        float *d_tokens_cur = d_tokens;
        float *d_tokens_next = d_tokens_after_mlp;

        for (int layer_idx = 0; layer_idx < layers_to_run; ++layer_idx) {
            const DeviceWorldLayerWeights *lw = &d_layers[layer_idx];
            DeviceWorldLayerCache *cache = &d_caches[layer_idx];
            fprintf(stderr, "  standalone layer %02d/%02d\n", layer_idx, layers_to_run);

            if (wm_cuda_add_bias_silu_f32(d_cond, lw->cond_bias, d_cond_act, D)) goto cleanup_device;
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_cond_act, lw->cond_proj_weight, d_layer_mod, 1, D, 6 * D);
            float *d_s0 = d_layer_mod;
            float *d_b0 = d_layer_mod + D;
            float *d_g0 = d_layer_mod + 2 * D;
            float *d_s1 = d_layer_mod + 3 * D;
            float *d_b1 = d_layer_mod + 4 * D;
            float *d_g1 = d_layer_mod + 5 * D;

            if (wm_cuda_ada_rms_norm_f32(d_tokens_cur, d_s0, d_b0, d_norm, T, D, 1.0e-6f)) goto cleanup_device;
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_norm, lw->qkv_proj_weight, d_qkv_raw, T, D, D + 2 * kv_dim);
            float *d_v_cur = (cfg->value_residual && layer_idx == 0) ? d_v_first : d_v;

            {
                if (wm_cuda_qkv_fused_rms_rope_f32(
                    d_qkv_raw,
                    d_q, d_k, d_v_cur,
                    d_x_pos, d_y_pos, d_t_pos, d_xy_table, d_inv_t,
                    T, cfg->n_heads, cfg->n_kv_heads, d_head, cfg->width, cfg->height, 1.0e-6f)) goto cleanup_device;
            }
            TRY_CUDA2(cudaGetLastError());

            if (cfg->value_residual) {
                if (layer_idx != 0) {
                    if (wm_cuda_lerp_inplace_f32(
                        d_v, d_v_first, lw->v_lamb, (int64_t)kv_rope_elems)) goto cleanup_device;
                    TRY_CUDA2(cudaGetLastError());
                }
            }

            {
                int bucket = (current_frame_idx + (cache->pinned_dilation - 1)) / cache->pinned_dilation;
                int num_buckets = (cache->ring_length / T) / cache->pinned_dilation;
                int base = (bucket % num_buckets) * T;
                bool write_step = (current_frame_idx % cache->pinned_dilation) == 0;
                if (wm_cuda_kv_cache_upsert_copy_f32(
                    cache->k, cache->v, d_k, d_v_cur, cache->written,
                    cfg->n_kv_heads, T, d_head, cache->ring_length, base, write_step, (bool)frozen_pass)) goto cleanup_device;
                TRY_CUDA2(cudaGetLastError());
                if (wm_cuda_collect_cache_frame_indices(
                    cache->written, cache->indices, cache->block_ids, cache->index_count,
                    cache->capacity, T, base, write_step)) goto cleanup_device;
                TRY_CUDA2(cudaGetLastError());
                if (wm_cuda_indexed_attention_f32(
                    d_q, cache->k, cache->v, cache->indices, cache->index_count, d_attn,
                    cfg->n_heads, cfg->n_kv_heads, T, cache->capacity, d_head,
                    1.0f / sqrtf((float)d_head))) goto cleanup_device;
            }
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_attn, lw->out_proj_weight, d_attn_out, T, D, D);
            if (wm_cuda_gated_residual_add_f32(
                d_tokens_cur, d_attn_out, d_g0, d_tokens_after_attn, T, D)) goto cleanup_device;
            TRY_CUDA2(cudaGetLastError());

            float *d_tokens_ctrl = d_tokens_after_attn;
            if (lw->has_ctrl) {
                if (wm_cuda_rms_norm_rows_f32(d_tokens_after_attn, d_ctrl_norm, T, D, 1.0e-6f)) goto cleanup_device;
                TRY_CUDA2(cudaGetLastError());
                TRY_LINEAR2(d_ctrl_norm, lw->ctrl_fc1_x_weight, d_ctrl_hidden, T, D, D);
                if (wm_cuda_add_channel_silu_inplace_f32(
                    d_ctrl_hidden, d_ctrl_cond_by_layer + (size_t)layer_idx * D, T, D)) goto cleanup_device;
                TRY_CUDA2(cudaGetLastError());
                TRY_LINEAR2(d_ctrl_hidden, lw->ctrl_fc2_weight, d_ctrl_out, T, D, D);
                if (wm_cuda_add_f32(
                    d_tokens_after_attn, d_ctrl_out, d_tokens_after_ctrl, token_elems)) goto cleanup_device;
                TRY_CUDA2(cudaGetLastError());
                d_tokens_ctrl = d_tokens_after_ctrl;
            }

            if (wm_cuda_ada_rms_norm_f32(d_tokens_ctrl, d_s1, d_b1, d_mlp_in, T, D, 1.0e-6f)) goto cleanup_device;
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_mlp_in, lw->dit_mlp_fc1_weight, d_mlp_hidden, T, D, mlp_hidden);
            if (wm_cuda_silu_f32(
                d_mlp_hidden, d_mlp_hidden, (int64_t)T * mlp_hidden)) goto cleanup_device;
            TRY_CUDA2(cudaGetLastError());
            TRY_LINEAR2(d_mlp_hidden, lw->dit_mlp_fc2_weight, d_mlp_out, T, mlp_hidden, D);
            if (wm_cuda_gated_residual_add_f32(
                d_tokens_ctrl, d_mlp_out, d_g1, d_tokens_next, T, D)) goto cleanup_device;
            TRY_CUDA2(cudaGetLastError());

            float *d_swap = d_tokens_cur;
            d_tokens_cur = d_tokens_next;
            d_tokens_next = d_swap;
        }

        if (is_cache_pass) {
            continue;
        }

        if (wm_cuda_silu_f32(d_cond, d_cond_act, D)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());
        TRY_LINEAR2(d_cond_act, d_out_norm_w, d_out_mod, 1, D, 2 * D);
        if (wm_cuda_out_norm_silu_f32(d_tokens_cur, d_out_mod, d_final_tokens, T, D, 1.0e-6f)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());
        if (wm_cuda_unpatchify_f32(
            d_final_tokens, d_unpatch_w, d_unpatch_b, d_latent_out,
            T, D, C, H, W, ph, pw, cfg->width, out_dim)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());

        if (is_last_step) {
            TRY_CUDA2(cudaMemcpy(h_tokens, d_tokens_cur, token_elems * sizeof(float), cudaMemcpyDeviceToHost));
            TRY_CUDA2(cudaMemcpy(h_latent_out, d_latent_out, latent_elems * sizeof(float), cudaMemcpyDeviceToHost));
        }

        if (wm_cuda_latent_update_f32(
            d_latent, d_latent_out, dsigma, (int64_t)latent_elems)) goto cleanup_device;
        TRY_CUDA2(cudaGetLastError());
    }

        if (probe_vae && probe_vae_decode_ppm(
                probe_vae,
                d_latent,
                out_path,
                frame_ordinal * decoded_frames_per_latent)) {
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
    wm_cuda_vae_destroy(probe_vae);
    wm_cuda_free_device_world_layers(d_layers, layers_to_run);
    wm_cuda_free_device_world_caches(d_caches, layers_to_run);
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
