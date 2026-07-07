#define _FILE_OFFSET_BITS 64

#include "safetensors.h"
#include "world_config.h"
#include "world_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATH_BUF 4096

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--model-dir DIR] [--weights FILE] [--seed N] [--sigma X] [--dump-prefix PATH]\n"
            "\n"
            "Standalone C+CUDA probe. Loads Waypoint config and safetensors without PyTorch,\n"
            "then starts the generation path through layer0 Q/K/V RoPE.\n",
            argv0);
}

static int join_path(char *out, size_t out_size, const char *a, const char *b) {
    int n = snprintf(out, out_size, "%s/%s", a, b);
    return n < 0 || (size_t)n >= out_size;
}

static int expect_shape(const SafeTensorEntry *e, const int64_t *shape, int ndim) {
    if (!e || e->ndim != ndim) return 1;
    for (int i = 0; i < ndim; ++i) {
        if (e->shape[i] != shape[i]) return 1;
    }
    return 0;
}

static int load_required_f32(const SafeTensors *st, const char *name, const int64_t *shape, int ndim, float **out) {
    const SafeTensorEntry *e = safetensors_find(st, name);
    if (!e) {
        fprintf(stderr, "missing tensor: %s\n", name);
        return 1;
    }
    safetensors_print_entry(e);
    if (strcmp(e->dtype, "F32") != 0) {
        fprintf(stderr, "expected F32 tensor for standalone probe: %s has %s\n", name, e->dtype);
        return 1;
    }
    if (expect_shape(e, shape, ndim)) {
        fprintf(stderr, "shape mismatch for %s\n", name);
        return 1;
    }
    void *data = NULL;
    size_t bytes = 0;
    if (safetensors_read_tensor(st, e, &data, &bytes)) {
        fprintf(stderr, "failed to read tensor: %s\n", name);
        return 1;
    }
    *out = (float *)data;
    return 0;
}

int main(int argc, char **argv) {
    const char *model_dir = "../Waypoint-1.5-1B";
    const char *weights = NULL;
    const char *dump_prefix = NULL;
    unsigned int seed = 1234;
    float sigma = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            sigma = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    char config_path[PATH_BUF];
    if (join_path(config_path, sizeof(config_path), model_dir, "config.yaml")) {
        fprintf(stderr, "config path too long\n");
        return 1;
    }

    char default_weights[PATH_BUF];
    if (!weights) {
        if (join_path(default_weights, sizeof(default_weights), model_dir, "transformer/diffusion_pytorch_model.safetensors")) {
            fprintf(stderr, "weights path too long\n");
            return 1;
        }
        weights = default_weights;
    }

    WorldConfig cfg;
    if (world_config_load(&cfg, config_path)) return 1;
    world_config_print(&cfg);

    SafeTensors st;
    fprintf(stderr, "loading safetensors index: %s\n", weights);
    if (safetensors_open(&st, weights)) return 1;
    fprintf(stderr, "safetensors tensors: %d\n", st.count);

    int64_t patch_shape[4] = {cfg.d_model, cfg.channels, cfg.patch_h, cfg.patch_w};
    int hidden = cfg.d_model * cfg.mlp_ratio;
    int d_head = cfg.d_model / cfg.n_heads;
    int kv_dim = cfg.n_kv_heads * d_head;
    int64_t denoise_fc1_shape[2] = {hidden, 512};
    int64_t denoise_fc2_shape[2] = {cfg.d_model, hidden};
    int64_t hidden_d_shape[2] = {hidden, cfg.d_model};
    int64_t d_shape[1] = {cfg.d_model};
    int64_t dxd_shape[2] = {cfg.d_model, cfg.d_model};
    int64_t kv_proj_shape[2] = {kv_dim, cfg.d_model};
    float *patchify_weight = NULL;
    float *denoise_fc1_weight = NULL;
    float *denoise_fc2_weight = NULL;
    float *layer0_cond_bias = NULL;
    float *layer0_attn_cond_s_weight = NULL;
    float *layer0_attn_cond_b_weight = NULL;
    float *layer0_attn_cond_g_weight = NULL;
    float *layer0_q_proj_weight = NULL;
    float *layer0_k_proj_weight = NULL;
    float *layer0_v_proj_weight = NULL;
    float *layer0_out_proj_weight = NULL;
    float *layer0_mlp_cond_s_weight = NULL;
    float *layer0_mlp_cond_b_weight = NULL;
    float *layer0_mlp_cond_g_weight = NULL;
    float *layer0_ctrl_fc1_x_weight = NULL;
    float *layer0_ctrl_fc2_weight = NULL;
    float *layer0_dit_mlp_fc1_weight = NULL;
    float *layer0_dit_mlp_fc2_weight = NULL;
    int rc = 1;

    if (load_required_f32(&st, "patchify.weight", patch_shape, 4, &patchify_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "denoise_step_emb.mlp.fc1.weight", denoise_fc1_shape, 2, &denoise_fc1_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "denoise_step_emb.mlp.fc2.weight", denoise_fc2_shape, 2, &denoise_fc2_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.mlp_cond_head.bias_in", d_shape, 1, &layer0_cond_bias)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn_cond_head.cond_proj.0.weight", dxd_shape, 2, &layer0_attn_cond_s_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn_cond_head.cond_proj.1.weight", dxd_shape, 2, &layer0_attn_cond_b_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn_cond_head.cond_proj.2.weight", dxd_shape, 2, &layer0_attn_cond_g_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn.q_proj.weight", dxd_shape, 2, &layer0_q_proj_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn.k_proj.weight", kv_proj_shape, 2, &layer0_k_proj_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn.v_proj.weight", kv_proj_shape, 2, &layer0_v_proj_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn.out_proj.weight", dxd_shape, 2, &layer0_out_proj_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.mlp_cond_head.cond_proj.0.weight", dxd_shape, 2, &layer0_mlp_cond_s_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.mlp_cond_head.cond_proj.1.weight", dxd_shape, 2, &layer0_mlp_cond_b_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.mlp_cond_head.cond_proj.2.weight", dxd_shape, 2, &layer0_mlp_cond_g_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.ctrl_mlpfusion.fc1_x.weight", dxd_shape, 2, &layer0_ctrl_fc1_x_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.ctrl_mlpfusion.fc2.weight", dxd_shape, 2, &layer0_ctrl_fc2_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.dit_mlp.fc1.weight", hidden_d_shape, 2, &layer0_dit_mlp_fc1_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "transformer.blocks.0.dit_mlp.fc2.weight", denoise_fc2_shape, 2, &layer0_dit_mlp_fc2_weight)) {
        goto cleanup;
    }

    WorldLayer0ProbeWeights layer0 = {
        patchify_weight,
        denoise_fc1_weight,
        denoise_fc2_weight,
        layer0_cond_bias,
        layer0_attn_cond_s_weight,
        layer0_attn_cond_b_weight,
        layer0_attn_cond_g_weight,
        layer0_q_proj_weight,
        layer0_k_proj_weight,
        layer0_v_proj_weight,
        layer0_out_proj_weight,
        layer0_mlp_cond_s_weight,
        layer0_mlp_cond_b_weight,
        layer0_mlp_cond_g_weight,
        layer0_ctrl_fc1_x_weight,
        layer0_ctrl_fc2_weight,
        layer0_dit_mlp_fc1_weight,
        layer0_dit_mlp_fc2_weight,
    };
    rc = world_cuda_layer0_probe(&cfg, &layer0, sigma, seed, dump_prefix);

cleanup:
    free(patchify_weight);
    free(denoise_fc1_weight);
    free(denoise_fc2_weight);
    free(layer0_cond_bias);
    free(layer0_attn_cond_s_weight);
    free(layer0_attn_cond_b_weight);
    free(layer0_attn_cond_g_weight);
    free(layer0_q_proj_weight);
    free(layer0_k_proj_weight);
    free(layer0_v_proj_weight);
    free(layer0_out_proj_weight);
    free(layer0_mlp_cond_s_weight);
    free(layer0_mlp_cond_b_weight);
    free(layer0_mlp_cond_g_weight);
    free(layer0_ctrl_fc1_x_weight);
    free(layer0_ctrl_fc2_weight);
    free(layer0_dit_mlp_fc1_weight);
    free(layer0_dit_mlp_fc2_weight);
    safetensors_close(&st);
    return rc;
}
