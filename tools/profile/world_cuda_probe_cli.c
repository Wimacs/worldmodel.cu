#define _FILE_OFFSET_BITS 64

#include "safetensors.h"
#include "world_config.h"
#include "world_cuda_probe.h"
#include "world_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--model-dir DIR] [--weights FILE] [--vae-weights FILE] [--control FILE] [--control-seq FILE] [--latent FILE] [--seed N] [--noise normal|uniform] [--sigma X] [--layers N] [--steps N] [--frames N] [--frame-idx N] [--cache-pass] [--vae-only] [--dump-prefix PATH] [--out PATH]\n"
            "\n"
            "Standalone C+CUDA probe. Loads Waypoint config and safetensors without PyTorch,\n"
            "then runs the WorldDiT latent path and optionally decodes an RGB PPM.\n",
            argv0);
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

static int load_optional_f32(const SafeTensors *st, const char *name, const int64_t *shape, int ndim, float **out) {
    const SafeTensorEntry *e = safetensors_find(st, name);
    if (!e) {
        *out = NULL;
        return 0;
    }
    return load_required_f32(st, name, shape, ndim, out);
}

static int load_layer_f32(const SafeTensors *st, int layer, const char *suffix, const int64_t *shape, int ndim, float **out) {
    char name[256];
    int n = snprintf(name, sizeof(name), "transformer.blocks.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return 1;
    return load_required_f32(st, name, shape, ndim, out);
}

static int load_optional_layer_f32(const SafeTensors *st, int layer, const char *suffix, const int64_t *shape, int ndim, float **out) {
    char name[256];
    int n = snprintf(name, sizeof(name), "transformer.blocks.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return 1;
    return load_optional_f32(st, name, shape, ndim, out);
}

int main(int argc, char **argv) {
    const char *model_dir = "../Waypoint-1.5-1B";
    const char *weights = NULL;
    const char *vae_weights = NULL;
    const char *control_path = NULL;
    const char *control_seq_path = NULL;
    const char *latent_path = NULL;
    const char *dump_prefix = NULL;
    const char *out_path = NULL;
    unsigned int seed = 1234;
    int noise_mode = WORLD_NOISE_NORMAL;
    float sigma = 1.0f;
    int layers_to_run = -1;
    int steps_to_run = 1;
    int frames_to_run = 1;
    int frame_idx = 0;
    int cache_pass = 0;
    int vae_only = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
        } else if (strcmp(argv[i], "--vae-weights") == 0 && i + 1 < argc) {
            vae_weights = argv[++i];
        } else if (strcmp(argv[i], "--control") == 0 && i + 1 < argc) {
            control_path = argv[++i];
        } else if (strcmp(argv[i], "--control-seq") == 0 && i + 1 < argc) {
            control_seq_path = argv[++i];
        } else if (strcmp(argv[i], "--latent") == 0 && i + 1 < argc) {
            latent_path = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--noise") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "normal") == 0) {
                noise_mode = WORLD_NOISE_NORMAL;
            } else if (strcmp(mode, "uniform") == 0) {
                noise_mode = WORLD_NOISE_UNIFORM;
            } else {
                fprintf(stderr, "invalid --noise %s, expected normal or uniform\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            sigma = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            layers_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--frame-idx") == 0 && i + 1 < argc) {
            frame_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache-pass") == 0) {
            cache_pass = 1;
        } else if (strcmp(argv[i], "--vae-only") == 0) {
            vae_only = 1;
        } else if (strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    char config_path[WORLD_PATH_MAX];
    if (world_path_join(config_path, sizeof(config_path), model_dir, "config.yaml")) {
        fprintf(stderr, "config path too long\n");
        return 1;
    }

    char default_weights[WORLD_PATH_MAX];
    if (!weights) {
        if (world_path_join(default_weights, sizeof(default_weights), model_dir, "transformer/diffusion_pytorch_model.safetensors")) {
            fprintf(stderr, "weights path too long\n");
            return 1;
        }
        weights = default_weights;
    }

    char default_vae_weights[WORLD_PATH_MAX];
    if (!vae_weights && (out_path || vae_only)) {
        if (world_path_join(default_vae_weights, sizeof(default_vae_weights), model_dir, "vae/diffusion_pytorch_model.safetensors")) {
            fprintf(stderr, "VAE weights path too long\n");
            return 1;
        }
        vae_weights = default_vae_weights;
    }

    WorldConfig cfg;
    if (world_config_load(&cfg, config_path)) return 1;
    if (vae_only) {
        if (!latent_path || !out_path) {
            fprintf(stderr, "--vae-only requires --latent FILE and --out PATH\n");
            return 1;
        }
        world_config_print(&cfg);
        WorldVaeDecoderWeights vae;
        float *latent = NULL;
        memset(&vae, 0, sizeof(vae));
        size_t latent_elems = (size_t)cfg.channels *
                              (size_t)(cfg.height * cfg.patch_h) *
                              (size_t)(cfg.width * cfg.patch_w);
        int rc = 1;
        if (world_load_vae_decoder_weights(vae_weights, &vae)) goto vae_only_cleanup;
        if (frames_to_run <= 0) {
            fprintf(stderr, "invalid --frames %d, expected >= 1\n", frames_to_run);
            goto vae_only_cleanup;
        }
        if (world_read_f32_file_exact(latent_path, latent_elems * (size_t)frames_to_run, &latent)) goto vae_only_cleanup;
        rc = world_cuda_vae_decode_sequence_probe(&cfg, latent, frames_to_run, &vae, out_path);
vae_only_cleanup:
        free(latent);
        world_free_vae_decoder_weights(&vae);
        return rc;
    }
    if (layers_to_run < 0) layers_to_run = cfg.n_layers;
    if (layers_to_run <= 0 || layers_to_run > cfg.n_layers) {
        fprintf(stderr, "invalid --layers %d, expected 1..%d\n", layers_to_run, cfg.n_layers);
        return 1;
    }
    if (steps_to_run <= 0 || steps_to_run >= cfg.scheduler_sigmas_count) {
        fprintf(stderr, "invalid --steps %d, expected 1..%d\n", steps_to_run, cfg.scheduler_sigmas_count - 1);
        return 1;
    }
    if (frames_to_run <= 0) {
        fprintf(stderr, "invalid --frames %d, expected >= 1\n", frames_to_run);
        return 1;
    }
    if (frame_idx < 0) {
        fprintf(stderr, "invalid --frame-idx %d, expected >= 0\n", frame_idx);
        return 1;
    }
    if (control_path && control_seq_path) {
        fprintf(stderr, "--control and --control-seq are mutually exclusive\n");
        return 1;
    }
    if (frames_to_run > 1 && !cache_pass) {
        fprintf(stderr, "--frames %d: enabling --cache-pass so generated frame history persists in KV cache\n", frames_to_run);
        cache_pass = 1;
    }
    world_config_print(&cfg);
    fprintf(stderr, "noise: %s seed=%u\n", noise_mode == WORLD_NOISE_NORMAL ? "normal" : "uniform", seed);

    SafeTensors st;
    fprintf(stderr, "loading safetensors index: %s\n", weights);
    if (safetensors_open(&st, weights)) return 1;
    fprintf(stderr, "safetensors tensors: %d\n", st.count);

    int64_t patch_shape[4] = {cfg.d_model, cfg.channels, cfg.patch_h, cfg.patch_w};
    int hidden = cfg.d_model * cfg.mlp_ratio;
    int d_head = cfg.d_model / cfg.n_heads;
    int kv_dim = cfg.n_kv_heads * d_head;
    int ctrl_dim = cfg.n_buttons + 3;
    int64_t denoise_fc1_shape[2] = {hidden, 512};
    int64_t denoise_fc2_shape[2] = {cfg.d_model, hidden};
    int64_t ctrl_emb_fc1_shape[2] = {hidden, ctrl_dim};
    int64_t ctrl_emb_fc2_shape[2] = {cfg.d_model, hidden};
    int64_t hidden_d_shape[2] = {hidden, cfg.d_model};
    int64_t d_shape[1] = {cfg.d_model};
    int64_t dxd_shape[2] = {cfg.d_model, cfg.d_model};
    int64_t out_norm_shape[2] = {cfg.d_model * 2, cfg.d_model};
    int64_t kv_proj_shape[2] = {kv_dim, cfg.d_model};
    int64_t unpatch_bias_shape[1] = {cfg.channels};
    float *patchify_weight = NULL;
    float *denoise_fc1_weight = NULL;
    float *denoise_fc2_weight = NULL;
    float *ctrl_emb_fc1_weight = NULL;
    float *ctrl_emb_fc2_weight = NULL;
    float *control_inputs = NULL;
    float *initial_latents = NULL;
    float *out_norm_fc_weight = NULL;
    float *unpatchify_weight = NULL;
    float *unpatchify_bias = NULL;
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
    float *layer0_ctrl_fc1_c_weight = NULL;
    float *layer0_ctrl_fc2_weight = NULL;
    float *layer0_dit_mlp_fc1_weight = NULL;
    float *layer0_dit_mlp_fc2_weight = NULL;
    WorldLayerWeights *layers = NULL;
    WorldVaeDecoderWeights vae;
    memset(&vae, 0, sizeof(vae));
    int rc = 1;

    if (out_path && world_load_vae_decoder_weights(vae_weights, &vae)) {
        goto cleanup;
    }

    size_t latent_elems_per_frame = (size_t)cfg.channels * (size_t)(cfg.height * cfg.patch_h) * (size_t)(cfg.width * cfg.patch_w);
    size_t latent_elems = (size_t)frames_to_run * latent_elems_per_frame;
    if (latent_path) {
        if (world_read_f32_file_exact(latent_path, latent_elems, &initial_latents)) goto cleanup;
        fprintf(stderr, "loaded initial latent(s): %s frames=%d elems_per_frame=%zu\n",
                latent_path, frames_to_run, latent_elems_per_frame);
    }

    size_t control_elems = (size_t)frames_to_run * (size_t)ctrl_dim;
    if (control_seq_path) {
        if (world_read_f32_file_exact(control_seq_path, control_elems, &control_inputs)) goto cleanup;
    } else if (control_path) {
        float *control_one = NULL;
        if (world_read_f32_file_exact(control_path, (size_t)ctrl_dim, &control_one)) goto cleanup;
        control_inputs = (float *)malloc(control_elems * sizeof(float));
        if (!control_inputs) {
            fprintf(stderr, "failed to allocate broadcast control input\n");
            free(control_one);
            goto cleanup;
        }
        for (int frame = 0; frame < frames_to_run; ++frame) {
            memcpy(control_inputs + (size_t)frame * ctrl_dim, control_one, (size_t)ctrl_dim * sizeof(float));
        }
        free(control_one);
    } else {
        control_inputs = (float *)calloc(control_elems, sizeof(float));
        if (!control_inputs) {
            fprintf(stderr, "failed to allocate zero control inputs\n");
            goto cleanup;
        }
    }

    if (load_required_f32(&st, "patchify.weight", patch_shape, 4, &patchify_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "denoise_step_emb.mlp.fc1.weight", denoise_fc1_shape, 2, &denoise_fc1_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "denoise_step_emb.mlp.fc2.weight", denoise_fc2_shape, 2, &denoise_fc2_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "ctrl_emb.mlp.fc1.weight", ctrl_emb_fc1_shape, 2, &ctrl_emb_fc1_weight)) {
        goto cleanup;
    }
    if (load_required_f32(&st, "ctrl_emb.mlp.fc2.weight", ctrl_emb_fc2_shape, 2, &ctrl_emb_fc2_weight)) {
        goto cleanup;
    }

    if (layers_to_run != 1) {
        if (load_required_f32(&st, "out_norm.fc.weight", out_norm_shape, 2, &out_norm_fc_weight)) {
            goto cleanup;
        }
        if (load_required_f32(&st, "unpatchify.weight", patch_shape, 4, &unpatchify_weight)) {
            goto cleanup;
        }
        if (load_required_f32(&st, "unpatchify.bias", unpatch_bias_shape, 1, &unpatchify_bias)) {
            goto cleanup;
        }
        layers = (WorldLayerWeights *)calloc((size_t)layers_to_run, sizeof(*layers));
        if (!layers) {
            fprintf(stderr, "failed to allocate layer weights\n");
            goto cleanup;
        }
        for (int layer = 0; layer < layers_to_run; ++layer) {
            WorldLayerWeights *lw = &layers[layer];
            if (load_layer_f32(&st, layer, "mlp_cond_head.bias_in", d_shape, 1, (float **)&lw->cond_bias)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn_cond_head.cond_proj.0.weight", dxd_shape, 2, (float **)&lw->attn_cond_s_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn_cond_head.cond_proj.1.weight", dxd_shape, 2, (float **)&lw->attn_cond_b_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn_cond_head.cond_proj.2.weight", dxd_shape, 2, (float **)&lw->attn_cond_g_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn.q_proj.weight", dxd_shape, 2, (float **)&lw->q_proj_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn.k_proj.weight", kv_proj_shape, 2, (float **)&lw->k_proj_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn.v_proj.weight", kv_proj_shape, 2, (float **)&lw->v_proj_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn.out_proj.weight", dxd_shape, 2, (float **)&lw->out_proj_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "attn.v_lamb", NULL, 0, (float **)&lw->v_lamb)) goto cleanup;
            if (load_layer_f32(&st, layer, "mlp_cond_head.cond_proj.0.weight", dxd_shape, 2, (float **)&lw->mlp_cond_s_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "mlp_cond_head.cond_proj.1.weight", dxd_shape, 2, (float **)&lw->mlp_cond_b_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "mlp_cond_head.cond_proj.2.weight", dxd_shape, 2, (float **)&lw->mlp_cond_g_weight)) goto cleanup;
            if (load_optional_layer_f32(&st, layer, "ctrl_mlpfusion.fc1_x.weight", dxd_shape, 2, (float **)&lw->ctrl_fc1_x_weight)) goto cleanup;
            lw->has_ctrl = lw->ctrl_fc1_x_weight != NULL;
            if (lw->has_ctrl && load_layer_f32(&st, layer, "ctrl_mlpfusion.fc1_c.weight", dxd_shape, 2, (float **)&lw->ctrl_fc1_c_weight)) goto cleanup;
            if (lw->has_ctrl && load_layer_f32(&st, layer, "ctrl_mlpfusion.fc2.weight", dxd_shape, 2, (float **)&lw->ctrl_fc2_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "dit_mlp.fc1.weight", hidden_d_shape, 2, (float **)&lw->dit_mlp_fc1_weight)) goto cleanup;
            if (load_layer_f32(&st, layer, "dit_mlp.fc2.weight", denoise_fc2_shape, 2, (float **)&lw->dit_mlp_fc2_weight)) goto cleanup;
        }
        WorldModelWeights model = {
            patchify_weight,
            denoise_fc1_weight,
            denoise_fc2_weight,
            ctrl_emb_fc1_weight,
            ctrl_emb_fc2_weight,
            control_inputs,
            initial_latents,
            layers,
            layers_to_run,
            out_norm_fc_weight,
            unpatchify_weight,
            unpatchify_bias,
        };
        rc = world_cuda_transformer_probe(&cfg, &model, layers_to_run, steps_to_run, frames_to_run, frame_idx, cache_pass, sigma, seed, noise_mode, dump_prefix, out_path ? &vae : NULL, out_path);
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
    if (load_required_f32(&st, "transformer.blocks.0.ctrl_mlpfusion.fc1_c.weight", dxd_shape, 2, &layer0_ctrl_fc1_c_weight)) {
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
        ctrl_emb_fc1_weight,
        ctrl_emb_fc2_weight,
        control_inputs,
        initial_latents,
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
        layer0_ctrl_fc1_c_weight,
        layer0_ctrl_fc2_weight,
        layer0_dit_mlp_fc1_weight,
        layer0_dit_mlp_fc2_weight,
    };
    rc = world_cuda_layer0_probe(&cfg, &layer0, sigma, seed, noise_mode, dump_prefix);

cleanup:
    free(patchify_weight);
    free(denoise_fc1_weight);
    free(denoise_fc2_weight);
    free(ctrl_emb_fc1_weight);
    free(ctrl_emb_fc2_weight);
    free(control_inputs);
    free(initial_latents);
    free(out_norm_fc_weight);
    free(unpatchify_weight);
    free(unpatchify_bias);
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
    free(layer0_ctrl_fc1_c_weight);
    free(layer0_ctrl_fc2_weight);
    free(layer0_dit_mlp_fc1_weight);
    free(layer0_dit_mlp_fc2_weight);
    world_free_layer_weights(layers, layers_to_run);
    world_free_vae_decoder_weights(&vae);
    safetensors_close(&st);
    return rc;
}
