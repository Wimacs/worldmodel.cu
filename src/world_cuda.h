#ifndef WORLD_CUDA_H
#define WORLD_CUDA_H

#include "world_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WORLD_NOISE_UNIFORM 0
#define WORLD_NOISE_NORMAL 1

#define WORLD_VAE_DECODER_CONV_COUNT 35

enum {
    WORLD_VAE_DEC_CONV_IN = 0,
    WORLD_VAE_DEC_MB3_0,
    WORLD_VAE_DEC_MB3_2,
    WORLD_VAE_DEC_MB3_4,
    WORLD_VAE_DEC_MB4_0,
    WORLD_VAE_DEC_MB4_2,
    WORLD_VAE_DEC_MB4_4,
    WORLD_VAE_DEC_MB5_0,
    WORLD_VAE_DEC_MB5_2,
    WORLD_VAE_DEC_MB5_4,
    WORLD_VAE_DEC_TGROW7,
    WORLD_VAE_DEC_CONV8,
    WORLD_VAE_DEC_MB9_0,
    WORLD_VAE_DEC_MB9_2,
    WORLD_VAE_DEC_MB9_4,
    WORLD_VAE_DEC_MB10_0,
    WORLD_VAE_DEC_MB10_2,
    WORLD_VAE_DEC_MB10_4,
    WORLD_VAE_DEC_MB11_0,
    WORLD_VAE_DEC_MB11_2,
    WORLD_VAE_DEC_MB11_4,
    WORLD_VAE_DEC_TGROW13,
    WORLD_VAE_DEC_CONV14,
    WORLD_VAE_DEC_MB15_0,
    WORLD_VAE_DEC_MB15_2,
    WORLD_VAE_DEC_MB15_4,
    WORLD_VAE_DEC_MB16_0,
    WORLD_VAE_DEC_MB16_2,
    WORLD_VAE_DEC_MB16_4,
    WORLD_VAE_DEC_MB17_0,
    WORLD_VAE_DEC_MB17_2,
    WORLD_VAE_DEC_MB17_4,
    WORLD_VAE_DEC_TGROW19,
    WORLD_VAE_DEC_CONV20,
    WORLD_VAE_DEC_CONV_OUT,
};

typedef struct {
    const float *weight;
    const float *bias;
    int out_c;
    int in_c;
    int kernel;
    int has_bias;
} WorldVaeConvWeight;

typedef struct {
    WorldVaeConvWeight convs[WORLD_VAE_DECODER_CONV_COUNT];
} WorldVaeDecoderWeights;

typedef struct {
    const float *patchify_weight;
    const float *denoise_fc1_weight;
    const float *denoise_fc2_weight;
    const float *ctrl_emb_fc1_weight;
    const float *ctrl_emb_fc2_weight;
    const float *control_input;
    const float *initial_latent;
    const float *layer0_cond_bias;
    const float *layer0_attn_cond_s_weight;
    const float *layer0_attn_cond_b_weight;
    const float *layer0_attn_cond_g_weight;
    const float *layer0_q_proj_weight;
    const float *layer0_k_proj_weight;
    const float *layer0_v_proj_weight;
    const float *layer0_out_proj_weight;
    const float *layer0_mlp_cond_s_weight;
    const float *layer0_mlp_cond_b_weight;
    const float *layer0_mlp_cond_g_weight;
    const float *layer0_ctrl_fc1_x_weight;
    const float *layer0_ctrl_fc1_c_weight;
    const float *layer0_ctrl_fc2_weight;
    const float *layer0_dit_mlp_fc1_weight;
    const float *layer0_dit_mlp_fc2_weight;
} WorldLayer0ProbeWeights;

typedef struct {
    const float *cond_bias;
    const float *attn_cond_s_weight;
    const float *attn_cond_b_weight;
    const float *attn_cond_g_weight;
    const float *q_proj_weight;
    const float *k_proj_weight;
    const float *v_proj_weight;
    const float *out_proj_weight;
    const float *v_lamb;
    const float *mlp_cond_s_weight;
    const float *mlp_cond_b_weight;
    const float *mlp_cond_g_weight;
    const float *ctrl_fc1_x_weight;
    const float *ctrl_fc1_c_weight;
    const float *ctrl_fc2_weight;
    const float *dit_mlp_fc1_weight;
    const float *dit_mlp_fc2_weight;
    int has_ctrl;
} WorldLayerWeights;

typedef struct {
    const float *patchify_weight;
    const float *denoise_fc1_weight;
    const float *denoise_fc2_weight;
    const float *ctrl_emb_fc1_weight;
    const float *ctrl_emb_fc2_weight;
    const float *control_inputs;
    const float *initial_latents;
    const WorldLayerWeights *layers;
    int n_layers;
    const float *out_norm_fc_weight;
    const float *unpatchify_weight;
    const float *unpatchify_bias;
} WorldModelProbeWeights;

int world_cuda_generation_probe(
        const WorldConfig *cfg,
        const float *patchify_weight,
        const float *q_proj_weight,
        unsigned int seed);

int world_cuda_layer0_probe(
        const WorldConfig *cfg,
        const WorldLayer0ProbeWeights *weights,
        float sigma,
        unsigned int seed,
        int noise_mode,
        const char *dump_prefix);

int world_cuda_transformer_probe(
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
        const char *out_path);

#ifdef __cplusplus
}
#endif

#endif
