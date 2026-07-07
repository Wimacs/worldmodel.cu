#ifndef WORLD_CUDA_H
#define WORLD_CUDA_H

#include "world_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *patchify_weight;
    const float *denoise_fc1_weight;
    const float *denoise_fc2_weight;
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
    const float *ctrl_fc2_weight;
    const float *dit_mlp_fc1_weight;
    const float *dit_mlp_fc2_weight;
    int has_ctrl;
} WorldLayerWeights;

typedef struct {
    const float *patchify_weight;
    const float *denoise_fc1_weight;
    const float *denoise_fc2_weight;
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
        const char *dump_prefix);

int world_cuda_transformer_probe(
        const WorldConfig *cfg,
        const WorldModelProbeWeights *weights,
        int layers_to_run,
        int steps_to_run,
        float sigma,
        unsigned int seed,
        const char *dump_prefix);

#ifdef __cplusplus
}
#endif

#endif
