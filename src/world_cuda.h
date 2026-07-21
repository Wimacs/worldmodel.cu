#ifndef WORLD_CUDA_H
#define WORLD_CUDA_H

#include "world_model.h"

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct WorldCudaRuntime WorldCudaRuntime;

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

int world_cuda_vae_decode_probe(
        const WorldConfig *cfg,
        const float *latent,
        const WorldVaeDecoderWeights *vae,
        const char *out_path);

int world_cuda_vae_decode_sequence_probe(
        const WorldConfig *cfg,
        const float *latents,
        int latent_count,
        const WorldVaeDecoderWeights *vae,
        const char *out_path);

int world_cuda_runtime_create(
        WorldCudaRuntime **out,
        const WorldConfig *cfg,
        const WorldModelWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frame_idx,
        unsigned int seed,
        int noise_mode,
        const WorldVaeDecoderWeights *vae);

int world_cuda_runtime_step_rgb(
        WorldCudaRuntime *rt,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out);

int world_cuda_runtime_seed_latent_rgb(
        WorldCudaRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out);

int world_cuda_runtime_init_vae_encoder(
        WorldCudaRuntime *rt,
        const WorldVaeEncoderWeights *encoder);

int world_cuda_runtime_encode_image_rgb(
        WorldCudaRuntime *rt,
        const float *rgb,
        int width,
        int height,
        float *latent_out,
        float *seconds_out);

int world_cuda_runtime_reset(WorldCudaRuntime *rt, int frame_idx, unsigned int seed);

void world_cuda_runtime_destroy(WorldCudaRuntime *rt);

#ifdef __cplusplus
}
#endif

#endif
