#ifndef WORLD_VULKAN_H
#define WORLD_VULKAN_H

#include "world_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WorldVulkanRuntime WorldVulkanRuntime;

int world_vulkan_runtime_create(
        WorldVulkanRuntime **out,
        const WorldConfig *cfg,
        const WorldModelProbeWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frame_idx,
        unsigned int seed,
        int noise_mode,
        const WorldVaeDecoderWeights *vae);

int world_vulkan_runtime_step_rgb(
        WorldVulkanRuntime *rt,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out);

int world_vulkan_runtime_seed_latent_rgb(
        WorldVulkanRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out);

void world_vulkan_runtime_destroy(WorldVulkanRuntime *rt);

int world_vulkan_linear_f32_probe(void);
int world_vulkan_silu_f32_probe(void);
int world_vulkan_add_bias_silu_f32_probe(void);
int world_vulkan_rms_norm_f32_probe(void);
int world_vulkan_control_embedding_f32_probe(void);
int world_vulkan_denoise_out_norm_f32_probe(void);
int world_vulkan_ada_rms_norm_f32_probe(void);
int world_vulkan_ortho_rope_f32_probe(void);
int world_vulkan_qkv_rms_rope_f32_probe(void);
int world_vulkan_runtime_layer0_qkv_f32_probe(void);
int world_vulkan_masked_attention_f32_probe(void);
int world_vulkan_gated_residual_add_f32_probe(void);
int world_vulkan_kv_cache_upsert_f32_probe(void);
int world_vulkan_cache_frame_indices_probe(void);
int world_vulkan_patchify_f32_probe(void);
int world_vulkan_unpatchify_f32_probe(void);
int world_vulkan_indexed_attention_f32_probe(void);

#ifdef __cplusplus
}
#endif

#endif
