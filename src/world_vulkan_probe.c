#include "world_vulkan.h"

#include <stdio.h>

int world_vulkan_linear_f32_probe(void);
int world_vulkan_silu_f32_probe(void);
int world_vulkan_rms_norm_f32_probe(void);
int world_vulkan_ada_rms_norm_f32_probe(void);
int world_vulkan_ortho_rope_f32_probe(void);
int world_vulkan_qkv_rms_rope_f32_probe(void);
int world_vulkan_masked_attention_f32_probe(void);
int world_vulkan_kv_cache_upsert_f32_probe(void);
int world_vulkan_cache_frame_indices_probe(void);
int world_vulkan_patchify_f32_probe(void);
int world_vulkan_unpatchify_f32_probe(void);

int main(void) {
    if (world_vulkan_linear_f32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_probe: ok\n");
    if (world_vulkan_silu_f32_probe()) {
        fprintf(stderr, "world_vulkan_silu_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_silu_f32_probe: ok\n");
    if (world_vulkan_rms_norm_f32_probe()) {
        fprintf(stderr, "world_vulkan_rms_norm_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_rms_norm_f32_probe: ok\n");
    if (world_vulkan_ada_rms_norm_f32_probe()) {
        fprintf(stderr, "world_vulkan_ada_rms_norm_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_ada_rms_norm_f32_probe: ok\n");
    if (world_vulkan_ortho_rope_f32_probe()) {
        fprintf(stderr, "world_vulkan_ortho_rope_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_ortho_rope_f32_probe: ok\n");
    if (world_vulkan_qkv_rms_rope_f32_probe()) {
        fprintf(stderr, "world_vulkan_qkv_rms_rope_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_qkv_rms_rope_f32_probe: ok\n");
    if (world_vulkan_masked_attention_f32_probe()) {
        fprintf(stderr, "world_vulkan_masked_attention_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_masked_attention_f32_probe: ok\n");
    if (world_vulkan_kv_cache_upsert_f32_probe()) {
        fprintf(stderr, "world_vulkan_kv_cache_upsert_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_kv_cache_upsert_f32_probe: ok\n");
    if (world_vulkan_cache_frame_indices_probe()) {
        fprintf(stderr, "world_vulkan_cache_frame_indices_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_cache_frame_indices_probe: ok\n");
    if (world_vulkan_patchify_f32_probe()) {
        fprintf(stderr, "world_vulkan_patchify_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_patchify_f32_probe: ok\n");
    if (world_vulkan_unpatchify_f32_probe()) {
        fprintf(stderr, "world_vulkan_unpatchify_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_unpatchify_f32_probe: ok\n");
    return 0;
}
