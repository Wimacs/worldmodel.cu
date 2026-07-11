#include "world_vulkan.h"

#include <stdio.h>
#include <string.h>

int world_vulkan_linear_f32_probe(void);
int world_vulkan_linear_f16_coopmat_probe(void);
int world_vulkan_linear_f32_coopmat_probe(void);
int world_vulkan_linear_f32_wf16_coopmat_probe(void);
int world_vulkan_linear_f32_wf16_gated_residual_n32_probe(void);
int world_vulkan_linear_f16x_wf16_coopmat_probe(void);
int world_vulkan_linear_f16x_wf16_silu_f16_n32_probe(void);
int world_vulkan_linear_f16x_wf16_gated_residual_probe(void);
int world_vulkan_linear_f16x_wf16_gated_residual_n16_probe(void);
int world_vulkan_linear_f16x_wf16_gated_residual_n32_probe(void);
int world_vulkan_linear_f32_wf16_silu_f16_n32_probe(void);
int world_vulkan_silu_f32_probe(void);
int world_vulkan_silu_f32_to_f16_probe(void);
int world_vulkan_add_bias_silu_f32_probe(void);
int world_vulkan_add_channel_silu_f32_probe(void);
int world_vulkan_add_f32_probe(void);
int world_vulkan_out_norm_silu_f32_probe(void);
int world_vulkan_latent_update_f32_probe(void);
int world_vulkan_lerp_inplace_f32_probe(void);
int world_vulkan_taehv_primitives_probe(void);
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
int world_vulkan_sparse_gqa_fmha_probe(void);
int world_vulkan_half_boundary_ops_probe(void);

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--taehv") == 0) {
        if (world_vulkan_taehv_primitives_probe()) {
            fprintf(stderr, "world_vulkan_taehv_primitives_probe: failed\n");
            return 1;
        }
        fprintf(stderr, "world_vulkan_taehv_primitives_probe: ok\n");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--sparse-fmha") == 0) {
        if (world_vulkan_sparse_gqa_fmha_probe()) {
            fprintf(stderr, "world_vulkan_sparse_gqa_fmha_probe: failed\n");
            return 1;
        }
        fprintf(stderr, "world_vulkan_sparse_gqa_fmha_probe: ok\n");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--cache-indices") == 0) {
        if (world_vulkan_cache_frame_indices_probe()) {
            fprintf(stderr, "world_vulkan_cache_frame_indices_probe: failed\n");
            return 1;
        }
        fprintf(stderr, "world_vulkan_cache_frame_indices_probe: ok\n");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--half-boundary") == 0) {
        if (world_vulkan_half_boundary_ops_probe()) {
            fprintf(stderr, "world_vulkan_half_boundary_ops_probe: failed\n");
            return 1;
        }
        fprintf(stderr, "world_vulkan_half_boundary_ops_probe: ok\n");
        return 0;
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [--taehv|--sparse-fmha|--cache-indices|--half-boundary]\n", argv[0]);
        return 2;
    }
    if (world_vulkan_linear_f32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_probe: ok\n");
    if (world_vulkan_linear_f16_coopmat_probe()) {
        fprintf(stderr, "world_vulkan_linear_f16_coopmat_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f16_coopmat_probe: ok\n");
    if (world_vulkan_linear_f32_coopmat_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_coopmat_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_coopmat_probe: ok\n");
    if (world_vulkan_linear_f32_wf16_coopmat_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_wf16_coopmat_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_wf16_coopmat_probe: ok\n");
    if (world_vulkan_linear_f32_wf16_gated_residual_n32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_wf16_gated_residual_n32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_wf16_gated_residual_n32_probe: ok\n");
    if (world_vulkan_linear_f16x_wf16_coopmat_probe()) {
        fprintf(stderr, "world_vulkan_linear_f16x_wf16_coopmat_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f16x_wf16_coopmat_probe: ok\n");
    if (world_vulkan_linear_f16x_wf16_silu_f16_n32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f16x_wf16_silu_f16_n32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f16x_wf16_silu_f16_n32_probe: ok\n");
    if (world_vulkan_linear_f16x_wf16_gated_residual_probe()) {
        fprintf(stderr, "world_vulkan_linear_f16x_wf16_gated_residual_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f16x_wf16_gated_residual_probe: ok\n");
    if (world_vulkan_linear_f16x_wf16_gated_residual_n16_probe()) {
        fprintf(stderr, "world_vulkan_linear_f16x_wf16_gated_residual_n16_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f16x_wf16_gated_residual_n16_probe: ok\n");
    if (world_vulkan_linear_f16x_wf16_gated_residual_n32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f16x_wf16_gated_residual_n32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f16x_wf16_gated_residual_n32_probe: ok\n");
    if (world_vulkan_linear_f32_wf16_silu_f16_n32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_wf16_silu_f16_n32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_wf16_silu_f16_n32_probe: ok\n");
    if (world_vulkan_silu_f32_probe()) {
        fprintf(stderr, "world_vulkan_silu_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_silu_f32_probe: ok\n");
    if (world_vulkan_silu_f32_to_f16_probe()) {
        fprintf(stderr, "world_vulkan_silu_f32_to_f16_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_silu_f32_to_f16_probe: ok\n");
    if (world_vulkan_add_bias_silu_f32_probe()) {
        fprintf(stderr, "world_vulkan_add_bias_silu_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_add_bias_silu_f32_probe: ok\n");
    if (world_vulkan_add_channel_silu_f32_probe()) {
        fprintf(stderr, "world_vulkan_add_channel_silu_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_add_channel_silu_f32_probe: ok\n");
    if (world_vulkan_add_f32_probe()) {
        fprintf(stderr, "world_vulkan_add_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_add_f32_probe: ok\n");
    if (world_vulkan_out_norm_silu_f32_probe()) {
        fprintf(stderr, "world_vulkan_out_norm_silu_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_out_norm_silu_f32_probe: ok\n");
    if (world_vulkan_latent_update_f32_probe()) {
        fprintf(stderr, "world_vulkan_latent_update_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_latent_update_f32_probe: ok\n");
    if (world_vulkan_lerp_inplace_f32_probe()) {
        fprintf(stderr, "world_vulkan_lerp_inplace_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_lerp_inplace_f32_probe: ok\n");
    if (world_vulkan_taehv_primitives_probe()) {
        fprintf(stderr, "world_vulkan_taehv_primitives_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_taehv_primitives_probe: ok\n");
    if (world_vulkan_rms_norm_f32_probe()) {
        fprintf(stderr, "world_vulkan_rms_norm_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_rms_norm_f32_probe: ok\n");
    if (world_vulkan_control_embedding_f32_probe()) {
        fprintf(stderr, "world_vulkan_control_embedding_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_control_embedding_f32_probe: ok\n");
    if (world_vulkan_denoise_out_norm_f32_probe()) {
        fprintf(stderr, "world_vulkan_denoise_out_norm_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_denoise_out_norm_f32_probe: ok\n");
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
    if (world_vulkan_indexed_attention_f32_probe()) {
        fprintf(stderr, "world_vulkan_indexed_attention_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_indexed_attention_f32_probe: ok\n");
    if (world_vulkan_runtime_layer0_qkv_f32_probe()) {
        fprintf(stderr, "world_vulkan_runtime_layer0_qkv_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_runtime_layer0_qkv_f32_probe: ok\n");
    if (world_vulkan_masked_attention_f32_probe()) {
        fprintf(stderr, "world_vulkan_masked_attention_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_masked_attention_f32_probe: ok\n");
    if (world_vulkan_gated_residual_add_f32_probe()) {
        fprintf(stderr, "world_vulkan_gated_residual_add_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_gated_residual_add_f32_probe: ok\n");
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
