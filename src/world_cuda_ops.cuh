#ifndef WORLD_CUDA_OPS_CUH
#define WORLD_CUDA_OPS_CUH

#include <cuda_fp16.h>

#include <stddef.h>
#include <stdint.h>

// Backend-private CUDA operator entry points.  These functions launch work
// but do not own model state, weights, caches, or scheduling policy.
int wm_cuda_silu_f32(const float *x, float *y, int64_t n);
int wm_cuda_f32_to_f16(const float *x, __half *y, int64_t n);
int wm_cuda_silu_f32_to_f16(const float *x, __half *y, int64_t n);
int wm_cuda_add_bias_silu_f32(const float *x, const float *bias, float *y, int64_t n);
int wm_cuda_add_channel_silu_inplace_f32(float *x, const float *bias, int rows, int d);

int wm_cuda_quantize_rows_f32_i8(
        const float *x, int8_t *q, float *scales, int rows, int cols);
int wm_cuda_rms_norm_quantize_rows_i8(
        const float *x,
        const float *mod_scale,
        const float *bias,
        int8_t *q,
        float *q_scales,
        int rows,
        int cols,
        float eps);
int wm_cuda_dequant_silu_quantize_rows_i8(
        const int32_t *acc,
        const float *input_row_scales,
        const float *weight_scales,
        const float *bias,
        int8_t *q,
        float *output_row_scales,
        int rows,
        int cols);
int wm_cuda_dequant_gated_residual_f32(
        const int32_t *acc,
        const float *row_scales,
        const float *col_scales,
        const float *residual,
        const float *gate,
        float *out,
        int rows,
        int cols);
int wm_cuda_dequant_add_residual_f32(
        const int32_t *acc,
        const float *row_scales,
        const float *col_scales,
        const float *residual,
        float *out,
        int rows,
        int cols);

int wm_cuda_ada_rms_norm_f32(
        const float *x,
        const float *scale,
        const float *bias,
        float *y,
        int rows,
        int d,
        float eps);
int wm_cuda_ada_rms_norm_f16(
        const float *x,
        const float *scale,
        const float *bias,
        __half *y,
        int rows,
        int d,
        float eps);
int wm_cuda_rms_norm_rows_f32(
        const float *x, float *y, int rows, int d, float eps);
int wm_cuda_out_norm_silu_f32(
        const float *tokens,
        const float *mod,
        float *out,
        int rows,
        int d,
        float eps);

int wm_cuda_qkv_separate_rms_rope_f32(
        const float *q_raw,
        const float *k_raw,
        const float *v_raw,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int tokens,
        int n_heads,
        int n_kv_heads,
        int d_head,
        int width,
        int height,
        float eps);
int wm_cuda_qkv_fused_rms_rope_f32(
        const float *qkv_raw,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int tokens,
        int n_heads,
        int n_kv_heads,
        int d_head,
        int width,
        int height,
        float eps);
int wm_cuda_qkv_fused_rms_rope_i32_dequant(
        const int32_t *qkv_acc,
        const float *row_scales,
        const float *weight_scales,
        float *q,
        float *k,
        float *v,
        const int64_t *x_pos,
        const int64_t *y_pos,
        const int64_t *t_pos,
        const float *xy,
        const float *inv_t,
        int tokens,
        int n_heads,
        int n_kv_heads,
        int d_head,
        int width,
        int height,
        float eps);

int wm_cuda_current_frame_attention_f32(
        const float *q,
        const float *k,
        const float *v,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int d_head,
        float scale);
int wm_cuda_init_cache_written(bool *written, int ring_length, int tokens);
int wm_cuda_kv_cache_upsert_copy_f32(
        float *cache_k,
        float *cache_v,
        const float *k,
        const float *v,
        bool *written,
        int n_kv_heads,
        int tokens,
        int d_head,
        int ring_length,
        int base,
        bool write_step,
        bool frozen);
int wm_cuda_kv_cache_upsert_copy_f16(
        __half *cache_k,
        __half *cache_v,
        const float *k,
        const float *v,
        bool *written,
        int n_kv_heads,
        int tokens,
        int d_head,
        int ring_length,
        int base,
        bool write_step,
        bool frozen);
int wm_cuda_collect_cache_frame_indices(
        const bool *written,
        int64_t *indices,
        int32_t *block_ids,
        int *count,
        int capacity,
        int tokens,
        int base,
        bool write_step);

int wm_cuda_indexed_attention_f32(
        const float *q,
        const float *cache_k,
        const float *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        int d_head,
        float scale);
int wm_cuda_indexed_attention_d64_warp_f32(
        const float *q,
        const float *cache_k,
        const float *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_indexed_attention_d64_warp_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_indexed_attention_d64_flash_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_indexed_attention_d64_flash_f32(
        const float *q,
        const float *cache_k,
        const float *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_indexed_attention_d64_q4_shared_f32(
        const float *q,
        const float *cache_k,
        const float *cache_v,
        const int64_t *indices,
        const int *index_count,
        float *out_tokens,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);

int wm_cuda_gated_residual_add_f32(
        const float *residual,
        const float *update,
        const float *gate,
        float *out,
        int tokens,
        int d);
int wm_cuda_add_f32(const float *a, const float *b, float *out, int64_t n);
int wm_cuda_latent_update_f32(float *latent, const float *velocity, float dsigma, int64_t n);
int wm_cuda_lerp_inplace_f32(float *x, const float *end, float weight, int64_t n);
int wm_cuda_patchify_f32(
        const float *x,
        const float *weight,
        float *tokens,
        int channels,
        int height,
        int width,
        int d,
        int patch_h,
        int patch_w,
        int token_h,
        int token_w);
int wm_cuda_patchify_im2row_f32(
        const float *x,
        float *rows,
        int channels,
        int height,
        int width,
        int patch_h,
        int patch_w,
        int token_h,
        int token_w);
int wm_cuda_unpatchify_f32(
        const float *tokens,
        const float *weight,
        const float *bias,
        float *x,
        int token_count,
        int d,
        int channels,
        int height,
        int width,
        int patch_h,
        int patch_w,
        int token_w,
        int out_dim);

int wm_cuda_linear_f32(
        const float *x_rm,
        const float *w_rm,
        float *y_rm,
        int m,
        int k,
        int n);
int wm_cuda_gemm_i8_i32_can_implement(
        const int8_t *x_i8,
        const int8_t *w_rm_i8,
        int32_t *acc_i32,
        int m,
        int k,
        int n);
int wm_cuda_gemm_i8_i32(
        const int8_t *x_i8,
        const int8_t *w_rm_i8,
        int32_t *acc_i32,
        int m,
        int k,
        int n);
int wm_cuda_linear_fp16_weight_simt(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n);
int wm_cuda_linear_fp16_weight_tensorop(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n);
int wm_cuda_linear_fp16_weight_tensorop_m64n64(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n);
int wm_cuda_linear_fp16_input_weight_tensorop_m64n64(
        const __half *x_rm_h,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n);
int wm_cuda_linear_fp16_weight_tensorop_m64n64_silu_half(
        const float *x_rm,
        __half *x_half_tmp,
        const __half *w_rm_h,
        __half *y_rm_h,
        int m,
        int k,
        int n);
int wm_cuda_linear_fp16_input_weight_tensorop_m64n64_silu_half(
        const __half *x_rm_h,
        const __half *w_rm_h,
        __half *y_rm_h,
        int m,
        int k,
        int n);
int wm_cuda_should_use_m64n64_tensorop(int enabled, int m, int k, int n);
size_t wm_cuda_linear_fp16_weight_tensorop_splitk_workspace_size(
        int m, int k, int n, int split_k_slices);
int wm_cuda_linear_fp16_input_weight_tensorop_splitk(
        const __half *x_rm_h,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n,
        int split_k_slices,
        void *workspace,
        size_t workspace_bytes);
size_t wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel_workspace_size(
        int m, int k, int n, int split_k_slices);
int wm_cuda_linear_fp16_input_weight_tensorop_splitk_parallel(
        const __half *x_rm_h,
        const __half *w_rm_h,
        float *y_rm,
        int m,
        int k,
        int n,
        int split_k_slices,
        void *workspace,
        size_t workspace_bytes);

int wm_cuda_has_cutlass_fmha(void);
int wm_cuda_attention_d64_cutlass_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        int index_count,
        float *out_tokens,
        __half *q_half,
        __half *k_compact,
        __half *v_compact,
        float *scores,
        __half *probs_half,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_attention_d64_fmha_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        int index_count,
        float *out_tokens,
        __half *out_tokens_half,
        int output_half,
        __half *q_bmhd,
        __half *k_bnhd,
        __half *v_bnhd,
        __half *out_bmhd,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_attention_d64_sparse_fmha_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int32_t *block_ids,
        int block_count,
        float *out_tokens,
        __half *out_tokens_half,
        int output_half,
        __half *q_bmhd,
        __half *out_bmhd,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);
int wm_cuda_attention_d64_cutlass_grouped_f16_kv(
        const float *q,
        const __half *cache_k,
        const __half *cache_v,
        const int64_t *indices,
        int index_count,
        float *out_tokens,
        __half *q_half,
        __half *k_compact,
        __half *v_compact,
        float *scores,
        __half *probs_half,
        int n_heads,
        int n_kv_heads,
        int tokens,
        int capacity,
        float scale);

#endif
