#ifndef WORLD_CUDA_INTERNAL_CUH
#define WORLD_CUDA_INTERNAL_CUH

#include "world_model.h"

#include <cuda_fp16.h>

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float *cond_bias;
    float *cond_proj_weight;
    float *qkv_proj_weight;
    __half *qkv_proj_weight_h;
    int8_t *qkv_proj_weight_i8;
    float *qkv_proj_weight_i8_scales;
    float *out_proj_weight;
    __half *out_proj_weight_h;
    int8_t *out_proj_weight_i8;
    float *out_proj_weight_i8_scales;
    float v_lamb;
    float *ctrl_fc1_x_weight;
    __half *ctrl_fc1_x_weight_h;
    int8_t *ctrl_fc1_x_weight_i8;
    float *ctrl_fc1_x_weight_i8_scales;
    float *ctrl_fc1_c_weight;
    float *ctrl_fc2_weight;
    __half *ctrl_fc2_weight_h;
    int8_t *ctrl_fc2_weight_i8;
    float *ctrl_fc2_weight_i8_scales;
    float *dit_mlp_fc1_weight;
    __half *dit_mlp_fc1_weight_h;
    int8_t *dit_mlp_fc1_weight_i8;
    float *dit_mlp_fc1_weight_i8_scales;
    float *dit_mlp_fc2_weight;
    __half *dit_mlp_fc2_weight_h;
    int8_t *dit_mlp_fc2_weight_i8;
    float *dit_mlp_fc2_weight_i8_scales;
    int has_ctrl;
} DeviceWorldLayerWeights;

typedef struct {
    float *k;
    float *v;
    __half *k_h;
    __half *v_h;
    bool *written;
    unsigned char *h_slot_written;
    int64_t *indices;
    int32_t *block_ids;
    int *index_count;
    int ring_length;
    int capacity;
    int slot_count;
    int pinned_dilation;
    int is_global;
} DeviceWorldLayerCache;

void wm_cuda_fill_latent(float *x, int n, unsigned int seed, int noise_mode);
void wm_cuda_fill_noise_embedding(float *emb, float sigma);
void wm_cuda_fill_positions(
        int64_t *x_pos,
        int64_t *y_pos,
        int64_t *t_pos,
        int T,
        int width,
        int frame_timestamp);
void wm_cuda_fill_rope_tables(
        float *xy,
        float *inv_t,
        int d_head,
        int height,
        int width);

int wm_cuda_copy_f32_to_device(float **dst, const float *src, size_t n);
int wm_cuda_copy_world_layers_to_device(
        DeviceWorldLayerWeights **dst_layers,
        const WorldLayerWeights *src_layers,
        int n_layers,
        int D,
        int kv_dim,
        int mlp_hidden,
        int w8a8_drop_fallback,
        int w8a8_mask,
        int w8a8_layer_begin,
        int w8a8_layer_end);
void wm_cuda_free_device_world_layers(DeviceWorldLayerWeights *layers, int n_layers);

int wm_cuda_alloc_device_world_caches(
        DeviceWorldLayerCache **dst_caches,
        const WorldConfig *cfg,
        int n_layers,
        int T,
        int n_kv_heads,
        int d_head,
        int alloc_half_cache);
void wm_cuda_free_device_world_caches(DeviceWorldLayerCache *caches, int n_layers);

#endif
