#include "world_vulkan.h"

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef WORLD_VULKAN_SHADER_DIR
#define WORLD_VULKAN_SHADER_DIR "shaders/vulkan"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WORLD_VULKAN_MAX_PASSES 32
#define WORLD_VULKAN_MAX_VAE_OPS 128

enum {
    WORLD_VULKAN_VAE_REPEAT_CLAMP = 0,
    WORLD_VULKAN_VAE_CONV,
    WORLD_VULKAN_VAE_RELU,
    WORLD_VULKAN_VAE_CONCAT_PAST,
    WORLD_VULKAN_VAE_ADD_RELU,
    WORLD_VULKAN_VAE_UPSAMPLE2,
    WORLD_VULKAN_VAE_TGROW,
    WORLD_VULKAN_VAE_PIXEL_SHUFFLE,
    WORLD_VULKAN_VAE_CONV_OUT_RGBA,
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t frames;
    uint32_t frame_ordinal;
    float control_x;
    float control_y;
} WorldVulkanFillPush;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t inner;
    uint32_t has_bias;
} WorldVulkanLinearPush;

typedef struct {
    uint32_t n;
} WorldVulkanSiluPush;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} WorldVulkanRmsNormPush;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t N;
    uint32_t D;
    float eps;
} WorldVulkanAdaRmsNormPush;

typedef struct {
    uint32_t B;
    uint32_t H;
    uint32_t T;
    uint32_t D;
    uint32_t width;
    uint32_t height;
} WorldVulkanOrthoRopePush;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t D;
    uint32_t width;
    uint32_t height;
    float eps;
} WorldVulkanQkvRmsRopePush;

typedef struct {
    uint32_t B;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t Tq;
    uint32_t Tk;
    uint32_t D;
    float scale;
} WorldVulkanMaskedAttentionPush;

typedef struct {
    uint32_t capacity;
    uint32_t T;
    uint32_t base;
    uint32_t write_step;
} WorldVulkanKvCacheMaskPush;

typedef struct {
    uint32_t B;
    uint32_t H;
    uint32_t T;
    uint32_t D;
    uint32_t L;
    uint32_t base;
    uint32_t write_step;
    uint32_t frozen;
} WorldVulkanKvCacheUpsertPush;

typedef struct {
    uint32_t capacity;
    uint32_t T;
    uint32_t base;
    uint32_t write_step;
} WorldVulkanCacheFrameIndicesPush;

typedef struct {
    uint32_t B;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t D;
    uint32_t ph;
    uint32_t pw;
    uint32_t Hp;
    uint32_t Wp;
} WorldVulkanPatchifyPush;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t D;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t ph;
    uint32_t pw;
    uint32_t Hp;
    uint32_t Wp;
    uint32_t out_dim;
} WorldVulkanUnpatchifyPush;

typedef struct {
    uint32_t T;
    uint32_t D;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t ph;
    uint32_t pw;
    uint32_t Wp;
    uint32_t out_dim;
} WorldVulkanUnpatchifyOrigPush;

typedef struct {
    uint32_t out_width;
    uint32_t out_height;
    uint32_t frames;
    uint32_t latent_c;
    uint32_t latent_h;
    uint32_t latent_w;
    uint32_t frame_ordinal;
    uint32_t ctrl_dim;
    float control_x;
    float control_y;
} WorldVulkanLatentRgbaPush;

typedef struct {
    uint32_t B;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t Tq;
    uint32_t Nkv;
    uint32_t Tk;
    uint32_t D;
    float scale;
} WorldVulkanIndexedAttentionPush;

typedef struct {
    uint32_t T;
    uint32_t D;
} WorldVulkanGatedResidualPush;

typedef struct {
    uint32_t rows;
    uint32_t D;
} WorldVulkanAddChannelSiluPush;

typedef struct {
    uint32_t rows;
    uint32_t D;
    float eps;
} WorldVulkanOutNormSiluPush;

typedef struct {
    uint32_t n;
    float dsigma;
} WorldVulkanLatentUpdatePush;

typedef struct {
    uint32_t n;
    float weight;
} WorldVulkanLerpPush;

typedef struct {
    uint32_t C;
    uint32_t H;
    uint32_t W;
} WorldVulkanTaehvRepeatClampPush;

typedef struct {
    uint32_t N;
    uint32_t C_in;
    uint32_t C_out;
    uint32_t H;
    uint32_t W;
    uint32_t K;
    uint32_t has_bias;
} WorldVulkanTaehvConv2dPush;

typedef struct {
    uint32_t N;
    uint32_t C;
    uint32_t H;
    uint32_t W;
} WorldVulkanTaehvNchwPush;

typedef struct {
    uint32_t N;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t stride;
} WorldVulkanTaehvTgrowPush;

typedef struct {
    uint32_t N;
    uint32_t H;
    uint32_t W;
} WorldVulkanTaehvPixelShufflePush;

typedef struct {
    int kind;
    int conv_idx;
    int src;
    int dst;
    int aux;
    int N;
    int C;
    int C_out;
    int H;
    int W;
    int stride;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
} WorldVulkanVaeOp;

typedef struct {
    const char *label;
    uint32_t begin_query;
    uint32_t end_query;
} WorldVulkanProfileEvent;

typedef struct {
    const char *label;
    double total_ms;
    double max_ms;
    uint32_t count;
} WorldVulkanProfileAgg;

static int vae_conv_uses_1x1_c4(const WorldVulkanVaeOp *op) {
    return op && op->kind == WORLD_VULKAN_VAE_CONV && op->stride == 1 && (op->C_out % 4) == 0;
}

struct WorldVulkanRuntime {
    WorldConfig cfg;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family;
    int shader_float16_enabled;
    int storage_16bit_enabled;
    int cooperative_matrix_enabled;
    uint32_t max_compute_shared_memory_size;
    VkQueue queue;
    VkShaderModule fill_shader;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline fill_pipeline;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkQueryPool profile_query_pool;
    uint64_t *profile_query_values;
    WorldVulkanProfileEvent *profile_events;
    uint32_t profile_max_events;
    uint32_t profile_event_count;
    uint32_t profile_timestamp_valid_bits;
    float profile_timestamp_period_ns;
    int profile_enabled;
    int profile_overflow_warned;
    int profile_top_n;
    VkBuffer output_buffer;
    VkDeviceMemory output_memory;
    VkBuffer output_readback_buffer;
    VkDeviceMemory output_readback_memory;
    void *output_mapped;
    unsigned char *rgb_host;
    size_t pixel_count;
    size_t rgb_bytes;
    int width;
    int height;
    int frames;
    int frame_ordinal;
    int layers_to_run;
    int steps_to_run;
    int total_passes;
    int precomputed_total_passes;
    int forced_pass_table_idx;
    unsigned int seed;
    int noise_mode;
    int model_slice_enabled;
    int C;
    int H;
    int W;
    int D;
    int ph;
    int pw;
    int T;
    int out_dim;
    int mlp_hidden;
    int ctrl_dim;
    int d_head;
    int kv_dim;
    int qkv_dim;
    int d_xy;
    int d_t;
    int frame_stride;
    int cache_ring_length;
    int cache_capacity;
    int cache_pinned_dilation;
    int *layer_has_ctrl;
    float *layer_v_lamb;
    int *cache_ring_lengths;
    int *cache_capacities;
    int *cache_pinned_dilations;
    int *cache_is_global;
    size_t *cache_kv_offsets;
    size_t *cache_meta_offsets;
    size_t cache_total_kv_elems;
    size_t cache_total_meta_elems;
    size_t latent_elems;
    size_t token_elems;
    int use_external_latent_once;
    float rms_eps;
    int ctrl_embedding_enabled;
    int denoise_out_norm_enabled;
    int layer_mod_enabled;
    int layer_qkv_enabled;
    int layer_attention_enabled;
    int layer_attn_out_enabled;
    int layer_ctrl_enabled;
    int layer_mlp_enabled;
    VkShaderModule runtime_linear_shader;
    VkDescriptorSetLayout runtime_linear_set_layout;
    VkPipelineLayout runtime_linear_pipeline_layout;
    VkPipeline runtime_linear_pipeline;
    VkShaderModule runtime_linear_coopmat_shader;
    VkPipelineLayout runtime_linear_coopmat_pipeline_layout;
    VkPipeline runtime_linear_coopmat_pipeline;
    int runtime_linear_coopmat_enabled;
    VkShaderModule runtime_linear_wf16_coopmat_shader;
    VkPipelineLayout runtime_linear_wf16_coopmat_pipeline_layout;
    VkPipeline runtime_linear_wf16_coopmat_pipeline;
    VkShaderModule runtime_linear_wf16_coopmat_n16_shader;
    VkPipelineLayout runtime_linear_wf16_coopmat_n16_pipeline_layout;
    VkPipeline runtime_linear_wf16_coopmat_n16_pipeline;
    VkShaderModule runtime_linear_f16x_wf16_coopmat_shader;
    VkPipelineLayout runtime_linear_f16x_wf16_coopmat_pipeline_layout;
    VkPipeline runtime_linear_f16x_wf16_coopmat_pipeline;
    int runtime_linear_wf16_coopmat_enabled;
    VkShaderModule runtime_linear_tile64_shader;
    VkPipelineLayout runtime_linear_tile64_pipeline_layout;
    VkPipeline runtime_linear_tile64_pipeline;
    int runtime_linear_tile64_enabled;
    VkShaderModule runtime_silu_shader;
    VkDescriptorSetLayout runtime_silu_set_layout;
    VkPipelineLayout runtime_silu_pipeline_layout;
    VkPipeline runtime_silu_pipeline;
    VkShaderModule runtime_silu_f16_shader;
    VkDescriptorSetLayout runtime_silu_f16_set_layout;
    VkPipelineLayout runtime_silu_f16_pipeline_layout;
    VkPipeline runtime_silu_f16_pipeline;
    VkShaderModule runtime_add_bias_silu_shader;
    VkDescriptorSetLayout runtime_add_bias_silu_set_layout;
    VkPipelineLayout runtime_add_bias_silu_pipeline_layout;
    VkPipeline runtime_add_bias_silu_pipeline;
    VkShaderModule runtime_rms_shader;
    VkDescriptorSetLayout runtime_rms_set_layout;
    VkPipelineLayout runtime_rms_pipeline_layout;
    VkPipeline runtime_rms_pipeline;
    VkShaderModule runtime_ada_rms_shader;
    VkDescriptorSetLayout runtime_ada_rms_set_layout;
    VkPipelineLayout runtime_ada_rms_pipeline_layout;
    VkPipeline runtime_ada_rms_pipeline;
    VkShaderModule runtime_qkv_rms_rope_shader;
    VkDescriptorSetLayout runtime_qkv_rms_rope_set_layout;
    VkPipelineLayout runtime_qkv_rms_rope_pipeline_layout;
    VkPipeline runtime_qkv_rms_rope_pipeline;
    VkShaderModule runtime_kv_upsert_shader;
    VkDescriptorSetLayout runtime_kv_upsert_set_layout;
    VkPipelineLayout runtime_kv_upsert_pipeline_layout;
    VkPipeline runtime_kv_upsert_pipeline;
    VkShaderModule runtime_cache_indices_shader;
    VkDescriptorSetLayout runtime_cache_indices_set_layout;
    VkPipelineLayout runtime_cache_indices_pipeline_layout;
    VkPipeline runtime_cache_indices_pipeline;
    VkShaderModule runtime_indexed_attention_shader;
    VkDescriptorSetLayout runtime_indexed_attention_set_layout;
    VkPipelineLayout runtime_indexed_attention_pipeline_layout;
    VkPipeline runtime_indexed_attention_pipeline;
    VkShaderModule runtime_indexed_attention_d64_shader;
    VkPipelineLayout runtime_indexed_attention_d64_pipeline_layout;
    VkPipeline runtime_indexed_attention_d64_pipeline;
    VkShaderModule runtime_indexed_attention_d64_flash_shader;
    VkPipelineLayout runtime_indexed_attention_d64_flash_pipeline_layout;
    VkPipeline runtime_indexed_attention_d64_flash_pipeline;
    int runtime_indexed_attention_d64_flash_enabled;
    VkShaderModule runtime_indexed_attention_d64_flash96_shader;
    VkPipelineLayout runtime_indexed_attention_d64_flash96_pipeline_layout;
    VkPipeline runtime_indexed_attention_d64_flash96_pipeline;
    int runtime_indexed_attention_d64_flash96_enabled;
    VkShaderModule runtime_gated_residual_shader;
    VkDescriptorSetLayout runtime_gated_residual_set_layout;
    VkPipelineLayout runtime_gated_residual_pipeline_layout;
    VkPipeline runtime_gated_residual_pipeline;
    VkShaderModule runtime_add_channel_silu_shader;
    VkDescriptorSetLayout runtime_add_channel_silu_set_layout;
    VkPipelineLayout runtime_add_channel_silu_pipeline_layout;
    VkPipeline runtime_add_channel_silu_pipeline;
    VkShaderModule runtime_add_shader;
    VkDescriptorSetLayout runtime_add_set_layout;
    VkPipelineLayout runtime_add_pipeline_layout;
    VkPipeline runtime_add_pipeline;
    VkShaderModule runtime_out_norm_silu_shader;
    VkDescriptorSetLayout runtime_out_norm_silu_set_layout;
    VkPipelineLayout runtime_out_norm_silu_pipeline_layout;
    VkPipeline runtime_out_norm_silu_pipeline;
    VkShaderModule runtime_latent_update_shader;
    VkDescriptorSetLayout runtime_latent_update_set_layout;
    VkPipelineLayout runtime_latent_update_pipeline_layout;
    VkPipeline runtime_latent_update_pipeline;
    VkShaderModule runtime_lerp_shader;
    VkDescriptorSetLayout runtime_lerp_set_layout;
    VkPipelineLayout runtime_lerp_pipeline_layout;
    VkPipeline runtime_lerp_pipeline;
    VkDescriptorPool ctrl_fc1_descriptor_pool;
    VkDescriptorSet ctrl_fc1_descriptor_set;
    VkDescriptorPool ctrl_silu_descriptor_pool;
    VkDescriptorSet ctrl_silu_descriptor_set;
    VkDescriptorPool ctrl_fc2_descriptor_pool;
    VkDescriptorSet ctrl_fc2_descriptor_set;
    VkDescriptorPool ctrl_rms_descriptor_pool;
    VkDescriptorSet ctrl_rms_descriptor_set;
    VkDescriptorPool denoise_fc1_descriptor_pool;
    VkDescriptorSet denoise_fc1_descriptor_set;
    VkDescriptorPool denoise_silu_descriptor_pool;
    VkDescriptorSet denoise_silu_descriptor_set;
    VkDescriptorPool denoise_fc2_descriptor_pool;
    VkDescriptorSet denoise_fc2_descriptor_set;
    VkDescriptorPool denoise_cond_silu_descriptor_pool;
    VkDescriptorSet denoise_cond_silu_descriptor_set;
    VkDescriptorPool out_norm_descriptor_pool[WORLD_VULKAN_MAX_PASSES];
    VkDescriptorSet out_norm_descriptor_set[WORLD_VULKAN_MAX_PASSES];
    VkDescriptorPool *layer_bias_silu_descriptor_pools;
    VkDescriptorSet *layer_bias_silu_descriptor_sets;
    VkDescriptorPool *layer_mod_descriptor_pools;
    VkDescriptorSet *layer_mod_descriptor_sets;
    VkDescriptorPool *attn_ada_descriptor_pools;
    VkDescriptorSet *attn_ada_descriptor_sets;
    VkDescriptorPool *qkv_proj_descriptor_pools;
    VkDescriptorSet *qkv_proj_descriptor_sets;
    VkDescriptorPool *qkv_proj_wf16_descriptor_pools;
    VkDescriptorSet *qkv_proj_wf16_descriptor_sets;
    VkDescriptorPool qkv_rms_rope_descriptor_pool;
    VkDescriptorSet qkv_rms_rope_descriptor_set;
    VkDescriptorPool qkv_rms_rope_first_descriptor_pool;
    VkDescriptorSet qkv_rms_rope_first_descriptor_set;
    VkDescriptorPool *kv_upsert_descriptor_pools;
    VkDescriptorSet *kv_upsert_descriptor_sets;
    VkDescriptorPool *cache_indices_descriptor_pools;
    VkDescriptorSet *cache_indices_descriptor_sets;
    VkDescriptorPool *indexed_attention_descriptor_pools;
    VkDescriptorSet *indexed_attention_descriptor_sets;
    VkDescriptorPool *attn_out_proj_descriptor_pools;
    VkDescriptorSet *attn_out_proj_descriptor_sets;
    VkDescriptorPool *attn_out_proj_wf16_descriptor_pools;
    VkDescriptorSet *attn_out_proj_wf16_descriptor_sets;
    VkDescriptorPool *attn_residual_descriptor_pools;
    VkDescriptorSet *attn_residual_descriptor_sets;
    VkDescriptorPool *ctrl_cond_descriptor_pools;
    VkDescriptorSet *ctrl_cond_descriptor_sets;
    VkDescriptorPool ctrl_norm_descriptor_pool;
    VkDescriptorSet ctrl_norm_descriptor_set;
    VkDescriptorPool *ctrl_fc1_x_descriptor_pools;
    VkDescriptorSet *ctrl_fc1_x_descriptor_sets;
    VkDescriptorPool *ctrl_add_silu_descriptor_pools;
    VkDescriptorSet *ctrl_add_silu_descriptor_sets;
    VkDescriptorPool *ctrl_fc2_descriptor_pools_layer;
    VkDescriptorSet *ctrl_fc2_descriptor_sets_layer;
    VkDescriptorPool ctrl_add_descriptor_pool;
    VkDescriptorSet ctrl_add_descriptor_set;
    VkDescriptorPool *mlp_ada_descriptor_pools;
    VkDescriptorSet *mlp_ada_descriptor_sets;
    VkDescriptorPool *mlp_fc1_descriptor_pools;
    VkDescriptorSet *mlp_fc1_descriptor_sets;
    VkDescriptorPool *mlp_fc1_wf16_descriptor_pools;
    VkDescriptorSet *mlp_fc1_wf16_descriptor_sets;
    VkDescriptorPool mlp_silu_descriptor_pool;
    VkDescriptorSet mlp_silu_descriptor_set;
    VkDescriptorPool mlp_silu_f16_descriptor_pool;
    VkDescriptorSet mlp_silu_f16_descriptor_set;
    VkDescriptorPool *mlp_fc2_descriptor_pools;
    VkDescriptorSet *mlp_fc2_descriptor_sets;
    VkDescriptorPool *mlp_fc2_wf16_descriptor_pools;
    VkDescriptorSet *mlp_fc2_wf16_descriptor_sets;
    VkDescriptorPool *mlp_fc2_f16x_descriptor_pools;
    VkDescriptorSet *mlp_fc2_f16x_descriptor_sets;
    VkDescriptorPool *mlp_residual_descriptor_pools;
    VkDescriptorSet *mlp_residual_descriptor_sets;
    VkDescriptorPool out_norm_silu_descriptor_pool[WORLD_VULKAN_MAX_PASSES];
    VkDescriptorSet out_norm_silu_descriptor_set[WORLD_VULKAN_MAX_PASSES];
    VkDescriptorPool latent_update_descriptor_pool;
    VkDescriptorSet latent_update_descriptor_set;
    VkDescriptorPool value_residual_descriptor_pool;
    VkDescriptorSet value_residual_descriptor_set;
    VkShaderModule patchify_shader;
    VkDescriptorSetLayout patchify_set_layout;
    VkPipelineLayout patchify_pipeline_layout;
    VkPipeline patchify_pipeline;
    VkDescriptorPool patchify_descriptor_pool;
    VkDescriptorSet patchify_descriptor_set;
    VkShaderModule unpatch_orig_shader;
    VkDescriptorSetLayout unpatch_orig_set_layout;
    VkPipelineLayout unpatch_orig_pipeline_layout;
    VkPipeline unpatch_orig_pipeline;
    VkDescriptorPool unpatch_orig_descriptor_pool;
    VkDescriptorSet unpatch_orig_descriptor_set;
    VkShaderModule latent_rgba_shader;
    VkDescriptorSetLayout latent_rgba_set_layout;
    VkPipelineLayout latent_rgba_pipeline_layout;
    VkPipeline latent_rgba_pipeline;
    VkDescriptorPool latent_rgba_descriptor_pool;
    VkDescriptorSet latent_rgba_descriptor_set;
    int vae_enabled;
    int vae_op_count;
    size_t vae_max_elems;
    size_t vae_scratch_bytes;
    WorldVulkanVaeOp vae_ops[WORLD_VULKAN_MAX_VAE_OPS];
    VkShaderModule taehv_repeat_clamp_shader;
    VkDescriptorSetLayout taehv_repeat_clamp_set_layout;
    VkPipelineLayout taehv_repeat_clamp_pipeline_layout;
    VkPipeline taehv_repeat_clamp_pipeline;
    VkShaderModule taehv_conv_shader;
    VkDescriptorSetLayout taehv_conv_set_layout;
    VkPipelineLayout taehv_conv_pipeline_layout;
    VkPipeline taehv_conv_pipeline;
    VkShaderModule taehv_conv_c4_shader;
    VkDescriptorSetLayout taehv_conv_c4_set_layout;
    VkPipelineLayout taehv_conv_c4_pipeline_layout;
    VkPipeline taehv_conv_c4_pipeline;
    VkShaderModule taehv_conv_1x1_c4_shader;
    VkDescriptorSetLayout taehv_conv_1x1_c4_set_layout;
    VkPipelineLayout taehv_conv_1x1_c4_pipeline_layout;
    VkPipeline taehv_conv_1x1_c4_pipeline;
    VkShaderModule taehv_conv_out_rgba_shader;
    VkDescriptorSetLayout taehv_conv_out_rgba_set_layout;
    VkPipelineLayout taehv_conv_out_rgba_pipeline_layout;
    VkPipeline taehv_conv_out_rgba_pipeline;
    VkShaderModule taehv_relu_shader;
    VkDescriptorSetLayout taehv_relu_set_layout;
    VkPipelineLayout taehv_relu_pipeline_layout;
    VkPipeline taehv_relu_pipeline;
    VkShaderModule taehv_concat_past_shader;
    VkDescriptorSetLayout taehv_concat_past_set_layout;
    VkPipelineLayout taehv_concat_past_pipeline_layout;
    VkPipeline taehv_concat_past_pipeline;
    VkShaderModule taehv_add_relu_shader;
    VkDescriptorSetLayout taehv_add_relu_set_layout;
    VkPipelineLayout taehv_add_relu_pipeline_layout;
    VkPipeline taehv_add_relu_pipeline;
    VkShaderModule taehv_upsample2_shader;
    VkDescriptorSetLayout taehv_upsample2_set_layout;
    VkPipelineLayout taehv_upsample2_pipeline_layout;
    VkPipeline taehv_upsample2_pipeline;
    VkShaderModule taehv_tgrow_shader;
    VkDescriptorSetLayout taehv_tgrow_set_layout;
    VkPipelineLayout taehv_tgrow_pipeline_layout;
    VkPipeline taehv_tgrow_pipeline;
    VkShaderModule taehv_pixel_shuffle_shader;
    VkDescriptorSetLayout taehv_pixel_shuffle_set_layout;
    VkPipelineLayout taehv_pixel_shuffle_pipeline_layout;
    VkPipeline taehv_pixel_shuffle_pipeline;
    VkBuffer vae_buf0_buffer;
    VkDeviceMemory vae_buf0_memory;
    void *vae_buf0_mapped;
    VkBuffer vae_buf1_buffer;
    VkDeviceMemory vae_buf1_memory;
    void *vae_buf1_mapped;
    VkBuffer vae_buf2_buffer;
    VkDeviceMemory vae_buf2_memory;
    void *vae_buf2_mapped;
    VkBuffer vae_weight_buffer[WORLD_VAE_DECODER_CONV_COUNT];
    VkDeviceMemory vae_weight_memory[WORLD_VAE_DECODER_CONV_COUNT];
    void *vae_weight_mapped[WORLD_VAE_DECODER_CONV_COUNT];
    size_t vae_weight_bytes[WORLD_VAE_DECODER_CONV_COUNT];
    VkBuffer vae_bias_buffer[WORLD_VAE_DECODER_CONV_COUNT];
    VkDeviceMemory vae_bias_memory[WORLD_VAE_DECODER_CONV_COUNT];
    void *vae_bias_mapped[WORLD_VAE_DECODER_CONV_COUNT];
    size_t vae_bias_bytes[WORLD_VAE_DECODER_CONV_COUNT];
    VkBuffer latent_buffer;
    VkDeviceMemory latent_memory;
    void *latent_mapped;
    VkBuffer control_buffer;
    VkDeviceMemory control_memory;
    void *control_mapped;
    VkBuffer ctrl_fc1_weight_buffer;
    VkDeviceMemory ctrl_fc1_weight_memory;
    void *ctrl_fc1_weight_mapped;
    VkBuffer ctrl_fc2_weight_buffer;
    VkDeviceMemory ctrl_fc2_weight_memory;
    void *ctrl_fc2_weight_mapped;
    VkBuffer ctrl_hidden_buffer;
    VkDeviceMemory ctrl_hidden_memory;
    void *ctrl_hidden_mapped;
    VkBuffer ctrl_emb_buffer;
    VkDeviceMemory ctrl_emb_memory;
    void *ctrl_emb_mapped;
    VkBuffer ctrl_emb_norm_buffer;
    VkDeviceMemory ctrl_emb_norm_memory;
    void *ctrl_emb_norm_mapped;
    VkBuffer dummy_bias_buffer;
    VkDeviceMemory dummy_bias_memory;
    void *dummy_bias_mapped;
    VkBuffer rms_weight_buffer;
    VkDeviceMemory rms_weight_memory;
    void *rms_weight_mapped;
    VkBuffer noise_buffer;
    VkDeviceMemory noise_memory;
    void *noise_mapped;
    VkBuffer denoise_fc1_weight_buffer;
    VkDeviceMemory denoise_fc1_weight_memory;
    void *denoise_fc1_weight_mapped;
    VkBuffer denoise_fc2_weight_buffer;
    VkDeviceMemory denoise_fc2_weight_memory;
    void *denoise_fc2_weight_mapped;
    VkBuffer noise_hidden_buffer;
    VkDeviceMemory noise_hidden_memory;
    void *noise_hidden_mapped;
    VkBuffer cond_buffer;
    VkDeviceMemory cond_memory;
    void *cond_mapped;
    VkBuffer cond_act_buffer;
    VkDeviceMemory cond_act_memory;
    void *cond_act_mapped;
    VkBuffer out_norm_weight_buffer;
    VkDeviceMemory out_norm_weight_memory;
    void *out_norm_weight_mapped;
    VkBuffer out_mod_table_buffer;
    VkDeviceMemory out_mod_table_memory;
    void *out_mod_table_mapped;
    VkBuffer layer_cond_bias_buffer;
    VkDeviceMemory layer_cond_bias_memory;
    void *layer_cond_bias_mapped;
    VkBuffer layer_cond_proj_weight_buffer;
    VkDeviceMemory layer_cond_proj_weight_memory;
    void *layer_cond_proj_weight_mapped;
    VkBuffer layer_mod_table_buffer;
    VkDeviceMemory layer_mod_table_memory;
    void *layer_mod_table_mapped;
    VkBuffer qkv_proj_weight_buffer;
    VkDeviceMemory qkv_proj_weight_memory;
    void *qkv_proj_weight_mapped;
    VkBuffer qkv_proj_weight_f16_buffer;
    VkDeviceMemory qkv_proj_weight_f16_memory;
    VkBuffer patch_weight_buffer;
    VkDeviceMemory patch_weight_memory;
    void *patch_weight_mapped;
    VkBuffer tokens_buffer;
    VkDeviceMemory tokens_memory;
    void *tokens_mapped;
    VkBuffer norm_buffer;
    VkDeviceMemory norm_memory;
    void *norm_mapped;
    VkBuffer qkv_raw_buffer;
    VkDeviceMemory qkv_raw_memory;
    void *qkv_raw_mapped;
    VkBuffer q_buffer;
    VkDeviceMemory q_memory;
    void *q_mapped;
    VkBuffer k_buffer;
    VkDeviceMemory k_memory;
    void *k_mapped;
    VkBuffer v_buffer;
    VkDeviceMemory v_memory;
    void *v_mapped;
    VkBuffer v_first_buffer;
    VkDeviceMemory v_first_memory;
    void *v_first_mapped;
    VkBuffer x_pos_buffer;
    VkDeviceMemory x_pos_memory;
    void *x_pos_mapped;
    VkBuffer y_pos_buffer;
    VkDeviceMemory y_pos_memory;
    void *y_pos_mapped;
    VkBuffer t_pos_buffer;
    VkDeviceMemory t_pos_memory;
    void *t_pos_mapped;
    VkBuffer xy_buffer;
    VkDeviceMemory xy_memory;
    void *xy_mapped;
    VkBuffer inv_t_buffer;
    VkDeviceMemory inv_t_memory;
    void *inv_t_mapped;
    VkBuffer cache_k_buffer;
    VkDeviceMemory cache_k_memory;
    void *cache_k_mapped;
    VkBuffer cache_v_buffer;
    VkDeviceMemory cache_v_memory;
    void *cache_v_mapped;
    VkBuffer cache_written_buffer;
    VkDeviceMemory cache_written_memory;
    void *cache_written_mapped;
    VkBuffer cache_indices_buffer;
    VkDeviceMemory cache_indices_memory;
    void *cache_indices_mapped;
    VkBuffer cache_index_count_buffer;
    VkDeviceMemory cache_index_count_memory;
    void *cache_index_count_mapped;
    VkBuffer attn_buffer;
    VkDeviceMemory attn_memory;
    void *attn_mapped;
    VkBuffer attn_out_proj_weight_buffer;
    VkDeviceMemory attn_out_proj_weight_memory;
    void *attn_out_proj_weight_mapped;
    VkBuffer attn_out_proj_weight_f16_buffer;
    VkDeviceMemory attn_out_proj_weight_f16_memory;
    VkBuffer attn_proj_buffer;
    VkDeviceMemory attn_proj_memory;
    void *attn_proj_mapped;
    VkBuffer tokens_after_attn_buffer;
    VkDeviceMemory tokens_after_attn_memory;
    void *tokens_after_attn_mapped;
    VkBuffer ctrl_fc1_c_weight_buffer;
    VkDeviceMemory ctrl_fc1_c_weight_memory;
    void *ctrl_fc1_c_weight_mapped;
    VkBuffer ctrl_fc1_x_weight_buffer;
    VkDeviceMemory ctrl_fc1_x_weight_memory;
    void *ctrl_fc1_x_weight_mapped;
    VkBuffer ctrl_fc2_weight_buffer_layer;
    VkDeviceMemory ctrl_fc2_weight_memory_layer;
    void *ctrl_fc2_weight_mapped_layer;
    VkBuffer ctrl_cond_buffer;
    VkDeviceMemory ctrl_cond_memory;
    void *ctrl_cond_mapped;
    VkBuffer ctrl_norm_buffer;
    VkDeviceMemory ctrl_norm_memory;
    void *ctrl_norm_mapped;
    VkBuffer ctrl_hidden_layer_buffer;
    VkDeviceMemory ctrl_hidden_layer_memory;
    void *ctrl_hidden_layer_mapped;
    VkBuffer ctrl_out_buffer;
    VkDeviceMemory ctrl_out_memory;
    void *ctrl_out_mapped;
    VkBuffer tokens_after_ctrl_buffer;
    VkDeviceMemory tokens_after_ctrl_memory;
    void *tokens_after_ctrl_mapped;
    VkBuffer dit_mlp_fc1_weight_buffer;
    VkDeviceMemory dit_mlp_fc1_weight_memory;
    void *dit_mlp_fc1_weight_mapped;
    VkBuffer dit_mlp_fc1_weight_f16_buffer;
    VkDeviceMemory dit_mlp_fc1_weight_f16_memory;
    VkBuffer dit_mlp_fc2_weight_buffer;
    VkDeviceMemory dit_mlp_fc2_weight_memory;
    void *dit_mlp_fc2_weight_mapped;
    VkBuffer dit_mlp_fc2_weight_f16_buffer;
    VkDeviceMemory dit_mlp_fc2_weight_f16_memory;
    VkBuffer mlp_in_buffer;
    VkDeviceMemory mlp_in_memory;
    void *mlp_in_mapped;
    VkBuffer mlp_hidden_buffer;
    VkDeviceMemory mlp_hidden_memory;
    void *mlp_hidden_mapped;
    VkBuffer mlp_hidden_f16_buffer;
    VkDeviceMemory mlp_hidden_f16_memory;
    VkBuffer mlp_out_buffer;
    VkDeviceMemory mlp_out_memory;
    void *mlp_out_mapped;
    VkBuffer tokens_after_mlp_buffer;
    VkDeviceMemory tokens_after_mlp_memory;
    void *tokens_after_mlp_mapped;
    VkBuffer final_tokens_buffer;
    VkDeviceMemory final_tokens_memory;
    void *final_tokens_mapped;
    VkBuffer unpatch_weight_buffer;
    VkDeviceMemory unpatch_weight_memory;
    void *unpatch_weight_mapped;
    VkBuffer unpatch_bias_buffer;
    VkDeviceMemory unpatch_bias_memory;
    void *unpatch_bias_mapped;
    VkBuffer latent_out_buffer;
    VkDeviceMemory latent_out_memory;
    void *latent_out_mapped;
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int positive_mod_int(int x, int m) {
    int r = x % m;
    return r < 0 ? r + m : r;
}

static const char *vk_result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "VK_RESULT_UNKNOWN";
    }
}

static int vk_check(VkResult r, const char *expr, const char *file, int line) {
    if (r == VK_SUCCESS) return 0;
    fprintf(stderr, "Vulkan error at %s:%d: %s -> %s (%d)\n", file, line, expr, vk_result_name(r), (int)r);
    return 1;
}

#define VK_CALL(expr) do { if (vk_check((expr), #expr, __FILE__, __LINE__)) goto fail; } while (0)
#define VK_CALL_RET(expr) do { if (vk_check((expr), #expr, __FILE__, __LINE__)) return 1; } while (0)

static int read_file_bytes(const char *path, void **data_out, size_t *bytes_out) {
    *data_out = NULL;
    *bytes_out = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open Vulkan shader: %s\n", path);
        return 1;
    }
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return 1;
    }
    long n = ftell(f);
    if (n <= 0 || (n % 4) != 0) {
        fclose(f);
        fprintf(stderr, "invalid SPIR-V size for %s: %ld\n", path, n);
        return 1;
    }
    rewind(f);
    void *data = malloc((size_t)n);
    if (!data) {
        fclose(f);
        return 1;
    }
    size_t got = fread(data, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(data);
        fprintf(stderr, "short read for Vulkan shader: %s\n", path);
        return 1;
    }
    *data_out = data;
    *bytes_out = (size_t)n;
    return 0;
}

static int find_queue_family(VkPhysicalDevice dev, uint32_t *family_out) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, NULL);
    if (count == 0) return 1;
    VkQueueFamilyProperties *props = (VkQueueFamilyProperties *)calloc(count, sizeof(*props));
    if (!props) return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props);
    int found = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            *family_out = i;
            found = 1;
            break;
        }
    }
    free(props);
    return found ? 0 : 1;
}

static int physical_device_has_extension(VkPhysicalDevice dev, const char *name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(dev, NULL, &count, NULL) != VK_SUCCESS) return 0;
    VkExtensionProperties *props =
        (VkExtensionProperties *)calloc(count ? count : 1, sizeof(*props));
    if (!props) return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(dev, NULL, &count, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(props[i].extensionName, name) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(props);
    return found;
}

static int pick_physical_device(WorldVulkanRuntime *rt) {
    uint32_t count = 0;
    VK_CALL_RET(vkEnumeratePhysicalDevices(rt->instance, &count, NULL));
    if (count == 0) {
        fprintf(stderr, "no Vulkan physical devices found\n");
        return 1;
    }
    VkPhysicalDevice *devices = (VkPhysicalDevice *)calloc(count, sizeof(*devices));
    if (!devices) return 1;
    if (vkEnumeratePhysicalDevices(rt->instance, &count, devices) != VK_SUCCESS) {
        free(devices);
        return 1;
    }
    int best = -1;
    uint32_t best_family = 0;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props;
        uint32_t family = 0;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (find_queue_family(devices[i], &family)) continue;
        if (best < 0 || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            best = (int)i;
            best_family = family;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
        }
    }
    if (best < 0) {
        free(devices);
        fprintf(stderr, "no Vulkan compute-capable device found\n");
        return 1;
    }
    rt->physical_device = devices[best];
    rt->queue_family = best_family;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(rt->physical_device, &props);
    rt->max_compute_shared_memory_size = props.limits.maxComputeSharedMemorySize;
    fprintf(stderr, "Vulkan device: %s queue_family=%u\n", props.deviceName, rt->queue_family);
    free(devices);
    return 0;
}

static int find_memory_type(WorldVulkanRuntime *rt, uint32_t type_bits, VkMemoryPropertyFlags flags, uint32_t *type_out) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(rt->physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            *type_out = i;
            return 0;
        }
    }
    return 1;
}

static int create_host_buffer(
        WorldVulkanRuntime *rt,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkBuffer *buffer,
        VkDeviceMemory *memory,
        void **mapped) {
    *buffer = VK_NULL_HANDLE;
    *memory = VK_NULL_HANDLE;
    *mapped = NULL;

    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CALL_RET(vkCreateBuffer(rt->device, &buffer_info, NULL, buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(rt->device, *buffer, &req);
    uint32_t type_index = 0;
    if (find_memory_type(rt, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &type_index)) {
        fprintf(stderr, "failed to find host-visible coherent Vulkan memory\n");
        return 1;
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = type_index;
    VK_CALL_RET(vkAllocateMemory(rt->device, &alloc_info, NULL, memory));
    VK_CALL_RET(vkBindBufferMemory(rt->device, *buffer, *memory, 0));
    VK_CALL_RET(vkMapMemory(rt->device, *memory, 0, size, 0, mapped));
    return 0;
}

static int create_device_buffer(
        WorldVulkanRuntime *rt,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkBuffer *buffer,
        VkDeviceMemory *memory) {
    *buffer = VK_NULL_HANDLE;
    *memory = VK_NULL_HANDLE;

    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CALL_RET(vkCreateBuffer(rt->device, &buffer_info, NULL, buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(rt->device, *buffer, &req);
    uint32_t type_index = 0;
    if (find_memory_type(rt, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type_index)) {
        fprintf(stderr, "failed to find device-local Vulkan memory\n");
        return 1;
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = type_index;
    VK_CALL_RET(vkAllocateMemory(rt->device, &alloc_info, NULL, memory));
    VK_CALL_RET(vkBindBufferMemory(rt->device, *buffer, *memory, 0));
    return 0;
}

static void destroy_host_buffer(WorldVulkanRuntime *rt, VkBuffer buffer, VkDeviceMemory memory, void *mapped) {
    if (mapped) vkUnmapMemory(rt->device, memory);
    if (buffer) vkDestroyBuffer(rt->device, buffer, NULL);
    if (memory) vkFreeMemory(rt->device, memory, NULL);
}

static uint16_t runtime_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

static int copy_buffer_region_to_device(
        WorldVulkanRuntime *rt,
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize src_offset,
        VkDeviceSize dst_offset,
        VkDeviceSize bytes) {
    int rc = 1;
    if (vkResetFences(rt->device, 1, &rt->fence) != VK_SUCCESS) goto cleanup;
    if (vkResetCommandBuffer(rt->command_buffer, 0) != VK_SUCCESS) goto cleanup;
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(rt->command_buffer, &begin_info) != VK_SUCCESS) goto cleanup;
    VkBufferCopy copy_region;
    memset(&copy_region, 0, sizeof(copy_region));
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = dst_offset;
    copy_region.size = bytes;
    vkCmdCopyBuffer(rt->command_buffer, src, dst, 1, &copy_region);
    if (vkEndCommandBuffer(rt->command_buffer) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    if (vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence) != VK_SUCCESS) goto cleanup;
    if (vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) goto cleanup;
    rc = 0;

cleanup:
    return rc;
}

static int copy_buffer_to_device(WorldVulkanRuntime *rt, VkBuffer src, VkBuffer dst, VkDeviceSize bytes) {
    return copy_buffer_region_to_device(rt, src, dst, 0, 0, bytes);
}

static int upload_to_device_buffer(WorldVulkanRuntime *rt, VkBuffer dst, const void *src, VkDeviceSize bytes) {
    if (!src || bytes == 0) return 0;
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *staging_mapped = NULL;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
    memcpy(staging_mapped, src, (size_t)bytes);
    int rc = copy_buffer_to_device(rt, staging_buffer, dst, bytes);
    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
    return rc;
}

static int upload_f32_as_f16_to_device_buffer(
        WorldVulkanRuntime *rt,
        VkBuffer dst,
        VkDeviceSize dst_offset,
        const float *src,
        size_t elems) {
    if (!src || elems == 0) return 0;
    const size_t chunk_elems = 16u * 1024u * 1024u;
    VkDeviceSize staging_bytes = (VkDeviceSize)(chunk_elems * sizeof(uint16_t));
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *staging_mapped = NULL;
    if (create_host_buffer(rt, staging_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
    int rc = 0;
    size_t offset = 0;
    while (offset < elems) {
        size_t n = elems - offset;
        if (n > chunk_elems) n = chunk_elems;
        uint16_t *dst16 = (uint16_t *)staging_mapped;
        for (size_t i = 0; i < n; ++i) {
            dst16[i] = runtime_f32_to_f16(src[offset + i]);
        }
        VkDeviceSize bytes = (VkDeviceSize)(n * sizeof(uint16_t));
        if (copy_buffer_region_to_device(rt, staging_buffer, dst, 0,
                    dst_offset + (VkDeviceSize)(offset * sizeof(uint16_t)), bytes)) {
            rc = 1;
            break;
        }
        offset += n;
    }
    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
    return rc;
}

static int create_weight_buffer_from_data(
        WorldVulkanRuntime *rt,
        VkDeviceSize bytes,
        const void *src,
        VkBuffer *buffer,
        VkDeviceMemory *memory) {
    if (create_device_buffer(rt, bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                buffer, memory)) return 1;
    return upload_to_device_buffer(rt, *buffer, src, bytes);
}

static int create_device_storage_buffer(
        WorldVulkanRuntime *rt,
        VkDeviceSize bytes,
        VkBuffer *buffer,
        VkDeviceMemory *memory) {
    return create_device_buffer(rt, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            buffer, memory);
}

static int clear_device_buffer(WorldVulkanRuntime *rt, VkBuffer buffer, VkDeviceSize bytes) {
    if (!buffer || bytes == 0) return 0;
    if ((bytes & 3u) != 0u) {
        fprintf(stderr, "Vulkan clear_device_buffer size must be 4-byte aligned: %llu\n",
                (unsigned long long)bytes);
        return 1;
    }
    int rc = 1;
    if (vkResetFences(rt->device, 1, &rt->fence) != VK_SUCCESS) goto cleanup;
    if (vkResetCommandBuffer(rt->command_buffer, 0) != VK_SUCCESS) goto cleanup;
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(rt->command_buffer, &begin_info) != VK_SUCCESS) goto cleanup;
    vkCmdFillBuffer(rt->command_buffer, buffer, 0, bytes, 0u);
    if (vkEndCommandBuffer(rt->command_buffer) != VK_SUCCESS) goto cleanup;

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    if (vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence) != VK_SUCCESS) goto cleanup;
    if (vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) goto cleanup;
    rc = 0;

cleanup:
    return rc;
}

static int download_from_device_buffer(
        WorldVulkanRuntime *rt,
        VkBuffer src,
        void *dst,
        VkDeviceSize bytes) {
    if (!src || !dst || bytes == 0) return 0;
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *staging_mapped = NULL;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
    int rc = copy_buffer_to_device(rt, src, staging_buffer, bytes);
    if (rc == 0) {
        memcpy(dst, staging_mapped, (size_t)bytes);
    }
    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
    return rc;
}

static int create_output_buffer(WorldVulkanRuntime *rt) {
    VkDeviceSize bytes = (VkDeviceSize)(rt->pixel_count * sizeof(uint32_t));
    if (create_device_buffer(rt, bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                &rt->output_buffer,
                &rt->output_memory)) return 1;
    return create_host_buffer(rt,
            bytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            &rt->output_readback_buffer,
            &rt->output_readback_memory,
            &rt->output_mapped);
}

static int create_shader_module_from_name(WorldVulkanRuntime *rt, const char *shader_name, VkShaderModule *module_out) {
    *module_out = VK_NULL_HANDLE;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s.spv", WORLD_VULKAN_SHADER_DIR, shader_name);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "Vulkan shader path too long\n");
        return 1;
    }

    void *spv = NULL;
    size_t spv_bytes = 0;
    if (read_file_bytes(path, &spv, &spv_bytes)) return 1;

    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_bytes;
    shader_info.pCode = (const uint32_t *)spv;
    VkResult shader_result = vkCreateShaderModule(rt->device, &shader_info, NULL, module_out);
    free(spv);
    if (vk_check(shader_result, "vkCreateShaderModule", __FILE__, __LINE__)) return 1;
    return 0;
}

static int create_fill_pipeline(WorldVulkanRuntime *rt) {
    if (create_shader_module_from_name(rt, "fill_rgba.comp", &rt->fill_shader)) return 1;

    VkDescriptorSetLayoutBinding binding;
    memset(&binding, 0, sizeof(binding));
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo set_info;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 1;
    set_info.pBindings = &binding;
    VK_CALL_RET(vkCreateDescriptorSetLayout(rt->device, &set_info, NULL, &rt->descriptor_set_layout));

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(WorldVulkanFillPush);

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &rt->descriptor_set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VK_CALL_RET(vkCreatePipelineLayout(rt->device, &layout_info, NULL, &rt->pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = rt->fill_shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = rt->pipeline_layout;
    VK_CALL_RET(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &rt->fill_pipeline));
    return 0;
}

static int create_descriptors(WorldVulkanRuntime *rt) {
    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CALL_RET(vkCreateDescriptorPool(rt->device, &pool_info, NULL, &rt->descriptor_pool));

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = rt->descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &rt->descriptor_set_layout;
    VK_CALL_RET(vkAllocateDescriptorSets(rt->device, &alloc_info, &rt->descriptor_set));

    VkDescriptorBufferInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.buffer = rt->output_buffer;
    buffer_info.offset = 0;
    buffer_info.range = rt->pixel_count * sizeof(uint32_t);

    VkWriteDescriptorSet write;
    memset(&write, 0, sizeof(write));
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = rt->descriptor_set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(rt->device, 1, &write, 0, NULL);
    return 0;
}

static int create_commands(WorldVulkanRuntime *rt) {
    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = rt->queue_family;
    VK_CALL_RET(vkCreateCommandPool(rt->device, &pool_info, NULL, &rt->command_pool));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = rt->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VK_CALL_RET(vkAllocateCommandBuffers(rt->device, &alloc_info, &rt->command_buffer));

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CALL_RET(vkCreateFence(rt->device, &fence_info, NULL, &rt->fence));
    return 0;
}

static int create_storage_pipeline(
        WorldVulkanRuntime *rt,
        const char *shader_name,
        uint32_t binding_count,
        uint32_t push_size,
        VkShaderModule *shader,
        VkDescriptorSetLayout *set_layout,
        VkPipelineLayout *pipeline_layout,
        VkPipeline *pipeline) {
    *shader = VK_NULL_HANDLE;
    *set_layout = VK_NULL_HANDLE;
    *pipeline_layout = VK_NULL_HANDLE;
    *pipeline = VK_NULL_HANDLE;
    if (create_shader_module_from_name(rt, shader_name, shader)) return 1;

    VkDescriptorSetLayoutBinding *bindings =
        (VkDescriptorSetLayoutBinding *)calloc(binding_count, sizeof(*bindings));
    if (!bindings) return 1;
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set_info;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = binding_count;
    set_info.pBindings = bindings;
    VkResult set_result = vkCreateDescriptorSetLayout(rt->device, &set_info, NULL, set_layout);
    free(bindings);
    if (vk_check(set_result, "vkCreateDescriptorSetLayout", __FILE__, __LINE__)) return 1;

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_size;

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = set_layout;
    layout_info.pushConstantRangeCount = push_size ? 1 : 0;
    layout_info.pPushConstantRanges = push_size ? &push_range : NULL;
    VK_CALL_RET(vkCreatePipelineLayout(rt->device, &layout_info, NULL, pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = *shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = *pipeline_layout;
    VK_CALL_RET(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline));
    return 0;
}

static int create_storage_pipeline_with_set_layout(
        WorldVulkanRuntime *rt,
        const char *shader_name,
        VkDescriptorSetLayout set_layout,
        uint32_t push_size,
        VkShaderModule *shader,
        VkPipelineLayout *pipeline_layout,
        VkPipeline *pipeline) {
    *shader = VK_NULL_HANDLE;
    *pipeline_layout = VK_NULL_HANDLE;
    *pipeline = VK_NULL_HANDLE;
    if (create_shader_module_from_name(rt, shader_name, shader)) return 1;

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_size;

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = push_size ? 1 : 0;
    layout_info.pPushConstantRanges = push_size ? &push_range : NULL;
    VK_CALL_RET(vkCreatePipelineLayout(rt->device, &layout_info, NULL, pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = *shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = *pipeline_layout;
    VK_CALL_RET(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline));
    return 0;
}

static int create_storage_descriptor_set(
        WorldVulkanRuntime *rt,
        VkDescriptorSetLayout set_layout,
        uint32_t binding_count,
        const VkBuffer *buffers,
        const VkDeviceSize *sizes,
        const VkDeviceSize *offsets,
        VkDescriptorPool *descriptor_pool,
        VkDescriptorSet *descriptor_set) {
    *descriptor_pool = VK_NULL_HANDLE;
    *descriptor_set = VK_NULL_HANDLE;
    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = binding_count;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CALL_RET(vkCreateDescriptorPool(rt->device, &pool_info, NULL, descriptor_pool));

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = *descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;
    VK_CALL_RET(vkAllocateDescriptorSets(rt->device, &alloc_info, descriptor_set));

    VkDescriptorBufferInfo *buffer_infos =
        (VkDescriptorBufferInfo *)calloc(binding_count, sizeof(*buffer_infos));
    VkWriteDescriptorSet *writes =
        (VkWriteDescriptorSet *)calloc(binding_count, sizeof(*writes));
    if (!buffer_infos || !writes) {
        free(buffer_infos);
        free(writes);
        return 1;
    }
    for (uint32_t i = 0; i < binding_count; ++i) {
        buffer_infos[i].buffer = buffers[i];
        buffer_infos[i].offset = offsets ? offsets[i] : 0;
        buffer_infos[i].range = sizes[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = *descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(rt->device, binding_count, writes, 0, NULL);
    free(buffer_infos);
    free(writes);
    return 0;
}

static int submit_compute(
        WorldVulkanRuntime *rt,
        VkPipeline pipeline,
        VkPipelineLayout pipeline_layout,
        VkDescriptorSet descriptor_set,
        const void *push,
        uint32_t push_size,
        uint32_t groups_x,
        uint32_t groups_y,
        uint32_t groups_z) {
    VK_CALL_RET(vkResetFences(rt->device, 1, &rt->fence));
    VK_CALL_RET(vkResetCommandBuffer(rt->command_buffer, 0));
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL_RET(vkBeginCommandBuffer(rt->command_buffer, &begin_info));
    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    if (push && push_size) {
        vkCmdPushConstants(rt->command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, push_size, push);
    }
    vkCmdDispatch(rt->command_buffer, groups_x, groups_y, groups_z);
    VK_CALL_RET(vkEndCommandBuffer(rt->command_buffer));

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    VK_CALL_RET(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
    VK_CALL_RET(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));
    return 0;
}

static uint32_t world_vulkan_lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float world_vulkan_lcg_uniform01(uint32_t *state) {
    uint32_t r = world_vulkan_lcg_next(state);
    return (float)((r >> 8) & 0x00ffffffu) / 16777216.0f;
}

static void fill_runtime_latent(WorldVulkanRuntime *rt, const float *control_input) {
    (void)control_input;
    float *latent = (float *)rt->latent_mapped;
    if (!latent) return;
    uint32_t state = rt->seed + (uint32_t)rt->frame_ordinal;
    if (state == 0u) state = 1u;
    if (rt->noise_mode == WORLD_NOISE_NORMAL) {
        for (size_t i = 0; i < rt->latent_elems; i += 2) {
            float u1 = world_vulkan_lcg_uniform01(&state);
            float u2 = world_vulkan_lcg_uniform01(&state);
            if (u1 < 1.0e-7f) u1 = 1.0e-7f;
            float mag = sqrtf(-2.0f * logf(u1));
            float phase = 2.0f * (float)M_PI * u2;
            latent[i] = mag * cosf(phase);
            if (i + 1 < rt->latent_elems) {
                latent[i + 1] = mag * sinf(phase);
            }
        }
    } else {
        for (size_t i = 0; i < rt->latent_elems; ++i) {
            float u = world_vulkan_lcg_uniform01(&state);
            latent[i] = 2.0f * u - 1.0f;
        }
    }
}

static void copy_runtime_control(WorldVulkanRuntime *rt, const float *control_input) {
    float *dst = (float *)rt->control_mapped;
    if (!dst) return;
    for (int i = 0; i < rt->ctrl_dim; ++i) {
        dst[i] = control_input ? control_input[i] : 0.0f;
    }
}

static void fill_runtime_positions(WorldVulkanRuntime *rt, int frame_timestamp) {
    uint32_t *x_pos = (uint32_t *)rt->x_pos_mapped;
    uint32_t *y_pos = (uint32_t *)rt->y_pos_mapped;
    uint32_t *t_pos = (uint32_t *)rt->t_pos_mapped;
    if (!x_pos || !y_pos || !t_pos) return;
    for (int i = 0; i < rt->T; ++i) {
        int y = i / rt->cfg.width;
        int x = i - y * rt->cfg.width;
        x_pos[i] = (uint32_t)x;
        y_pos[i] = (uint32_t)y;
        t_pos[i] = (uint32_t)frame_timestamp;
    }
}

static void fill_runtime_rope_tables(WorldVulkanRuntime *rt) {
    float *xy = (float *)rt->xy_mapped;
    float *inv_t = (float *)rt->inv_t_mapped;
    if (!xy || !inv_t) return;
    int n_xy = (rt->d_xy + 1) / 2;
    float max_freq = (float)(rt->cfg.height < rt->cfg.width ? rt->cfg.height : rt->cfg.width) * 0.8f;
    for (int i = 0; i < rt->d_xy; ++i) {
        int src = i / 2;
        float a = n_xy == 1 ? 0.0f : (float)src / (float)(n_xy - 1);
        xy[i] = (1.0f + (max_freq * 0.5f - 1.0f) * a) * (float)M_PI;
    }
    for (int i = 0; i < rt->d_t; ++i) {
        int src = i / 2;
        float exponent = (float)(2 * src) / (float)rt->d_t;
        inv_t[i] = 1.0f / powf(10000.0f, exponent);
    }
}

static void fill_noise_embedding(float *emb, float sigma) {
    int half = 256;
    float root2 = sqrtf(2.0f);
    for (int i = 0; i < half; ++i) {
        float a = half == 1 ? 0.0f : (float)i / (float)(half - 1);
        float freq = powf(10000.0f, -a);
        float phase = sigma * 1000.0f * freq;
        emb[i] = sinf(phase) * root2;
        emb[half + i] = cosf(phase) * root2;
    }
}

static void cmd_shader_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
}

static int profile_agg_desc_cmp(const void *a, const void *b) {
    const WorldVulkanProfileAgg *aa = (const WorldVulkanProfileAgg *)a;
    const WorldVulkanProfileAgg *bb = (const WorldVulkanProfileAgg *)b;
    if (aa->total_ms < bb->total_ms) return 1;
    if (aa->total_ms > bb->total_ms) return -1;
    return 0;
}

static int create_profile_queries(WorldVulkanRuntime *rt) {
    const char *profile_env = getenv("WORLD_VULKAN_PROFILE");
    if (!profile_env || !profile_env[0] || profile_env[0] == '0') return 0;

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(rt->physical_device, &family_count, NULL);
    VkQueueFamilyProperties *families =
        (VkQueueFamilyProperties *)calloc(family_count ? family_count : 1, sizeof(*families));
    if (!families) return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(rt->physical_device, &family_count, families);
    uint32_t valid_bits = rt->queue_family < family_count ?
        families[rt->queue_family].timestampValidBits : 0;
    free(families);
    if (valid_bits == 0) {
        fprintf(stderr, "Vulkan timestamp profiling requested, but queue family has timestampValidBits=0\n");
        return 0;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(rt->physical_device, &props);
    rt->profile_timestamp_period_ns = props.limits.timestampPeriod;
    rt->profile_timestamp_valid_bits = valid_bits;
    rt->profile_max_events = 8192;
    rt->profile_top_n = 40;
    {
        const char *events_env = getenv("WORLD_VULKAN_PROFILE_EVENTS");
        if (events_env && events_env[0]) {
            int v = atoi(events_env);
            if (v >= 128 && v <= 65536) rt->profile_max_events = (uint32_t)v;
        }
        const char *top_env = getenv("WORLD_VULKAN_PROFILE_TOP");
        if (top_env && top_env[0]) {
            int v = atoi(top_env);
            if (v >= 0 && v <= 1024) rt->profile_top_n = v;
        }
    }

    rt->profile_events = (WorldVulkanProfileEvent *)calloc(rt->profile_max_events, sizeof(*rt->profile_events));
    rt->profile_query_values = (uint64_t *)calloc((size_t)rt->profile_max_events * 2, sizeof(*rt->profile_query_values));
    if (!rt->profile_events || !rt->profile_query_values) return 1;

    VkQueryPoolCreateInfo query_info;
    memset(&query_info, 0, sizeof(query_info));
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = rt->profile_max_events * 2;
    VK_CALL_RET(vkCreateQueryPool(rt->device, &query_info, NULL, &rt->profile_query_pool));
    rt->profile_enabled = 1;
    fprintf(stderr,
            "Vulkan timestamp profile enabled: max_events=%u timestamp_period=%.3fns valid_bits=%u\n",
            rt->profile_max_events, rt->profile_timestamp_period_ns, rt->profile_timestamp_valid_bits);
    return 0;
}

static void profile_begin_frame(WorldVulkanRuntime *rt) {
    if (!rt->profile_enabled || !rt->profile_query_pool) return;
    rt->profile_event_count = 0;
    rt->profile_overflow_warned = 0;
    vkCmdResetQueryPool(rt->command_buffer, rt->profile_query_pool, 0, rt->profile_max_events * 2);
}

static uint32_t profile_begin_dispatch(WorldVulkanRuntime *rt, const char *label) {
    if (!rt->profile_enabled || !rt->profile_query_pool) return UINT32_MAX;
    if (rt->profile_event_count >= rt->profile_max_events) {
        if (!rt->profile_overflow_warned) {
            fprintf(stderr, "Vulkan timestamp profile overflow at %u events; increase WORLD_VULKAN_PROFILE_EVENTS\n",
                    rt->profile_max_events);
            rt->profile_overflow_warned = 1;
        }
        return UINT32_MAX;
    }
    uint32_t event_idx = rt->profile_event_count++;
    WorldVulkanProfileEvent *ev = &rt->profile_events[event_idx];
    ev->label = label;
    ev->begin_query = event_idx * 2;
    ev->end_query = event_idx * 2 + 1;
    vkCmdWriteTimestamp(rt->command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            rt->profile_query_pool, ev->begin_query);
    return event_idx;
}

static void profile_end_dispatch(WorldVulkanRuntime *rt, uint32_t event_idx) {
    if (event_idx == UINT32_MAX || !rt->profile_enabled || !rt->profile_query_pool) return;
    vkCmdWriteTimestamp(rt->command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            rt->profile_query_pool, rt->profile_events[event_idx].end_query);
}

static void profile_report_frame(WorldVulkanRuntime *rt) {
    if (!rt->profile_enabled || !rt->profile_query_pool || rt->profile_event_count == 0) return;
    uint32_t query_count = rt->profile_event_count * 2;
    VkResult res = vkGetQueryPoolResults(rt->device, rt->profile_query_pool,
            0, query_count,
            (size_t)query_count * sizeof(uint64_t), rt->profile_query_values,
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "Vulkan timestamp profile read failed: %d\n", (int)res);
        return;
    }

    WorldVulkanProfileAgg *aggs =
        (WorldVulkanProfileAgg *)calloc(rt->profile_event_count, sizeof(*aggs));
    if (!aggs) return;
    uint32_t agg_count = 0;
    double total_ms = 0.0;
    uint64_t mask = UINT64_MAX;
    if (rt->profile_timestamp_valid_bits < 64) {
        mask = (1ull << rt->profile_timestamp_valid_bits) - 1ull;
    }

    for (uint32_t i = 0; i < rt->profile_event_count; ++i) {
        const WorldVulkanProfileEvent *ev = &rt->profile_events[i];
        uint64_t begin = rt->profile_query_values[ev->begin_query] & mask;
        uint64_t end = rt->profile_query_values[ev->end_query] & mask;
        uint64_t delta = (end - begin) & mask;
        double ms = (double)delta * (double)rt->profile_timestamp_period_ns * 1.0e-6;
        total_ms += ms;

        uint32_t j = 0;
        for (; j < agg_count; ++j) {
            if (strcmp(aggs[j].label, ev->label) == 0) break;
        }
        if (j == agg_count) {
            aggs[j].label = ev->label;
            agg_count++;
        }
        aggs[j].total_ms += ms;
        if (ms > aggs[j].max_ms) aggs[j].max_ms = ms;
        aggs[j].count++;
    }

    qsort(aggs, agg_count, sizeof(*aggs), profile_agg_desc_cmp);
    fprintf(stderr, "Vulkan profile frame %d: dispatch_total=%.3fms events=%u labels=%u\n",
            rt->frame_ordinal, total_ms, rt->profile_event_count, agg_count);
    uint32_t limit = (rt->profile_top_n <= 0 || rt->profile_top_n > (int)agg_count) ?
        agg_count : (uint32_t)rt->profile_top_n;
    for (uint32_t i = 0; i < limit; ++i) {
        double avg = aggs[i].count ? aggs[i].total_ms / (double)aggs[i].count : 0.0;
        fprintf(stderr, "  %-28s total=%9.3fms count=%4u avg=%8.4fms max=%8.4fms\n",
                aggs[i].label, aggs[i].total_ms, aggs[i].count, avg, aggs[i].max_ms);
    }
    free(aggs);
}

static int use_runtime_linear_tile64(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner) {
    return rt->runtime_linear_tile64_enabled &&
        rt->runtime_linear_tile64_pipeline &&
        rows >= 64u &&
        cols >= 1024u &&
        inner >= 1024u;
}

static int use_runtime_linear_coopmat(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    return rt->runtime_linear_coopmat_enabled &&
        rt->runtime_linear_coopmat_pipeline &&
        !has_bias &&
        rows >= 16u &&
        cols >= 8u &&
        inner >= 16u &&
        (rows % 16u) == 0u &&
        (cols % 8u) == 0u &&
        (inner % 16u) == 0u;
}

static int use_runtime_linear_wf16_coopmat_n16(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    return rt->runtime_linear_wf16_coopmat_enabled &&
        rt->runtime_linear_wf16_coopmat_n16_pipeline &&
        !has_bias &&
        rows >= 16u &&
        cols >= 16u &&
        inner >= 16u &&
        (rows % 16u) == 0u &&
        (cols % 16u) == 0u &&
        (inner % 16u) == 0u;
}

static int use_runtime_linear_wf16_coopmat(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    return rt->runtime_linear_wf16_coopmat_enabled &&
        rt->runtime_linear_wf16_coopmat_pipeline &&
        !has_bias &&
        rows >= 16u &&
        cols >= 8u &&
        inner >= 16u &&
        (rows % 16u) == 0u &&
        (cols % 8u) == 0u &&
        (inner % 16u) == 0u;
}

static int use_runtime_linear_f16x_wf16_coopmat(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    return rt->runtime_linear_wf16_coopmat_enabled &&
        rt->runtime_linear_f16x_wf16_coopmat_pipeline &&
        !has_bias &&
        rows >= 16u &&
        cols >= 8u &&
        inner >= 16u &&
        (rows % 16u) == 0u &&
        (cols % 8u) == 0u &&
        (inner % 16u) == 0u;
}

static VkPipeline runtime_linear_pipeline_for(
        WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    if (use_runtime_linear_coopmat(rt, rows, cols, inner, has_bias)) {
        return rt->runtime_linear_coopmat_pipeline;
    }
    return use_runtime_linear_tile64(rt, rows, cols, inner) ?
        rt->runtime_linear_tile64_pipeline : rt->runtime_linear_pipeline;
}

static VkPipelineLayout runtime_linear_layout_for(
        WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    if (use_runtime_linear_coopmat(rt, rows, cols, inner, has_bias)) {
        return rt->runtime_linear_coopmat_pipeline_layout;
    }
    return use_runtime_linear_tile64(rt, rows, cols, inner) ?
        rt->runtime_linear_tile64_pipeline_layout : rt->runtime_linear_pipeline_layout;
}

static uint32_t runtime_linear_tile_x_for(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    if (use_runtime_linear_coopmat(rt, rows, cols, inner, has_bias)) return 8u;
    return use_runtime_linear_tile64(rt, rows, cols, inner) ? 64u : 16u;
}

static uint32_t runtime_linear_tile_y_for(
        const WorldVulkanRuntime *rt,
        uint32_t rows,
        uint32_t cols,
        uint32_t inner,
        uint32_t has_bias) {
    if (use_runtime_linear_coopmat(rt, rows, cols, inner, has_bias)) return 16u;
    return use_runtime_linear_tile64(rt, rows, cols, inner) ? 64u : 16u;
}

static void record_output_readback(WorldVulkanRuntime *rt) {
    VkDeviceSize bytes = (VkDeviceSize)(rt->pixel_count * sizeof(uint32_t));
    VkBufferCopy copy_region;
    memset(&copy_region, 0, sizeof(copy_region));
    copy_region.size = bytes;

    VkBufferMemoryBarrier to_transfer;
    memset(&to_transfer, 0, sizeof(to_transfer));
    to_transfer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.buffer = rt->output_buffer;
    to_transfer.offset = 0;
    to_transfer.size = bytes;
    vkCmdPipelineBarrier(rt->command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &to_transfer, 0, NULL);

    vkCmdCopyBuffer(rt->command_buffer, rt->output_buffer, rt->output_readback_buffer, 1, &copy_region);

    VkBufferMemoryBarrier to_host;
    memset(&to_host, 0, sizeof(to_host));
    to_host.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer = rt->output_readback_buffer;
    to_host.offset = 0;
    to_host.size = bytes;
    vkCmdPipelineBarrier(rt->command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, NULL, 1, &to_host, 0, NULL);
}

static int precompute_runtime_out_mods(WorldVulkanRuntime *rt) {
    if (!rt || !rt->command_buffer || !rt->denoise_out_norm_enabled) return 0;
    if (rt->total_passes <= 0 || rt->total_passes > WORLD_VULKAN_MAX_PASSES) return 1;

    WorldVulkanLinearPush fc1_push;
    memset(&fc1_push, 0, sizeof(fc1_push));
    fc1_push.rows = 1;
    fc1_push.cols = (uint32_t)rt->mlp_hidden;
    fc1_push.inner = 512;
    fc1_push.has_bias = 0;

    WorldVulkanSiluPush hidden_silu_push;
    memset(&hidden_silu_push, 0, sizeof(hidden_silu_push));
    hidden_silu_push.n = (uint32_t)rt->mlp_hidden;

    WorldVulkanLinearPush fc2_push;
    memset(&fc2_push, 0, sizeof(fc2_push));
    fc2_push.rows = 1;
    fc2_push.cols = (uint32_t)rt->D;
    fc2_push.inner = (uint32_t)rt->mlp_hidden;
    fc2_push.has_bias = 0;

    WorldVulkanSiluPush cond_silu_push;
    memset(&cond_silu_push, 0, sizeof(cond_silu_push));
    cond_silu_push.n = (uint32_t)rt->D;

    WorldVulkanLinearPush out_push;
    memset(&out_push, 0, sizeof(out_push));
    out_push.rows = 1;
    out_push.cols = (uint32_t)(2 * rt->D);
    out_push.inner = (uint32_t)rt->D;
    out_push.has_bias = 0;

    WorldVulkanSiluPush layer_bias_push;
    memset(&layer_bias_push, 0, sizeof(layer_bias_push));
    layer_bias_push.n = (uint32_t)rt->D;

    WorldVulkanLinearPush layer_mod_push;
    memset(&layer_mod_push, 0, sizeof(layer_mod_push));
    layer_mod_push.rows = 1;
    layer_mod_push.cols = (uint32_t)(6 * rt->D);
    layer_mod_push.inner = (uint32_t)rt->D;
    layer_mod_push.has_bias = 0;

    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= rt->steps_to_run;
        float sigma = is_cache_pass ? 0.0f : rt->cfg.scheduler_sigmas[pass_idx];
        fill_noise_embedding((float *)rt->noise_mapped, sigma);

        VK_CALL_RET(vkResetFences(rt->device, 1, &rt->fence));
        VK_CALL_RET(vkResetCommandBuffer(rt->command_buffer, 0));
        VkCommandBufferBeginInfo begin_info;
        memset(&begin_info, 0, sizeof(begin_info));
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CALL_RET(vkBeginCommandBuffer(rt->command_buffer, &begin_info));

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_linear_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_linear_pipeline_layout, 0, 1, &rt->denoise_fc1_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_linear_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(fc1_push), &fc1_push);
        vkCmdDispatch(rt->command_buffer, ((uint32_t)rt->mlp_hidden + 15u) / 16u, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_silu_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_silu_pipeline_layout, 0, 1, &rt->denoise_silu_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_silu_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(hidden_silu_push), &hidden_silu_push);
        vkCmdDispatch(rt->command_buffer, ((uint32_t)rt->mlp_hidden + 255u) / 256u, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_linear_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_linear_pipeline_layout, 0, 1, &rt->denoise_fc2_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_linear_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(fc2_push), &fc2_push);
        vkCmdDispatch(rt->command_buffer, ((uint32_t)rt->D + 15u) / 16u, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_silu_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_silu_pipeline_layout, 0, 1, &rt->denoise_cond_silu_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_silu_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(cond_silu_push), &cond_silu_push);
        vkCmdDispatch(rt->command_buffer, ((uint32_t)rt->D + 255u) / 256u, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_linear_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_linear_pipeline_layout, 0, 1, &rt->out_norm_descriptor_set[pass_idx], 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_linear_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(out_push), &out_push);
        vkCmdDispatch(rt->command_buffer, ((uint32_t)(2 * rt->D) + 15u) / 16u, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        if (rt->layer_mod_enabled) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                int table_idx = pass_idx * rt->layers_to_run + layer_idx;

                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_add_bias_silu_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->runtime_add_bias_silu_pipeline_layout, 0, 1,
                        &rt->layer_bias_silu_descriptor_sets[layer_idx], 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->runtime_add_bias_silu_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(layer_bias_push), &layer_bias_push);
                vkCmdDispatch(rt->command_buffer, ((uint32_t)rt->D + 255u) / 256u, 1, 1);
                cmd_shader_barrier(rt->command_buffer);

                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_linear_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->runtime_linear_pipeline_layout, 0, 1,
                        &rt->layer_mod_descriptor_sets[table_idx], 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->runtime_linear_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(layer_mod_push), &layer_mod_push);
                vkCmdDispatch(rt->command_buffer, ((uint32_t)(6 * rt->D) + 15u) / 16u, 1, 1);
                cmd_shader_barrier(rt->command_buffer);
            }
        }

        VK_CALL_RET(vkEndCommandBuffer(rt->command_buffer));
        VkSubmitInfo submit_info;
        memset(&submit_info, 0, sizeof(submit_info));
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &rt->command_buffer;
        VK_CALL_RET(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
        VK_CALL_RET(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));
    }
    fprintf(stderr, "Vulkan scheduler conditioning precomputed: passes=%d out_values=%d layer_mod_values=%d\n",
            rt->total_passes, rt->total_passes * 2 * rt->D,
            rt->layer_mod_enabled ? rt->total_passes * rt->layers_to_run * 6 * rt->D : 0);
    return 0;
}

static size_t vae_elems(int N, int C, int H, int W) {
    return (size_t)N * (size_t)C * (size_t)H * (size_t)W;
}

static VkBuffer vae_scratch_buffer(WorldVulkanRuntime *rt, int idx) {
    if (idx == 0) return rt->vae_buf0_buffer;
    if (idx == 1) return rt->vae_buf1_buffer;
    return rt->vae_buf2_buffer;
}

static void vae_pick_scratch(int cur, int *tmp, int *aux) {
    int first = cur != 0 ? 0 : 1;
    int second = -1;
    if (cur != 0 && first != 0) second = 0;
    if (second < 0 && cur != 1 && first != 1) second = 1;
    if (second < 0 && cur != 2 && first != 2) second = 2;
    if (second < 0) second = 2;
    *tmp = first;
    *aux = second;
}

static int vae_track_elems(WorldVulkanRuntime *rt, size_t elems) {
    if (elems > UINT32_MAX) {
        fprintf(stderr, "Vulkan VAE tensor too large: %zu elements\n", elems);
        return 1;
    }
    if (elems > rt->vae_max_elems) rt->vae_max_elems = elems;
    return 0;
}

static WorldVulkanVaeOp *vae_add_op(
        WorldVulkanRuntime *rt,
        int kind,
        int conv_idx,
        int src,
        int dst,
        int aux,
        int N,
        int C,
        int C_out,
        int H,
        int W,
        int stride) {
    if (rt->vae_op_count >= WORLD_VULKAN_MAX_VAE_OPS) {
        fprintf(stderr, "too many Vulkan VAE ops\n");
        return NULL;
    }
    WorldVulkanVaeOp *op = &rt->vae_ops[rt->vae_op_count++];
    memset(op, 0, sizeof(*op));
    op->kind = kind;
    op->conv_idx = conv_idx;
    op->src = src;
    op->dst = dst;
    op->aux = aux;
    op->N = N;
    op->C = C;
    op->C_out = C_out;
    op->H = H;
    op->W = W;
    op->stride = stride;
    return op;
}

static int vae_add_conv_op(WorldVulkanRuntime *rt, const WorldVaeDecoderWeights *vae,
        int idx, int src, int dst, int N, int H, int W) {
    const WorldVaeConvWeight *conv = &vae->convs[idx];
    if (!conv->weight || conv->out_c <= 0 || conv->in_c <= 0 || conv->kernel <= 0) {
        fprintf(stderr, "missing Vulkan VAE conv weight idx=%d\n", idx);
        return 1;
    }
    if (!vae_add_op(rt, WORLD_VULKAN_VAE_CONV, idx, src, dst, -1,
                N, conv->in_c, conv->out_c, H, W, conv->kernel)) return 1;
    return vae_track_elems(rt, vae_elems(N, conv->out_c, H, W));
}

static int vae_add_relu_op(WorldVulkanRuntime *rt, int src, int N, int C, int H, int W) {
    if (!vae_add_op(rt, WORLD_VULKAN_VAE_RELU, -1, src, src, -1, N, C, C, H, W, 1)) return 1;
    return 0;
}

static int vae_add_memblock_ops(WorldVulkanRuntime *rt, const WorldVaeDecoderWeights *vae,
        int *cur, int N, int C, int H, int W, int conv0, int conv2, int conv4) {
    int tmp = 0;
    int aux = 0;
    vae_pick_scratch(*cur, &tmp, &aux);
    if (!vae_add_op(rt, WORLD_VULKAN_VAE_CONCAT_PAST, -1, *cur, aux, -1,
                N, C, C * 2, H, W, 1)) return 1;
    if (vae_track_elems(rt, vae_elems(N, C * 2, H, W))) return 1;
    if (vae_add_conv_op(rt, vae, conv0, aux, tmp, N, H, W)) return 1;
    if (vae_add_relu_op(rt, tmp, N, C, H, W)) return 1;
    if (vae_add_conv_op(rt, vae, conv2, tmp, aux, N, H, W)) return 1;
    if (vae_add_relu_op(rt, aux, N, C, H, W)) return 1;
    if (vae_add_conv_op(rt, vae, conv4, aux, tmp, N, H, W)) return 1;
    if (!vae_add_op(rt, WORLD_VULKAN_VAE_ADD_RELU, -1, *cur, aux, tmp,
                N, C, C, H, W, 1)) return 1;
    if (vae_track_elems(rt, vae_elems(N, C, H, W))) return 1;
    *cur = aux;
    return 0;
}

static int build_runtime_vae_ops(WorldVulkanRuntime *rt, const WorldVaeDecoderWeights *vae) {
    rt->vae_op_count = 0;
    rt->vae_max_elems = 0;
    int N = 4;
    int C = rt->cfg.channels;
    int H = rt->cfg.height * rt->cfg.patch_h;
    int W = rt->cfg.width * rt->cfg.patch_w;
    int cur = 0;
    int tmp = 0;
    int aux = 0;

    if (!vae_add_op(rt, WORLD_VULKAN_VAE_REPEAT_CLAMP, -1, -1, cur, -1,
                N, C, C, H, W, 1)) return 1;
    if (vae_track_elems(rt, vae_elems(N, C, H, W))) return 1;

#define VAE_CONV_TO(idx, out_c) do { \
    vae_pick_scratch(cur, &tmp, &aux); \
    if (vae_add_conv_op(rt, vae, (idx), cur, tmp, N, H, W)) return 1; \
    cur = tmp; \
    C = (out_c); \
} while (0)
#define VAE_RELU() do { \
    if (vae_add_relu_op(rt, cur, N, C, H, W)) return 1; \
} while (0)
#define VAE_MEMBLOCK(a, b, c) do { \
    if (vae_add_memblock_ops(rt, vae, &cur, N, C, H, W, (a), (b), (c))) return 1; \
} while (0)
#define VAE_UPSAMPLE2() do { \
    vae_pick_scratch(cur, &tmp, &aux); \
    if (!vae_add_op(rt, WORLD_VULKAN_VAE_UPSAMPLE2, -1, cur, tmp, -1, N, C, C, H, W, 1)) return 1; \
    if (vae_track_elems(rt, vae_elems(N, C, H * 2, W * 2))) return 1; \
    cur = tmp; \
    H *= 2; \
    W *= 2; \
} while (0)
#define VAE_TGROW(idx, stride_value) do { \
    int _stride = (stride_value); \
    vae_pick_scratch(cur, &tmp, &aux); \
    if (vae_add_conv_op(rt, vae, (idx), cur, tmp, N, H, W)) return 1; \
    if (_stride == 1) { \
        cur = tmp; \
    } else { \
        if (!vae_add_op(rt, WORLD_VULKAN_VAE_TGROW, -1, tmp, cur, -1, N, C, C, H, W, _stride)) return 1; \
        if (vae_track_elems(rt, vae_elems(N * _stride, C, H, W))) return 1; \
        N *= _stride; \
    } \
} while (0)

    VAE_CONV_TO(WORLD_VAE_DEC_CONV_IN, 256);
    VAE_RELU();
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB3_0, WORLD_VAE_DEC_MB3_2, WORLD_VAE_DEC_MB3_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB4_0, WORLD_VAE_DEC_MB4_2, WORLD_VAE_DEC_MB4_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB5_0, WORLD_VAE_DEC_MB5_2, WORLD_VAE_DEC_MB5_4);
    VAE_UPSAMPLE2();
    VAE_TGROW(WORLD_VAE_DEC_TGROW7, 1);

    VAE_CONV_TO(WORLD_VAE_DEC_CONV8, 128);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB9_0, WORLD_VAE_DEC_MB9_2, WORLD_VAE_DEC_MB9_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB10_0, WORLD_VAE_DEC_MB10_2, WORLD_VAE_DEC_MB10_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB11_0, WORLD_VAE_DEC_MB11_2, WORLD_VAE_DEC_MB11_4);
    VAE_UPSAMPLE2();
    VAE_TGROW(WORLD_VAE_DEC_TGROW13, 2);

    VAE_CONV_TO(WORLD_VAE_DEC_CONV14, 64);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB15_0, WORLD_VAE_DEC_MB15_2, WORLD_VAE_DEC_MB15_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB16_0, WORLD_VAE_DEC_MB16_2, WORLD_VAE_DEC_MB16_4);
    VAE_MEMBLOCK(WORLD_VAE_DEC_MB17_0, WORLD_VAE_DEC_MB17_2, WORLD_VAE_DEC_MB17_4);
    VAE_UPSAMPLE2();
    VAE_TGROW(WORLD_VAE_DEC_TGROW19, 2);

    VAE_CONV_TO(WORLD_VAE_DEC_CONV20, 64);
    VAE_RELU();
    const WorldVaeConvWeight *conv_out = &vae->convs[WORLD_VAE_DEC_CONV_OUT];
    if (!vae_add_op(rt, WORLD_VULKAN_VAE_CONV_OUT_RGBA, WORLD_VAE_DEC_CONV_OUT, cur, -1, -1,
                N, C, 12, H, W, conv_out->kernel)) return 1;

#undef VAE_CONV_TO
#undef VAE_RELU
#undef VAE_MEMBLOCK
#undef VAE_UPSAMPLE2
#undef VAE_TGROW

    return 0;
}

static int create_runtime_vae_op_descriptor(WorldVulkanRuntime *rt, WorldVulkanVaeOp *op) {
    VkBuffer buffers[4];
    VkDeviceSize sizes[4];
    memset(buffers, 0, sizeof(buffers));
    memset(sizes, 0, sizeof(sizes));
    switch (op->kind) {
        case WORLD_VULKAN_VAE_REPEAT_CLAMP:
            buffers[0] = rt->latent_buffer;
            buffers[1] = vae_scratch_buffer(rt, op->dst);
            sizes[0] = rt->latent_elems * sizeof(float);
            sizes[1] = rt->vae_scratch_bytes;
            return create_storage_descriptor_set(rt, rt->taehv_repeat_clamp_set_layout,
                    2, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_CONV: {
            const int idx = op->conv_idx;
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = rt->vae_weight_buffer[idx];
            buffers[2] = rt->vae_bias_buffer[idx] ? rt->vae_bias_buffer[idx] : rt->dummy_bias_buffer;
            buffers[3] = vae_scratch_buffer(rt, op->dst);
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->vae_weight_bytes[idx];
            sizes[2] = rt->vae_bias_buffer[idx] ? rt->vae_bias_bytes[idx] : sizeof(float);
            sizes[3] = rt->vae_scratch_bytes;
            VkDescriptorSetLayout conv_layout = rt->taehv_conv_set_layout;
            if (vae_conv_uses_1x1_c4(op)) {
                conv_layout = rt->taehv_conv_1x1_c4_set_layout;
            } else if ((op->C_out % 4) == 0) {
                conv_layout = rt->taehv_conv_c4_set_layout;
            }
            return create_storage_descriptor_set(rt, conv_layout,
                    4, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        }
        case WORLD_VULKAN_VAE_RELU:
            buffers[0] = vae_scratch_buffer(rt, op->src);
            sizes[0] = rt->vae_scratch_bytes;
            return create_storage_descriptor_set(rt, rt->taehv_relu_set_layout,
                    1, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_CONCAT_PAST:
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = vae_scratch_buffer(rt, op->dst);
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->vae_scratch_bytes;
            return create_storage_descriptor_set(rt, rt->taehv_concat_past_set_layout,
                    2, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_ADD_RELU:
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = vae_scratch_buffer(rt, op->aux);
            buffers[2] = vae_scratch_buffer(rt, op->dst);
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->vae_scratch_bytes;
            sizes[2] = rt->vae_scratch_bytes;
            return create_storage_descriptor_set(rt, rt->taehv_add_relu_set_layout,
                    3, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_UPSAMPLE2:
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = vae_scratch_buffer(rt, op->dst);
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->vae_scratch_bytes;
            return create_storage_descriptor_set(rt, rt->taehv_upsample2_set_layout,
                    2, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_TGROW:
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = vae_scratch_buffer(rt, op->dst);
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->vae_scratch_bytes;
            return create_storage_descriptor_set(rt, rt->taehv_tgrow_set_layout,
                    2, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_PIXEL_SHUFFLE:
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = rt->output_buffer;
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->pixel_count * sizeof(uint32_t);
            return create_storage_descriptor_set(rt, rt->taehv_pixel_shuffle_set_layout,
                    2, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        case WORLD_VULKAN_VAE_CONV_OUT_RGBA: {
            const int idx = op->conv_idx;
            buffers[0] = vae_scratch_buffer(rt, op->src);
            buffers[1] = rt->vae_weight_buffer[idx];
            buffers[2] = rt->vae_bias_buffer[idx] ? rt->vae_bias_buffer[idx] : rt->dummy_bias_buffer;
            buffers[3] = rt->output_buffer;
            sizes[0] = rt->vae_scratch_bytes;
            sizes[1] = rt->vae_weight_bytes[idx];
            sizes[2] = rt->vae_bias_buffer[idx] ? rt->vae_bias_bytes[idx] : sizeof(float);
            sizes[3] = rt->pixel_count * sizeof(uint32_t);
            return create_storage_descriptor_set(rt, rt->taehv_conv_out_rgba_set_layout,
                    4, buffers, sizes, NULL, &op->descriptor_pool, &op->descriptor_set);
        }
        default:
            return 1;
    }
}

static int create_runtime_vae(WorldVulkanRuntime *rt, const WorldVaeDecoderWeights *vae) {
    if (!vae || !rt->model_slice_enabled || !rt->latent_buffer || !rt->dummy_bias_buffer) return 0;
    if (build_runtime_vae_ops(rt, vae)) return 1;
    rt->vae_scratch_bytes = rt->vae_max_elems * sizeof(float);
    if (rt->vae_scratch_bytes == 0) return 0;

    if (create_device_buffer(rt, rt->vae_scratch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->vae_buf0_buffer, &rt->vae_buf0_memory)) return 1;
    if (create_device_buffer(rt, rt->vae_scratch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->vae_buf1_buffer, &rt->vae_buf1_memory)) return 1;
    if (create_device_buffer(rt, rt->vae_scratch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->vae_buf2_buffer, &rt->vae_buf2_memory)) return 1;

    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
        const WorldVaeConvWeight *conv = &vae->convs[i];
        if (!conv->weight || conv->out_c <= 0 || conv->in_c <= 0 || conv->kernel <= 0) {
            fprintf(stderr, "missing Vulkan VAE weights for conv %d\n", i);
            return 1;
        }
        rt->vae_weight_bytes[i] = (size_t)conv->out_c * conv->in_c *
            (size_t)conv->kernel * conv->kernel * sizeof(float);
        if (create_device_buffer(rt, rt->vae_weight_bytes[i],
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    &rt->vae_weight_buffer[i], &rt->vae_weight_memory[i])) return 1;
        if (upload_to_device_buffer(rt, rt->vae_weight_buffer[i], conv->weight,
                    (VkDeviceSize)rt->vae_weight_bytes[i])) return 1;
        if (conv->has_bias) {
            if (!conv->bias) {
                fprintf(stderr, "missing Vulkan VAE bias for conv %d\n", i);
                return 1;
            }
            rt->vae_bias_bytes[i] = (size_t)conv->out_c * sizeof(float);
            if (create_device_buffer(rt, rt->vae_bias_bytes[i],
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        &rt->vae_bias_buffer[i], &rt->vae_bias_memory[i])) return 1;
            if (upload_to_device_buffer(rt, rt->vae_bias_buffer[i], conv->bias,
                        (VkDeviceSize)rt->vae_bias_bytes[i])) return 1;
        }
    }

    if (create_storage_pipeline(rt, "taehv_repeat_clamp_f32.comp", 2, sizeof(WorldVulkanTaehvRepeatClampPush),
                &rt->taehv_repeat_clamp_shader, &rt->taehv_repeat_clamp_set_layout,
                &rt->taehv_repeat_clamp_pipeline_layout, &rt->taehv_repeat_clamp_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_conv2d_nchw_f32.comp", 4, sizeof(WorldVulkanTaehvConv2dPush),
                &rt->taehv_conv_shader, &rt->taehv_conv_set_layout,
                &rt->taehv_conv_pipeline_layout, &rt->taehv_conv_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_conv2d_nchw_c4_f32.comp", 4, sizeof(WorldVulkanTaehvConv2dPush),
                &rt->taehv_conv_c4_shader, &rt->taehv_conv_c4_set_layout,
                &rt->taehv_conv_c4_pipeline_layout, &rt->taehv_conv_c4_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_conv2d_nchw_1x1_c4_f32.comp", 4, sizeof(WorldVulkanTaehvConv2dPush),
                &rt->taehv_conv_1x1_c4_shader, &rt->taehv_conv_1x1_c4_set_layout,
                &rt->taehv_conv_1x1_c4_pipeline_layout, &rt->taehv_conv_1x1_c4_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_conv_out_rgba_f32.comp", 4, sizeof(WorldVulkanTaehvConv2dPush),
                &rt->taehv_conv_out_rgba_shader, &rt->taehv_conv_out_rgba_set_layout,
                &rt->taehv_conv_out_rgba_pipeline_layout, &rt->taehv_conv_out_rgba_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_relu_f32.comp", 1, sizeof(WorldVulkanSiluPush),
                &rt->taehv_relu_shader, &rt->taehv_relu_set_layout,
                &rt->taehv_relu_pipeline_layout, &rt->taehv_relu_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_concat_past_nchw_f32.comp", 2, sizeof(WorldVulkanTaehvNchwPush),
                &rt->taehv_concat_past_shader, &rt->taehv_concat_past_set_layout,
                &rt->taehv_concat_past_pipeline_layout, &rt->taehv_concat_past_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_add_relu_f32.comp", 3, sizeof(WorldVulkanSiluPush),
                &rt->taehv_add_relu_shader, &rt->taehv_add_relu_set_layout,
                &rt->taehv_add_relu_pipeline_layout, &rt->taehv_add_relu_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_upsample2_nchw_f32.comp", 2, sizeof(WorldVulkanTaehvNchwPush),
                &rt->taehv_upsample2_shader, &rt->taehv_upsample2_set_layout,
                &rt->taehv_upsample2_pipeline_layout, &rt->taehv_upsample2_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_tgrow_reshape_nchw_f32.comp", 2, sizeof(WorldVulkanTaehvTgrowPush),
                &rt->taehv_tgrow_shader, &rt->taehv_tgrow_set_layout,
                &rt->taehv_tgrow_pipeline_layout, &rt->taehv_tgrow_pipeline)) return 1;
    if (create_storage_pipeline(rt, "taehv_pixel_shuffle_rgba_f32.comp", 2, sizeof(WorldVulkanTaehvPixelShufflePush),
                &rt->taehv_pixel_shuffle_shader, &rt->taehv_pixel_shuffle_set_layout,
                &rt->taehv_pixel_shuffle_pipeline_layout, &rt->taehv_pixel_shuffle_pipeline)) return 1;

    for (int i = 0; i < rt->vae_op_count; ++i) {
        if (create_runtime_vae_op_descriptor(rt, &rt->vae_ops[i])) return 1;
    }

    rt->vae_enabled = 1;
    fprintf(stderr, "Vulkan VAE F32/NCHW decode enabled: ops=%d device_scratch=%.2f MiB RGB=%dx%d frames=%d\n",
            rt->vae_op_count, (double)(3 * rt->vae_scratch_bytes) / (1024.0 * 1024.0),
            rt->width, rt->height, rt->frames);
    return 0;
}

static void record_runtime_vae_decode(WorldVulkanRuntime *rt) {
#define PROFILE_DISPATCH(label, gx, gy, gz) do { \
    uint32_t _profile_event = profile_begin_dispatch(rt, (label)); \
    vkCmdDispatch(rt->command_buffer, (gx), (gy), (gz)); \
    profile_end_dispatch(rt, _profile_event); \
} while (0)

    for (int i = 0; i < rt->vae_op_count; ++i) {
        WorldVulkanVaeOp *op = &rt->vae_ops[i];
        switch (op->kind) {
            case WORLD_VULKAN_VAE_REPEAT_CLAMP: {
                WorldVulkanTaehvRepeatClampPush push = {(uint32_t)op->C, (uint32_t)op->H, (uint32_t)op->W};
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_repeat_clamp_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_repeat_clamp_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_repeat_clamp_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.repeat_clamp",
                        ((uint32_t)vae_elems(op->N, op->C, op->H, op->W) + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_CONV: {
                WorldVulkanTaehvConv2dPush push;
                memset(&push, 0, sizeof(push));
                push.N = (uint32_t)op->N;
                push.C_in = (uint32_t)op->C;
                push.C_out = (uint32_t)op->C_out;
                push.H = (uint32_t)op->H;
                push.W = (uint32_t)op->W;
                push.K = (uint32_t)op->stride;
                push.has_bias = rt->vae_bias_buffer[op->conv_idx] ? 1u : 0u;
                VkPipeline conv_pipeline = rt->taehv_conv_pipeline;
                VkPipelineLayout conv_layout = rt->taehv_conv_pipeline_layout;
                if (vae_conv_uses_1x1_c4(op)) {
                    conv_pipeline = rt->taehv_conv_1x1_c4_pipeline;
                    conv_layout = rt->taehv_conv_1x1_c4_pipeline_layout;
                } else if ((op->C_out % 4) == 0) {
                    conv_pipeline = rt->taehv_conv_c4_pipeline;
                    conv_layout = rt->taehv_conv_c4_pipeline_layout;
                }
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, conv_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        conv_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, conv_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                uint32_t conv_work = (op->C_out % 4) == 0 ?
                    (uint32_t)vae_elems(op->N, op->C_out / 4, op->H, op->W) :
                    (uint32_t)vae_elems(op->N, op->C_out, op->H, op->W);
                PROFILE_DISPATCH(vae_conv_uses_1x1_c4(op) ? "vae.conv_1x1_c4" :
                        ((op->C_out % 4) == 0 ? "vae.conv_c4" : "vae.conv"),
                        (conv_work + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_RELU: {
                WorldVulkanSiluPush push = {(uint32_t)vae_elems(op->N, op->C, op->H, op->W)};
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_relu_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_relu_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_relu_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.relu", (push.n + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_CONCAT_PAST: {
                WorldVulkanTaehvNchwPush push = {(uint32_t)op->N, (uint32_t)op->C, (uint32_t)op->H, (uint32_t)op->W};
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_concat_past_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_concat_past_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_concat_past_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.concat_past",
                        ((uint32_t)vae_elems(op->N, op->C * 2, op->H, op->W) + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_ADD_RELU: {
                WorldVulkanSiluPush push = {(uint32_t)vae_elems(op->N, op->C, op->H, op->W)};
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_add_relu_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_add_relu_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_add_relu_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.add_relu", (push.n + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_UPSAMPLE2: {
                WorldVulkanTaehvNchwPush push = {(uint32_t)op->N, (uint32_t)op->C, (uint32_t)op->H, (uint32_t)op->W};
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_upsample2_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_upsample2_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_upsample2_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.upsample2",
                        ((uint32_t)vae_elems(op->N, op->C, op->H * 2, op->W * 2) + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_TGROW: {
                WorldVulkanTaehvTgrowPush push = {
                    (uint32_t)op->N, (uint32_t)op->C, (uint32_t)op->H, (uint32_t)op->W, (uint32_t)op->stride
                };
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_tgrow_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_tgrow_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_tgrow_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.tgrow",
                        ((uint32_t)vae_elems(op->N * op->stride, op->C, op->H, op->W) + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_PIXEL_SHUFFLE: {
                WorldVulkanTaehvPixelShufflePush push = {(uint32_t)op->N, (uint32_t)op->H, (uint32_t)op->W};
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_pixel_shuffle_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_pixel_shuffle_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_pixel_shuffle_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.pixel_shuffle",
                        ((uint32_t)rt->pixel_count + 255u) / 256u, 1, 1);
                break;
            }
            case WORLD_VULKAN_VAE_CONV_OUT_RGBA: {
                WorldVulkanTaehvConv2dPush push;
                memset(&push, 0, sizeof(push));
                push.N = (uint32_t)op->N;
                push.C_in = (uint32_t)op->C;
                push.C_out = (uint32_t)op->C_out;
                push.H = (uint32_t)op->H;
                push.W = (uint32_t)op->W;
                push.K = (uint32_t)op->stride;
                push.has_bias = rt->vae_bias_buffer[op->conv_idx] ? 1u : 0u;
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->taehv_conv_out_rgba_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->taehv_conv_out_rgba_pipeline_layout, 0, 1, &op->descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->taehv_conv_out_rgba_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                PROFILE_DISPATCH("vae.conv_out_rgba",
                        ((uint32_t)rt->pixel_count + 127u) / 128u, 1, 1);
                break;
            }
        }
        cmd_shader_barrier(rt->command_buffer);
    }
#undef PROFILE_DISPATCH
}

static int create_runtime_model_slice(
        WorldVulkanRuntime *rt,
        const WorldModelProbeWeights *weights) {
    if (!weights ||
            !weights->patchify_weight ||
            !weights->denoise_fc1_weight ||
            !weights->denoise_fc2_weight ||
            !weights->ctrl_emb_fc1_weight ||
            !weights->ctrl_emb_fc2_weight ||
            !weights->out_norm_fc_weight ||
            !weights->unpatchify_weight ||
            !weights->unpatchify_bias) {
        return 0;
    }

    rt->C = rt->cfg.channels;
    rt->H = rt->cfg.height * rt->cfg.patch_h;
    rt->W = rt->cfg.width * rt->cfg.patch_w;
    rt->D = rt->cfg.d_model;
    rt->ph = rt->cfg.patch_h;
    rt->pw = rt->cfg.patch_w;
    rt->T = rt->cfg.height * rt->cfg.width;
    rt->out_dim = rt->C * rt->ph * rt->pw;
    rt->mlp_hidden = rt->D * rt->cfg.mlp_ratio;
    rt->ctrl_dim = rt->cfg.n_buttons + 3;
    if (rt->cfg.n_heads <= 0 || rt->D % rt->cfg.n_heads != 0) {
        fprintf(stderr, "invalid Vulkan qkv head config D=%d n_heads=%d\n", rt->D, rt->cfg.n_heads);
        return 1;
    }
    rt->d_head = rt->D / rt->cfg.n_heads;
    rt->kv_dim = rt->cfg.n_kv_heads * rt->d_head;
    rt->qkv_dim = rt->D + 2 * rt->kv_dim;
    rt->d_xy = rt->d_head / 8;
    rt->d_t = rt->d_head / 4;
    {
        int latent_fps = rt->cfg.temporal_compression > 0 ?
            rt->cfg.inference_fps / rt->cfg.temporal_compression : 0;
        rt->frame_stride = latent_fps > 0 ? rt->cfg.base_fps / latent_fps : 1;
        if (rt->frame_stride <= 0) rt->frame_stride = 1;
    }
    rt->rms_eps = 1.0e-6f;
    rt->latent_elems = (size_t)rt->C * rt->H * rt->W;
    rt->token_elems = (size_t)rt->T * rt->D;

    if (rt->layers_to_run > 0) {
        if (!weights->layers) return 1;
        rt->layer_has_ctrl = (int *)calloc((size_t)rt->layers_to_run, sizeof(int));
        rt->layer_v_lamb = (float *)calloc((size_t)rt->layers_to_run, sizeof(float));
        rt->cache_ring_lengths = (int *)calloc((size_t)rt->layers_to_run, sizeof(int));
        rt->cache_capacities = (int *)calloc((size_t)rt->layers_to_run, sizeof(int));
        rt->cache_pinned_dilations = (int *)calloc((size_t)rt->layers_to_run, sizeof(int));
        rt->cache_is_global = (int *)calloc((size_t)rt->layers_to_run, sizeof(int));
        rt->cache_kv_offsets = (size_t *)calloc((size_t)rt->layers_to_run, sizeof(size_t));
        rt->cache_meta_offsets = (size_t *)calloc((size_t)rt->layers_to_run, sizeof(size_t));
        if (!rt->layer_has_ctrl || !rt->layer_v_lamb ||
                !rt->cache_ring_lengths || !rt->cache_capacities ||
                !rt->cache_pinned_dilations || !rt->cache_is_global ||
                !rt->cache_kv_offsets || !rt->cache_meta_offsets) return 1;
        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            const WorldLayerWeights *lw = &weights->layers[layer_idx];
            if (!lw->cond_bias ||
                    !lw->attn_cond_s_weight ||
                    !lw->attn_cond_b_weight ||
                    !lw->attn_cond_g_weight ||
                    !lw->mlp_cond_s_weight ||
                    !lw->mlp_cond_b_weight ||
                    !lw->mlp_cond_g_weight) {
                fprintf(stderr, "missing Vulkan layer modulation weights for layer %d\n", layer_idx);
                return 1;
            }
            rt->layer_has_ctrl[layer_idx] = lw->has_ctrl ? 1 : 0;
            rt->layer_v_lamb[layer_idx] = lw->v_lamb ? lw->v_lamb[0] : 0.0f;
        }
        rt->layer_qkv_enabled = (rt->d_head <= 256 && rt->d_xy > 0 && rt->d_t > 0);
        for (int layer_idx = 0; layer_idx < rt->layers_to_run && rt->layer_qkv_enabled; ++layer_idx) {
            const WorldLayerWeights *lw = &weights->layers[layer_idx];
            if (!lw->q_proj_weight || !lw->k_proj_weight || !lw->v_proj_weight) {
                rt->layer_qkv_enabled = 0;
            }
        }
        if (!rt->layer_qkv_enabled) {
            fprintf(stderr, "warning: Vulkan runtime layer QKV disabled; missing q/k/v weights or unsupported d_head=%d\n",
                    rt->d_head);
        }
        if (rt->layer_qkv_enabled && rt->cfg.n_kv_heads > 0 && rt->cfg.n_heads % rt->cfg.n_kv_heads == 0) {
            int period = rt->cfg.global_attn_period > 0 ? rt->cfg.global_attn_period : 1;
            int offset = positive_mod_int(rt->cfg.global_attn_offset, period);
            rt->layer_attention_enabled = 1;
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                int is_global = ((layer_idx - offset) % period) == 0;
                int window = is_global ? rt->cfg.global_window : rt->cfg.local_window;
                int pinned_dilation = is_global ? rt->cfg.global_pinned_dilation : 1;
                if (window <= 0 || pinned_dilation <= 0 ||
                        ((window % pinned_dilation) != 0)) {
                    rt->layer_attention_enabled = 0;
                    break;
                }
                rt->cache_is_global[layer_idx] = is_global;
                rt->cache_ring_lengths[layer_idx] = window * rt->T;
                rt->cache_capacities[layer_idx] = rt->cache_ring_lengths[layer_idx] + rt->T;
                rt->cache_pinned_dilations[layer_idx] = pinned_dilation;
                rt->cache_kv_offsets[layer_idx] = rt->cache_total_kv_elems;
                rt->cache_meta_offsets[layer_idx] = rt->cache_total_meta_elems;
                rt->cache_total_kv_elems += (size_t)rt->cfg.n_kv_heads *
                    (size_t)rt->cache_capacities[layer_idx] * (size_t)rt->d_head;
                rt->cache_total_meta_elems += (size_t)rt->cache_capacities[layer_idx];
            }
            if (rt->layer_attention_enabled && rt->layers_to_run > 0) {
                rt->cache_ring_length = rt->cache_ring_lengths[0];
                rt->cache_capacity = rt->cache_capacities[0];
                rt->cache_pinned_dilation = rt->cache_pinned_dilations[0];
            } else {
                rt->cache_total_kv_elems = 0;
                rt->cache_total_meta_elems = 0;
            }
        }
        if (rt->layer_qkv_enabled && !rt->layer_attention_enabled) {
            fprintf(stderr, "warning: Vulkan runtime layer attention disabled; invalid cache config\n");
        }
        if (rt->layer_attention_enabled) {
            rt->layer_attn_out_enabled = 1;
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                if (!weights->layers[layer_idx].out_proj_weight) {
                    rt->layer_attn_out_enabled = 0;
                    break;
                }
            }
            if (!rt->layer_attn_out_enabled) {
                fprintf(stderr, "warning: Vulkan runtime attention out projection disabled; missing out_proj weight\n");
            }
        }
        if (rt->layer_attn_out_enabled) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                const WorldLayerWeights *lw = &weights->layers[layer_idx];
                if (!lw->has_ctrl) continue;
                if (!lw->ctrl_fc1_c_weight || !lw->ctrl_fc1_x_weight || !lw->ctrl_fc2_weight) {
                    rt->layer_ctrl_enabled = 0;
                    fprintf(stderr, "warning: Vulkan runtime control fusion disabled; incomplete ctrl weights on layer %d\n",
                            layer_idx);
                    break;
                }
                rt->layer_ctrl_enabled = 1;
            }
        }
        if (rt->layer_attn_out_enabled) {
            rt->layer_mlp_enabled = 1;
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                if (!weights->layers[layer_idx].dit_mlp_fc1_weight ||
                        !weights->layers[layer_idx].dit_mlp_fc2_weight) {
                    rt->layer_mlp_enabled = 0;
                    break;
                }
            }
            if (!rt->layer_mlp_enabled) {
                fprintf(stderr, "warning: Vulkan runtime DiT MLP disabled; missing MLP weights\n");
            }
        }
    }

    {
        const char *wf16_env = getenv("WORLD_VULKAN_LINEAR_WF16_COOPMAT");
        if (wf16_env && wf16_env[0] == '1') {
            if (rt->shader_float16_enabled && rt->storage_16bit_enabled && rt->cooperative_matrix_enabled) {
                rt->runtime_linear_wf16_coopmat_enabled = 1;
            } else {
                fprintf(stderr,
                        "warning: WORLD_VULKAN_LINEAR_WF16_COOPMAT=1 ignored; shaderFloat16=%d storage16=%d cooperativeMatrix=%d\n",
                        rt->shader_float16_enabled, rt->storage_16bit_enabled, rt->cooperative_matrix_enabled);
            }
        }
    }

    size_t latent_bytes = rt->latent_elems * sizeof(float);
    size_t token_bytes = rt->token_elems * sizeof(float);
    size_t patch_weight_bytes = (size_t)rt->D * rt->C * rt->ph * rt->pw * sizeof(float);
    size_t unpatch_bias_bytes = (size_t)rt->C * sizeof(float);
    size_t control_bytes = (size_t)rt->ctrl_dim * sizeof(float);
    size_t ctrl_hidden_bytes = (size_t)rt->mlp_hidden * sizeof(float);
    size_t ctrl_emb_bytes = (size_t)rt->D * sizeof(float);
    size_t ctrl_fc1_weight_bytes = (size_t)rt->mlp_hidden * rt->ctrl_dim * sizeof(float);
    size_t ctrl_fc2_weight_bytes = (size_t)rt->D * rt->mlp_hidden * sizeof(float);
    size_t noise_bytes = 512u * sizeof(float);
    size_t denoise_fc1_weight_bytes = (size_t)rt->mlp_hidden * 512u * sizeof(float);
    size_t denoise_fc2_weight_bytes = (size_t)rt->D * rt->mlp_hidden * sizeof(float);
    size_t out_norm_weight_bytes = (size_t)2 * rt->D * rt->D * sizeof(float);
    size_t out_mod_pass_bytes = (size_t)2 * rt->D * sizeof(float);
    size_t out_mod_table_bytes = (size_t)rt->total_passes * out_mod_pass_bytes;
    size_t layer_cond_bias_bytes = (size_t)rt->layers_to_run * rt->D * sizeof(float);
    size_t layer_cond_proj_weight_bytes = (size_t)rt->layers_to_run * 6 * rt->D * rt->D * sizeof(float);
    size_t layer_mod_pass_layer_bytes = (size_t)6 * rt->D * sizeof(float);
    size_t layer_mod_table_bytes = (size_t)rt->total_passes * rt->layers_to_run * layer_mod_pass_layer_bytes;
    size_t qkv_proj_weight_layer_bytes = (size_t)rt->qkv_dim * rt->D * sizeof(float);
    size_t qkv_proj_weight_bytes = (size_t)rt->layers_to_run * qkv_proj_weight_layer_bytes;
    size_t qkv_proj_weight_layer_f16_bytes = (size_t)rt->qkv_dim * rt->D * sizeof(uint16_t);
    size_t qkv_proj_weight_f16_bytes = (size_t)rt->layers_to_run * qkv_proj_weight_layer_f16_bytes;
    size_t qkv_raw_bytes = (size_t)rt->T * rt->qkv_dim * sizeof(float);
    size_t q_rope_bytes = (size_t)rt->T * rt->D * sizeof(float);
    size_t kv_rope_bytes = (size_t)rt->T * rt->kv_dim * sizeof(float);
    size_t pos_bytes = (size_t)rt->T * sizeof(uint32_t);
    size_t xy_bytes = (size_t)rt->d_xy * sizeof(float);
    size_t inv_t_bytes = (size_t)rt->d_t * sizeof(float);
    size_t cache_kv_bytes = rt->cache_total_kv_elems * sizeof(float);
    size_t cache_meta_bytes = rt->cache_total_meta_elems * sizeof(uint32_t);
    size_t cache_count_bytes = (size_t)rt->layers_to_run * sizeof(uint32_t);
    size_t attn_out_proj_weight_layer_bytes = (size_t)rt->D * rt->D * sizeof(float);
    size_t attn_out_proj_weight_bytes = (size_t)rt->layers_to_run * attn_out_proj_weight_layer_bytes;
    size_t attn_out_proj_weight_layer_f16_bytes = (size_t)rt->D * rt->D * sizeof(uint16_t);
    size_t attn_out_proj_weight_f16_bytes = (size_t)rt->layers_to_run * attn_out_proj_weight_layer_f16_bytes;
    size_t ctrl_layer_weight_bytes = (size_t)rt->D * rt->D * sizeof(float);
    size_t ctrl_layer_weight_table_bytes = (size_t)rt->layers_to_run * ctrl_layer_weight_bytes;
    size_t ctrl_cond_table_bytes = (size_t)rt->layers_to_run * ctrl_emb_bytes;
    size_t dit_mlp_fc1_weight_layer_bytes = (size_t)rt->mlp_hidden * rt->D * sizeof(float);
    size_t dit_mlp_fc2_weight_layer_bytes = (size_t)rt->D * rt->mlp_hidden * sizeof(float);
    size_t dit_mlp_fc1_weight_bytes = (size_t)rt->layers_to_run * dit_mlp_fc1_weight_layer_bytes;
    size_t dit_mlp_fc2_weight_bytes = (size_t)rt->layers_to_run * dit_mlp_fc2_weight_layer_bytes;
    size_t dit_mlp_fc1_weight_layer_f16_bytes = (size_t)rt->mlp_hidden * rt->D * sizeof(uint16_t);
    size_t dit_mlp_fc2_weight_layer_f16_bytes = (size_t)rt->D * rt->mlp_hidden * sizeof(uint16_t);
    size_t dit_mlp_fc1_weight_f16_bytes = (size_t)rt->layers_to_run * dit_mlp_fc1_weight_layer_f16_bytes;
    size_t dit_mlp_fc2_weight_f16_bytes = (size_t)rt->layers_to_run * dit_mlp_fc2_weight_layer_f16_bytes;
    size_t mlp_hidden_token_bytes = (size_t)rt->T * rt->mlp_hidden * sizeof(float);
    size_t mlp_hidden_token_f16_bytes = (size_t)rt->T * rt->mlp_hidden * sizeof(uint16_t);
    size_t pass_layer_count = (size_t)rt->total_passes * (size_t)rt->layers_to_run;

    if (create_host_buffer(rt, latent_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->latent_buffer, &rt->latent_memory, &rt->latent_mapped)) return 1;
    if (create_host_buffer(rt, control_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->control_buffer, &rt->control_memory, &rt->control_mapped)) return 1;
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)ctrl_fc1_weight_bytes, weights->ctrl_emb_fc1_weight,
                &rt->ctrl_fc1_weight_buffer, &rt->ctrl_fc1_weight_memory)) return 1;
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)ctrl_fc2_weight_bytes, weights->ctrl_emb_fc2_weight,
                &rt->ctrl_fc2_weight_buffer, &rt->ctrl_fc2_weight_memory)) return 1;
    if (create_host_buffer(rt, ctrl_hidden_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->ctrl_hidden_buffer, &rt->ctrl_hidden_memory, &rt->ctrl_hidden_mapped)) return 1;
    if (create_host_buffer(rt, ctrl_emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->ctrl_emb_buffer, &rt->ctrl_emb_memory, &rt->ctrl_emb_mapped)) return 1;
    if (create_host_buffer(rt, ctrl_emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->ctrl_emb_norm_buffer, &rt->ctrl_emb_norm_memory, &rt->ctrl_emb_norm_mapped)) return 1;
    if (create_host_buffer(rt, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->dummy_bias_buffer, &rt->dummy_bias_memory, &rt->dummy_bias_mapped)) return 1;
    if (create_host_buffer(rt, ctrl_emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->rms_weight_buffer, &rt->rms_weight_memory, &rt->rms_weight_mapped)) return 1;
    if (create_host_buffer(rt, noise_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->noise_buffer, &rt->noise_memory, &rt->noise_mapped)) return 1;
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)denoise_fc1_weight_bytes, weights->denoise_fc1_weight,
                &rt->denoise_fc1_weight_buffer, &rt->denoise_fc1_weight_memory)) return 1;
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)denoise_fc2_weight_bytes, weights->denoise_fc2_weight,
                &rt->denoise_fc2_weight_buffer, &rt->denoise_fc2_weight_memory)) return 1;
    if (create_host_buffer(rt, ctrl_hidden_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->noise_hidden_buffer, &rt->noise_hidden_memory, &rt->noise_hidden_mapped)) return 1;
    if (create_host_buffer(rt, ctrl_emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->cond_buffer, &rt->cond_memory, &rt->cond_mapped)) return 1;
    if (create_host_buffer(rt, ctrl_emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->cond_act_buffer, &rt->cond_act_memory, &rt->cond_act_mapped)) return 1;
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)out_norm_weight_bytes, weights->out_norm_fc_weight,
                &rt->out_norm_weight_buffer, &rt->out_norm_weight_memory)) return 1;
    if (create_host_buffer(rt, out_mod_table_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->out_mod_table_buffer, &rt->out_mod_table_memory, &rt->out_mod_table_mapped)) return 1;
    if (rt->layers_to_run > 0) {
        if (create_host_buffer(rt, layer_cond_bias_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->layer_cond_bias_buffer, &rt->layer_cond_bias_memory, &rt->layer_cond_bias_mapped)) return 1;
        if (create_device_buffer(rt, (VkDeviceSize)layer_cond_proj_weight_bytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    &rt->layer_cond_proj_weight_buffer, &rt->layer_cond_proj_weight_memory)) return 1;
        if (create_host_buffer(rt, layer_mod_table_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->layer_mod_table_buffer, &rt->layer_mod_table_memory, &rt->layer_mod_table_mapped)) return 1;
        rt->layer_bias_silu_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
        rt->layer_bias_silu_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
        rt->layer_mod_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->total_passes * rt->layers_to_run, sizeof(VkDescriptorPool));
        rt->layer_mod_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->total_passes * rt->layers_to_run, sizeof(VkDescriptorSet));
        if (!rt->layer_bias_silu_descriptor_pools || !rt->layer_bias_silu_descriptor_sets ||
                !rt->layer_mod_descriptor_pools || !rt->layer_mod_descriptor_sets) return 1;
        if (rt->layer_qkv_enabled) {
            if (create_device_buffer(rt, (VkDeviceSize)qkv_proj_weight_bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        &rt->qkv_proj_weight_buffer, &rt->qkv_proj_weight_memory)) return 1;
            if (rt->runtime_linear_wf16_coopmat_enabled) {
                if (create_device_buffer(rt, (VkDeviceSize)qkv_proj_weight_f16_bytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            &rt->qkv_proj_weight_f16_buffer,
                            &rt->qkv_proj_weight_f16_memory)) return 1;
            }
            rt->attn_ada_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->total_passes * rt->layers_to_run, sizeof(VkDescriptorPool));
            rt->attn_ada_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->total_passes * rt->layers_to_run, sizeof(VkDescriptorSet));
            rt->qkv_proj_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
            rt->qkv_proj_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
            if (rt->runtime_linear_wf16_coopmat_enabled) {
                rt->qkv_proj_wf16_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                rt->qkv_proj_wf16_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
            }
            if (!rt->attn_ada_descriptor_pools || !rt->attn_ada_descriptor_sets ||
                    !rt->qkv_proj_descriptor_pools || !rt->qkv_proj_descriptor_sets ||
                    (rt->runtime_linear_wf16_coopmat_enabled &&
                     (!rt->qkv_proj_wf16_descriptor_pools || !rt->qkv_proj_wf16_descriptor_sets))) return 1;
        }
    }
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)patch_weight_bytes, weights->patchify_weight,
                &rt->patch_weight_buffer, &rt->patch_weight_memory)) return 1;
    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                &rt->tokens_buffer, &rt->tokens_memory)) return 1;
    if (rt->layer_qkv_enabled) {
        if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                    &rt->norm_buffer, &rt->norm_memory)) return 1;
        if (create_device_storage_buffer(rt, (VkDeviceSize)qkv_raw_bytes,
                    &rt->qkv_raw_buffer, &rt->qkv_raw_memory)) return 1;
        if (create_device_storage_buffer(rt, (VkDeviceSize)q_rope_bytes,
                    &rt->q_buffer, &rt->q_memory)) return 1;
        if (create_device_storage_buffer(rt, (VkDeviceSize)kv_rope_bytes,
                    &rt->k_buffer, &rt->k_memory)) return 1;
        if (create_device_storage_buffer(rt, (VkDeviceSize)kv_rope_bytes,
                    &rt->v_buffer, &rt->v_memory)) return 1;
        if (rt->cfg.value_residual) {
            if (create_device_storage_buffer(rt, (VkDeviceSize)kv_rope_bytes,
                        &rt->v_first_buffer, &rt->v_first_memory)) return 1;
        }
        if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->x_pos_buffer, &rt->x_pos_memory, &rt->x_pos_mapped)) return 1;
        if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->y_pos_buffer, &rt->y_pos_memory, &rt->y_pos_mapped)) return 1;
        if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->t_pos_buffer, &rt->t_pos_memory, &rt->t_pos_mapped)) return 1;
        if (create_host_buffer(rt, xy_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->xy_buffer, &rt->xy_memory, &rt->xy_mapped)) return 1;
        if (create_host_buffer(rt, inv_t_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    &rt->inv_t_buffer, &rt->inv_t_memory, &rt->inv_t_mapped)) return 1;
        if (rt->layer_attention_enabled) {
            rt->kv_upsert_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
            rt->kv_upsert_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
            rt->cache_indices_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
            rt->cache_indices_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
            rt->indexed_attention_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
            rt->indexed_attention_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
            if (!rt->kv_upsert_descriptor_pools || !rt->kv_upsert_descriptor_sets ||
                    !rt->cache_indices_descriptor_pools || !rt->cache_indices_descriptor_sets ||
                    !rt->indexed_attention_descriptor_pools || !rt->indexed_attention_descriptor_sets) return 1;
            if (create_device_storage_buffer(rt, (VkDeviceSize)cache_kv_bytes,
                        &rt->cache_k_buffer, &rt->cache_k_memory)) return 1;
            if (create_device_storage_buffer(rt, (VkDeviceSize)cache_kv_bytes,
                        &rt->cache_v_buffer, &rt->cache_v_memory)) return 1;
            if (create_device_storage_buffer(rt, (VkDeviceSize)cache_meta_bytes,
                        &rt->cache_written_buffer, &rt->cache_written_memory)) return 1;
            if (create_device_storage_buffer(rt, (VkDeviceSize)cache_meta_bytes,
                        &rt->cache_indices_buffer, &rt->cache_indices_memory)) return 1;
            if (create_device_storage_buffer(rt, (VkDeviceSize)cache_count_bytes,
                        &rt->cache_index_count_buffer, &rt->cache_index_count_memory)) return 1;
            if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                        &rt->attn_buffer, &rt->attn_memory)) return 1;
            if (rt->layer_attn_out_enabled) {
                rt->attn_out_proj_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                rt->attn_out_proj_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                rt->attn_residual_descriptor_pools = (VkDescriptorPool *)calloc(pass_layer_count, sizeof(VkDescriptorPool));
                rt->attn_residual_descriptor_sets = (VkDescriptorSet *)calloc(pass_layer_count, sizeof(VkDescriptorSet));
                if (!rt->attn_out_proj_descriptor_pools || !rt->attn_out_proj_descriptor_sets ||
                        !rt->attn_residual_descriptor_pools || !rt->attn_residual_descriptor_sets) return 1;
                if (create_device_buffer(rt, (VkDeviceSize)attn_out_proj_weight_bytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            &rt->attn_out_proj_weight_buffer, &rt->attn_out_proj_weight_memory)) return 1;
                if (rt->runtime_linear_wf16_coopmat_enabled) {
                    rt->attn_out_proj_wf16_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->attn_out_proj_wf16_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    if (!rt->attn_out_proj_wf16_descriptor_pools ||
                            !rt->attn_out_proj_wf16_descriptor_sets) return 1;
                    if (create_device_buffer(rt, (VkDeviceSize)attn_out_proj_weight_f16_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &rt->attn_out_proj_weight_f16_buffer,
                                &rt->attn_out_proj_weight_f16_memory)) return 1;
                }
                if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                            &rt->attn_proj_buffer, &rt->attn_proj_memory)) return 1;
                if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                            &rt->tokens_after_attn_buffer, &rt->tokens_after_attn_memory)) return 1;
                if (rt->layer_ctrl_enabled) {
                    rt->ctrl_cond_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->ctrl_cond_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    rt->ctrl_fc1_x_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->ctrl_fc1_x_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    rt->ctrl_add_silu_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->ctrl_add_silu_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    rt->ctrl_fc2_descriptor_pools_layer = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->ctrl_fc2_descriptor_sets_layer = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    if (!rt->ctrl_cond_descriptor_pools || !rt->ctrl_cond_descriptor_sets ||
                            !rt->ctrl_fc1_x_descriptor_pools || !rt->ctrl_fc1_x_descriptor_sets ||
                            !rt->ctrl_add_silu_descriptor_pools || !rt->ctrl_add_silu_descriptor_sets ||
                            !rt->ctrl_fc2_descriptor_pools_layer || !rt->ctrl_fc2_descriptor_sets_layer) return 1;
                    if (create_device_buffer(rt, (VkDeviceSize)ctrl_layer_weight_table_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &rt->ctrl_fc1_c_weight_buffer, &rt->ctrl_fc1_c_weight_memory)) return 1;
                    if (create_device_buffer(rt, (VkDeviceSize)ctrl_layer_weight_table_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &rt->ctrl_fc1_x_weight_buffer, &rt->ctrl_fc1_x_weight_memory)) return 1;
                    if (create_device_buffer(rt, (VkDeviceSize)ctrl_layer_weight_table_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &rt->ctrl_fc2_weight_buffer_layer, &rt->ctrl_fc2_weight_memory_layer)) return 1;
                    if (create_host_buffer(rt, ctrl_cond_table_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                &rt->ctrl_cond_buffer, &rt->ctrl_cond_memory, &rt->ctrl_cond_mapped)) return 1;
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->ctrl_norm_buffer, &rt->ctrl_norm_memory)) return 1;
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->ctrl_hidden_layer_buffer, &rt->ctrl_hidden_layer_memory)) return 1;
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->ctrl_out_buffer, &rt->ctrl_out_memory)) return 1;
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->tokens_after_ctrl_buffer, &rt->tokens_after_ctrl_memory)) return 1;
                }
                if (rt->layer_mlp_enabled) {
                    rt->mlp_ada_descriptor_pools = (VkDescriptorPool *)calloc(pass_layer_count, sizeof(VkDescriptorPool));
                    rt->mlp_ada_descriptor_sets = (VkDescriptorSet *)calloc(pass_layer_count, sizeof(VkDescriptorSet));
                    rt->mlp_fc1_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->mlp_fc1_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    rt->mlp_fc2_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                    rt->mlp_fc2_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    rt->mlp_residual_descriptor_pools = (VkDescriptorPool *)calloc(pass_layer_count, sizeof(VkDescriptorPool));
                    rt->mlp_residual_descriptor_sets = (VkDescriptorSet *)calloc(pass_layer_count, sizeof(VkDescriptorSet));
                    if (rt->runtime_linear_wf16_coopmat_enabled) {
                        rt->mlp_fc1_wf16_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                        rt->mlp_fc1_wf16_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                        rt->mlp_fc2_wf16_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                        rt->mlp_fc2_wf16_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                        rt->mlp_fc2_f16x_descriptor_pools = (VkDescriptorPool *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorPool));
                        rt->mlp_fc2_f16x_descriptor_sets = (VkDescriptorSet *)calloc((size_t)rt->layers_to_run, sizeof(VkDescriptorSet));
                    }
                    if (!rt->mlp_ada_descriptor_pools || !rt->mlp_ada_descriptor_sets ||
                            !rt->mlp_fc1_descriptor_pools || !rt->mlp_fc1_descriptor_sets ||
                            !rt->mlp_fc2_descriptor_pools || !rt->mlp_fc2_descriptor_sets ||
                            !rt->mlp_residual_descriptor_pools || !rt->mlp_residual_descriptor_sets ||
                            (rt->runtime_linear_wf16_coopmat_enabled &&
                             (!rt->mlp_fc1_wf16_descriptor_pools || !rt->mlp_fc1_wf16_descriptor_sets ||
                              !rt->mlp_fc2_wf16_descriptor_pools || !rt->mlp_fc2_wf16_descriptor_sets ||
                              !rt->mlp_fc2_f16x_descriptor_pools || !rt->mlp_fc2_f16x_descriptor_sets))) return 1;
                    if (create_device_buffer(rt, (VkDeviceSize)dit_mlp_fc1_weight_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &rt->dit_mlp_fc1_weight_buffer, &rt->dit_mlp_fc1_weight_memory)) return 1;
                    if (create_device_buffer(rt, (VkDeviceSize)dit_mlp_fc2_weight_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &rt->dit_mlp_fc2_weight_buffer, &rt->dit_mlp_fc2_weight_memory)) return 1;
                    if (rt->runtime_linear_wf16_coopmat_enabled) {
                        if (create_device_buffer(rt, (VkDeviceSize)dit_mlp_fc1_weight_f16_bytes,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &rt->dit_mlp_fc1_weight_f16_buffer,
                                    &rt->dit_mlp_fc1_weight_f16_memory)) return 1;
                        if (create_device_buffer(rt, (VkDeviceSize)dit_mlp_fc2_weight_f16_bytes,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &rt->dit_mlp_fc2_weight_f16_buffer,
                                    &rt->dit_mlp_fc2_weight_f16_memory)) return 1;
                    }
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->mlp_in_buffer, &rt->mlp_in_memory)) return 1;
                    if (create_device_storage_buffer(rt, (VkDeviceSize)mlp_hidden_token_bytes,
                                &rt->mlp_hidden_buffer, &rt->mlp_hidden_memory)) return 1;
                    if (rt->runtime_linear_wf16_coopmat_enabled) {
                        if (create_device_storage_buffer(rt, (VkDeviceSize)mlp_hidden_token_f16_bytes,
                                    &rt->mlp_hidden_f16_buffer, &rt->mlp_hidden_f16_memory)) return 1;
                    }
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->mlp_out_buffer, &rt->mlp_out_memory)) return 1;
                    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                                &rt->tokens_after_mlp_buffer, &rt->tokens_after_mlp_memory)) return 1;
                }
            }
        }
    }
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)patch_weight_bytes, weights->unpatchify_weight,
                &rt->unpatch_weight_buffer, &rt->unpatch_weight_memory)) return 1;
    if (create_weight_buffer_from_data(rt, (VkDeviceSize)unpatch_bias_bytes, weights->unpatchify_bias,
                &rt->unpatch_bias_buffer, &rt->unpatch_bias_memory)) return 1;
    if (create_device_storage_buffer(rt, (VkDeviceSize)token_bytes,
                &rt->final_tokens_buffer, &rt->final_tokens_memory)) return 1;
    if (create_device_storage_buffer(rt, (VkDeviceSize)latent_bytes,
                &rt->latent_out_buffer, &rt->latent_out_memory)) return 1;

    memset(rt->latent_mapped, 0, latent_bytes);
    memset(rt->control_mapped, 0, control_bytes);
    memset(rt->ctrl_hidden_mapped, 0, ctrl_hidden_bytes);
    memset(rt->ctrl_emb_mapped, 0, ctrl_emb_bytes);
    memset(rt->ctrl_emb_norm_mapped, 0, ctrl_emb_bytes);
    ((float *)rt->dummy_bias_mapped)[0] = 0.0f;
    for (int i = 0; i < rt->D; ++i) {
        ((float *)rt->rms_weight_mapped)[i] = 1.0f;
    }
    memset(rt->noise_mapped, 0, noise_bytes);
    memset(rt->noise_hidden_mapped, 0, ctrl_hidden_bytes);
    memset(rt->cond_mapped, 0, ctrl_emb_bytes);
    memset(rt->cond_act_mapped, 0, ctrl_emb_bytes);
    memset(rt->out_mod_table_mapped, 0, out_mod_table_bytes);
    if (rt->layers_to_run > 0) {
        float *bias_dst = (float *)rt->layer_cond_bias_mapped;
        size_t block = (size_t)rt->D * rt->D;
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        void *staging_mapped = NULL;
        if (create_host_buffer(rt, (VkDeviceSize)layer_cond_proj_weight_bytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    &staging_buffer, &staging_memory, &staging_mapped)) return 1;
        float *proj_dst = (float *)staging_mapped;
        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            const WorldLayerWeights *lw = &weights->layers[layer_idx];
            memcpy(bias_dst + (size_t)layer_idx * rt->D, lw->cond_bias, (size_t)rt->D * sizeof(float));
            float *dst = proj_dst + (size_t)layer_idx * 6 * block;
            memcpy(dst + 0 * block, lw->attn_cond_s_weight, block * sizeof(float));
            memcpy(dst + 1 * block, lw->attn_cond_b_weight, block * sizeof(float));
            memcpy(dst + 2 * block, lw->attn_cond_g_weight, block * sizeof(float));
            memcpy(dst + 3 * block, lw->mlp_cond_s_weight, block * sizeof(float));
            memcpy(dst + 4 * block, lw->mlp_cond_b_weight, block * sizeof(float));
            memcpy(dst + 5 * block, lw->mlp_cond_g_weight, block * sizeof(float));
        }
        int copy_failed = copy_buffer_to_device(rt, staging_buffer, rt->layer_cond_proj_weight_buffer,
                (VkDeviceSize)layer_cond_proj_weight_bytes);
        destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
        if (copy_failed) return 1;
        memset(rt->layer_mod_table_mapped, 0, layer_mod_table_bytes);
    }
    if (clear_device_buffer(rt, rt->tokens_buffer, (VkDeviceSize)token_bytes)) return 1;
    if (rt->layer_qkv_enabled) {
        size_t q_elems = (size_t)rt->D * rt->D;
        size_t kv_elems = (size_t)rt->kv_dim * rt->D;
        size_t layer_elems = q_elems + 2 * kv_elems;
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        void *staging_mapped = NULL;
        if (create_host_buffer(rt, (VkDeviceSize)qkv_proj_weight_bytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    &staging_buffer, &staging_memory, &staging_mapped)) return 1;
        float *qkv_dst = (float *)staging_mapped;
        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            const WorldLayerWeights *lw = &weights->layers[layer_idx];
            float *dst = qkv_dst + (size_t)layer_idx * layer_elems;
            memcpy(dst, lw->q_proj_weight, q_elems * sizeof(float));
            memcpy(dst + q_elems, lw->k_proj_weight, kv_elems * sizeof(float));
            memcpy(dst + q_elems + kv_elems, lw->v_proj_weight, kv_elems * sizeof(float));
        }
        int copy_failed = copy_buffer_to_device(rt, staging_buffer, rt->qkv_proj_weight_buffer,
                (VkDeviceSize)qkv_proj_weight_bytes);
        destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
        if (copy_failed) return 1;
        if (rt->runtime_linear_wf16_coopmat_enabled && rt->qkv_proj_weight_f16_buffer) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                const WorldLayerWeights *lw = &weights->layers[layer_idx];
                VkDeviceSize layer_base = (VkDeviceSize)((size_t)layer_idx *
                        qkv_proj_weight_layer_f16_bytes);
                if (upload_f32_as_f16_to_device_buffer(rt, rt->qkv_proj_weight_f16_buffer,
                            layer_base, lw->q_proj_weight, q_elems)) return 1;
                if (upload_f32_as_f16_to_device_buffer(rt, rt->qkv_proj_weight_f16_buffer,
                            layer_base + (VkDeviceSize)(q_elems * sizeof(uint16_t)),
                            lw->k_proj_weight, kv_elems)) return 1;
                if (upload_f32_as_f16_to_device_buffer(rt, rt->qkv_proj_weight_f16_buffer,
                            layer_base + (VkDeviceSize)((q_elems + kv_elems) * sizeof(uint16_t)),
                            lw->v_proj_weight, kv_elems)) return 1;
            }
        }
        if (clear_device_buffer(rt, rt->norm_buffer, (VkDeviceSize)token_bytes)) return 1;
        if (clear_device_buffer(rt, rt->qkv_raw_buffer, (VkDeviceSize)qkv_raw_bytes)) return 1;
        if (clear_device_buffer(rt, rt->q_buffer, (VkDeviceSize)q_rope_bytes)) return 1;
        if (clear_device_buffer(rt, rt->k_buffer, (VkDeviceSize)kv_rope_bytes)) return 1;
        if (clear_device_buffer(rt, rt->v_buffer, (VkDeviceSize)kv_rope_bytes)) return 1;
        if (rt->v_first_buffer &&
                clear_device_buffer(rt, rt->v_first_buffer, (VkDeviceSize)kv_rope_bytes)) return 1;
        memset(rt->x_pos_mapped, 0, pos_bytes);
        memset(rt->y_pos_mapped, 0, pos_bytes);
        memset(rt->t_pos_mapped, 0, pos_bytes);
        fill_runtime_rope_tables(rt);
        if (rt->layer_attention_enabled) {
            if (clear_device_buffer(rt, rt->cache_k_buffer, (VkDeviceSize)cache_kv_bytes)) return 1;
            if (clear_device_buffer(rt, rt->cache_v_buffer, (VkDeviceSize)cache_kv_bytes)) return 1;
            if (clear_device_buffer(rt, rt->cache_indices_buffer, (VkDeviceSize)cache_meta_bytes)) return 1;
            if (clear_device_buffer(rt, rt->cache_index_count_buffer, (VkDeviceSize)cache_count_bytes)) return 1;
            if (clear_device_buffer(rt, rt->attn_buffer, (VkDeviceSize)token_bytes)) return 1;
            {
                uint32_t *written = (uint32_t *)calloc(rt->cache_total_meta_elems, sizeof(uint32_t));
                if (!written) return 1;
                for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                    size_t base = rt->cache_meta_offsets[layer_idx];
                    for (int i = rt->cache_ring_lengths[layer_idx]; i < rt->cache_capacities[layer_idx]; ++i) {
                        written[base + (size_t)i] = 1u;
                    }
                }
                int upload_failed = upload_to_device_buffer(rt, rt->cache_written_buffer,
                        written, (VkDeviceSize)cache_meta_bytes);
                free(written);
                if (upload_failed) return 1;
            }
            if (rt->layer_attn_out_enabled) {
                {
                    VkBuffer staging_buffer = VK_NULL_HANDLE;
                    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
                    void *staging_mapped = NULL;
                    if (create_host_buffer(rt, (VkDeviceSize)attn_out_proj_weight_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
                    float *dst = (float *)staging_mapped;
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        memcpy(dst + (size_t)layer_idx * rt->D * rt->D,
                                weights->layers[layer_idx].out_proj_weight,
                                attn_out_proj_weight_layer_bytes);
                    }
                    int copy_failed = copy_buffer_to_device(rt, staging_buffer,
                            rt->attn_out_proj_weight_buffer,
                            (VkDeviceSize)attn_out_proj_weight_bytes);
                    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
                    if (copy_failed) return 1;
                }
                if (rt->runtime_linear_wf16_coopmat_enabled &&
                        rt->attn_out_proj_weight_f16_buffer) {
                    size_t out_layer_elems = (size_t)rt->D * rt->D;
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        VkDeviceSize dst_offset = (VkDeviceSize)((size_t)layer_idx *
                                attn_out_proj_weight_layer_f16_bytes);
                        if (upload_f32_as_f16_to_device_buffer(rt,
                                    rt->attn_out_proj_weight_f16_buffer,
                                    dst_offset,
                                    weights->layers[layer_idx].out_proj_weight,
                                    out_layer_elems)) return 1;
                    }
                }
                if (clear_device_buffer(rt, rt->attn_proj_buffer, (VkDeviceSize)token_bytes)) return 1;
                if (clear_device_buffer(rt, rt->tokens_after_attn_buffer, (VkDeviceSize)token_bytes)) return 1;
                if (rt->layer_ctrl_enabled) {
                    VkBuffer staging_buffer = VK_NULL_HANDLE;
                    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
                    void *staging_mapped = NULL;
                    if (create_host_buffer(rt, (VkDeviceSize)ctrl_layer_weight_table_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
                    memset(staging_mapped, 0, ctrl_layer_weight_table_bytes);
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        if (!rt->layer_has_ctrl[layer_idx]) continue;
                        size_t offset = (size_t)layer_idx * rt->D * rt->D;
                        memcpy((float *)staging_mapped + offset,
                                weights->layers[layer_idx].ctrl_fc1_c_weight,
                                ctrl_layer_weight_bytes);
                    }
                    int copy_failed = copy_buffer_to_device(rt, staging_buffer,
                            rt->ctrl_fc1_c_weight_buffer,
                            (VkDeviceSize)ctrl_layer_weight_table_bytes);
                    if (!copy_failed) {
                        memset(staging_mapped, 0, ctrl_layer_weight_table_bytes);
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            if (!rt->layer_has_ctrl[layer_idx]) continue;
                            size_t offset = (size_t)layer_idx * rt->D * rt->D;
                            memcpy((float *)staging_mapped + offset,
                                    weights->layers[layer_idx].ctrl_fc1_x_weight,
                                    ctrl_layer_weight_bytes);
                        }
                        copy_failed = copy_buffer_to_device(rt, staging_buffer,
                                rt->ctrl_fc1_x_weight_buffer,
                                (VkDeviceSize)ctrl_layer_weight_table_bytes);
                    }
                    if (!copy_failed) {
                        memset(staging_mapped, 0, ctrl_layer_weight_table_bytes);
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            if (!rt->layer_has_ctrl[layer_idx]) continue;
                            size_t offset = (size_t)layer_idx * rt->D * rt->D;
                            memcpy((float *)staging_mapped + offset,
                                    weights->layers[layer_idx].ctrl_fc2_weight,
                                    ctrl_layer_weight_bytes);
                        }
                        copy_failed = copy_buffer_to_device(rt, staging_buffer,
                                rt->ctrl_fc2_weight_buffer_layer,
                                (VkDeviceSize)ctrl_layer_weight_table_bytes);
                    }
                    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
                    if (copy_failed) return 1;
                    memset(rt->ctrl_cond_mapped, 0, ctrl_cond_table_bytes);
                    if (clear_device_buffer(rt, rt->ctrl_norm_buffer, (VkDeviceSize)token_bytes)) return 1;
                    if (clear_device_buffer(rt, rt->ctrl_hidden_layer_buffer, (VkDeviceSize)token_bytes)) return 1;
                    if (clear_device_buffer(rt, rt->ctrl_out_buffer, (VkDeviceSize)token_bytes)) return 1;
                    if (clear_device_buffer(rt, rt->tokens_after_ctrl_buffer, (VkDeviceSize)token_bytes)) return 1;
                }
                if (rt->layer_mlp_enabled) {
                    VkBuffer staging_buffer = VK_NULL_HANDLE;
                    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
                    void *staging_mapped = NULL;
                    if (create_host_buffer(rt, (VkDeviceSize)dit_mlp_fc1_weight_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        memcpy((float *)staging_mapped +
                                    (size_t)layer_idx * rt->mlp_hidden * rt->D,
                                weights->layers[layer_idx].dit_mlp_fc1_weight,
                                dit_mlp_fc1_weight_layer_bytes);
                    }
                    int copy_failed = copy_buffer_to_device(rt, staging_buffer,
                            rt->dit_mlp_fc1_weight_buffer,
                            (VkDeviceSize)dit_mlp_fc1_weight_bytes);
                    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
                    if (copy_failed) return 1;

                    staging_buffer = VK_NULL_HANDLE;
                    staging_memory = VK_NULL_HANDLE;
                    staging_mapped = NULL;
                    if (create_host_buffer(rt, (VkDeviceSize)dit_mlp_fc2_weight_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                &staging_buffer, &staging_memory, &staging_mapped)) return 1;
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        memcpy((float *)staging_mapped +
                                    (size_t)layer_idx * rt->D * rt->mlp_hidden,
                                weights->layers[layer_idx].dit_mlp_fc2_weight,
                                dit_mlp_fc2_weight_layer_bytes);
                    }
                    copy_failed = copy_buffer_to_device(rt, staging_buffer,
                            rt->dit_mlp_fc2_weight_buffer,
                            (VkDeviceSize)dit_mlp_fc2_weight_bytes);
                    destroy_host_buffer(rt, staging_buffer, staging_memory, staging_mapped);
                    if (copy_failed) return 1;
                    if (rt->runtime_linear_wf16_coopmat_enabled &&
                            rt->dit_mlp_fc1_weight_f16_buffer &&
                            rt->dit_mlp_fc2_weight_f16_buffer) {
                        size_t fc1_layer_elems = (size_t)rt->mlp_hidden * rt->D;
                        size_t fc2_layer_elems = (size_t)rt->D * rt->mlp_hidden;
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            VkDeviceSize dst_offset = (VkDeviceSize)((size_t)layer_idx *
                                    dit_mlp_fc1_weight_layer_f16_bytes);
                            if (upload_f32_as_f16_to_device_buffer(rt,
                                        rt->dit_mlp_fc1_weight_f16_buffer,
                                        dst_offset,
                                        weights->layers[layer_idx].dit_mlp_fc1_weight,
                                        fc1_layer_elems)) return 1;
                        }
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            VkDeviceSize dst_offset = (VkDeviceSize)((size_t)layer_idx *
                                    dit_mlp_fc2_weight_layer_f16_bytes);
                            if (upload_f32_as_f16_to_device_buffer(rt,
                                        rt->dit_mlp_fc2_weight_f16_buffer,
                                        dst_offset,
                                        weights->layers[layer_idx].dit_mlp_fc2_weight,
                                        fc2_layer_elems)) return 1;
                        }
                    }
                    if (clear_device_buffer(rt, rt->mlp_in_buffer, (VkDeviceSize)token_bytes)) return 1;
                    if (clear_device_buffer(rt, rt->mlp_hidden_buffer, (VkDeviceSize)mlp_hidden_token_bytes)) return 1;
                    if (rt->mlp_hidden_f16_buffer &&
                            clear_device_buffer(rt, rt->mlp_hidden_f16_buffer,
                                (VkDeviceSize)mlp_hidden_token_f16_bytes)) return 1;
                    if (clear_device_buffer(rt, rt->mlp_out_buffer, (VkDeviceSize)token_bytes)) return 1;
                    if (clear_device_buffer(rt, rt->tokens_after_mlp_buffer, (VkDeviceSize)token_bytes)) return 1;
                }
            }
        }
    }
    if (clear_device_buffer(rt, rt->final_tokens_buffer, (VkDeviceSize)token_bytes)) return 1;
    if (clear_device_buffer(rt, rt->latent_out_buffer, (VkDeviceSize)latent_bytes)) return 1;

    if (create_storage_pipeline(rt, "linear_f32.comp", 4, sizeof(WorldVulkanLinearPush),
                &rt->runtime_linear_shader, &rt->runtime_linear_set_layout,
                &rt->runtime_linear_pipeline_layout, &rt->runtime_linear_pipeline)) return 1;
    {
        const char *coopmat_env = getenv("WORLD_VULKAN_LINEAR_COOPMAT");
        if (coopmat_env && coopmat_env[0] == '1') {
            if (rt->shader_float16_enabled && rt->cooperative_matrix_enabled) {
                if (create_storage_pipeline_with_set_layout(rt, "linear_f32_coopmat.comp",
                            rt->runtime_linear_set_layout, sizeof(WorldVulkanLinearPush),
                            &rt->runtime_linear_coopmat_shader,
                            &rt->runtime_linear_coopmat_pipeline_layout,
                            &rt->runtime_linear_coopmat_pipeline)) return 1;
                rt->runtime_linear_coopmat_enabled = 1;
                fprintf(stderr, "Vulkan linear f32->f16 cooperative-matrix path enabled (WORLD_VULKAN_LINEAR_COOPMAT=1)\n");
            } else {
                fprintf(stderr, "warning: WORLD_VULKAN_LINEAR_COOPMAT=1 ignored; shaderFloat16=%d cooperativeMatrix=%d\n",
                        rt->shader_float16_enabled, rt->cooperative_matrix_enabled);
            }
        }
    }
    if (rt->runtime_linear_wf16_coopmat_enabled) {
        if (create_storage_pipeline_with_set_layout(rt, "linear_f32_wf16_coopmat.comp",
                    rt->runtime_linear_set_layout, sizeof(WorldVulkanLinearPush),
                    &rt->runtime_linear_wf16_coopmat_shader,
                    &rt->runtime_linear_wf16_coopmat_pipeline_layout,
                    &rt->runtime_linear_wf16_coopmat_pipeline)) return 1;
        if (create_storage_pipeline_with_set_layout(rt, "linear_f32_wf16_coopmat_n16.comp",
                    rt->runtime_linear_set_layout, sizeof(WorldVulkanLinearPush),
                    &rt->runtime_linear_wf16_coopmat_n16_shader,
                    &rt->runtime_linear_wf16_coopmat_n16_pipeline_layout,
                    &rt->runtime_linear_wf16_coopmat_n16_pipeline)) return 1;
        if (create_storage_pipeline_with_set_layout(rt, "linear_f16x_wf16_coopmat.comp",
                    rt->runtime_linear_set_layout, sizeof(WorldVulkanLinearPush),
                    &rt->runtime_linear_f16x_wf16_coopmat_shader,
                    &rt->runtime_linear_f16x_wf16_coopmat_pipeline_layout,
                    &rt->runtime_linear_f16x_wf16_coopmat_pipeline)) return 1;
        fprintf(stderr, "Vulkan linear f32 input + f16 weight cooperative-matrix path enabled for qkv/attn_out/MLP (WORLD_VULKAN_LINEAR_WF16_COOPMAT=1)\n");
    }
    {
        const char *tile64_env = getenv("WORLD_VULKAN_LINEAR_TILE64");
        if (tile64_env && tile64_env[0] == '1') {
            if (create_storage_pipeline_with_set_layout(rt, "linear_f32_tile64.comp",
                        rt->runtime_linear_set_layout, sizeof(WorldVulkanLinearPush),
                        &rt->runtime_linear_tile64_shader,
                        &rt->runtime_linear_tile64_pipeline_layout,
                        &rt->runtime_linear_tile64_pipeline)) return 1;
            rt->runtime_linear_tile64_enabled = 1;
            fprintf(stderr, "Vulkan linear tile64 enabled for large GEMMs (WORLD_VULKAN_LINEAR_TILE64=1)\n");
        }
    }
    if (create_storage_pipeline(rt, "silu_f32.comp", 2, sizeof(WorldVulkanSiluPush),
                &rt->runtime_silu_shader, &rt->runtime_silu_set_layout,
                &rt->runtime_silu_pipeline_layout, &rt->runtime_silu_pipeline)) return 1;
    if (rt->runtime_linear_wf16_coopmat_enabled) {
        if (create_storage_pipeline(rt, "silu_f32_to_f16.comp", 2, sizeof(WorldVulkanSiluPush),
                    &rt->runtime_silu_f16_shader, &rt->runtime_silu_f16_set_layout,
                    &rt->runtime_silu_f16_pipeline_layout, &rt->runtime_silu_f16_pipeline)) return 1;
    }
    if (rt->layers_to_run > 0) {
        if (create_storage_pipeline(rt, "add_bias_silu_f32.comp", 3, sizeof(WorldVulkanSiluPush),
                    &rt->runtime_add_bias_silu_shader, &rt->runtime_add_bias_silu_set_layout,
                    &rt->runtime_add_bias_silu_pipeline_layout, &rt->runtime_add_bias_silu_pipeline)) return 1;
    }
    if (create_storage_pipeline(rt, "rms_norm_f32.comp", 3, sizeof(WorldVulkanRmsNormPush),
                &rt->runtime_rms_shader, &rt->runtime_rms_set_layout,
                &rt->runtime_rms_pipeline_layout, &rt->runtime_rms_pipeline)) return 1;
    if (rt->layer_qkv_enabled) {
        if (create_storage_pipeline(rt, "ada_rms_norm_f32.comp", 4, sizeof(WorldVulkanAdaRmsNormPush),
                    &rt->runtime_ada_rms_shader, &rt->runtime_ada_rms_set_layout,
                    &rt->runtime_ada_rms_pipeline_layout, &rt->runtime_ada_rms_pipeline)) return 1;
        if (create_storage_pipeline(rt, "qkv_rms_rope_f32.comp", 9, sizeof(WorldVulkanQkvRmsRopePush),
                    &rt->runtime_qkv_rms_rope_shader, &rt->runtime_qkv_rms_rope_set_layout,
                    &rt->runtime_qkv_rms_rope_pipeline_layout, &rt->runtime_qkv_rms_rope_pipeline)) return 1;
        if (rt->layer_attention_enabled) {
            if (create_storage_pipeline(rt, "kv_cache_upsert_copy_f32.comp", 5, sizeof(WorldVulkanKvCacheUpsertPush),
                        &rt->runtime_kv_upsert_shader, &rt->runtime_kv_upsert_set_layout,
                        &rt->runtime_kv_upsert_pipeline_layout, &rt->runtime_kv_upsert_pipeline)) return 1;
            if (create_storage_pipeline(rt, "cache_frame_indices.comp", 3, sizeof(WorldVulkanCacheFrameIndicesPush),
                        &rt->runtime_cache_indices_shader, &rt->runtime_cache_indices_set_layout,
                        &rt->runtime_cache_indices_pipeline_layout, &rt->runtime_cache_indices_pipeline)) return 1;
            if (create_storage_pipeline(rt, "indexed_attention_f32.comp", 6, sizeof(WorldVulkanIndexedAttentionPush),
                        &rt->runtime_indexed_attention_shader, &rt->runtime_indexed_attention_set_layout,
                        &rt->runtime_indexed_attention_pipeline_layout, &rt->runtime_indexed_attention_pipeline)) return 1;
            if (rt->d_head == 64) {
                if (create_storage_pipeline_with_set_layout(rt, "indexed_attention_d64_warp_f32.comp",
                            rt->runtime_indexed_attention_set_layout, sizeof(WorldVulkanIndexedAttentionPush),
                            &rt->runtime_indexed_attention_d64_shader,
                            &rt->runtime_indexed_attention_d64_pipeline_layout,
                            &rt->runtime_indexed_attention_d64_pipeline)) return 1;
                {
                    const char *flash96_env = getenv("WORLD_VULKAN_FLASH_ATTN96");
                    if (flash96_env && flash96_env[0] == '1') {
                        if (rt->max_compute_shared_memory_size >= 96u * 64u * 2u * (uint32_t)sizeof(float)) {
                            if (create_storage_pipeline_with_set_layout(rt, "indexed_attention_d64_flash96_f32.comp",
                                        rt->runtime_indexed_attention_set_layout, sizeof(WorldVulkanIndexedAttentionPush),
                                        &rt->runtime_indexed_attention_d64_flash96_shader,
                                        &rt->runtime_indexed_attention_d64_flash96_pipeline_layout,
                                        &rt->runtime_indexed_attention_d64_flash96_pipeline)) return 1;
                            rt->runtime_indexed_attention_d64_flash96_enabled = 1;
                            fprintf(stderr, "Vulkan d64 flash96 attention enabled (WORLD_VULKAN_FLASH_ATTN96=1)\n");
                        } else {
                            fprintf(stderr,
                                    "warning: WORLD_VULKAN_FLASH_ATTN96=1 ignored; maxComputeSharedMemorySize=%u need=%u\n",
                                    rt->max_compute_shared_memory_size,
                                    96u * 64u * 2u * (uint32_t)sizeof(float));
                        }
                    }
                }
                const char *flash_env = getenv("WORLD_VULKAN_FLASH_ATTN");
                if (!flash_env || flash_env[0] != '0') {
                    if (create_storage_pipeline_with_set_layout(rt, "indexed_attention_d64_flash_f32.comp",
                                rt->runtime_indexed_attention_set_layout, sizeof(WorldVulkanIndexedAttentionPush),
                                &rt->runtime_indexed_attention_d64_flash_shader,
                                &rt->runtime_indexed_attention_d64_flash_pipeline_layout,
                                &rt->runtime_indexed_attention_d64_flash_pipeline)) return 1;
                    rt->runtime_indexed_attention_d64_flash_enabled = 1;
                    fprintf(stderr, "Vulkan d64 flash-style attention enabled; set WORLD_VULKAN_FLASH_ATTN=0 to disable\n");
                }
            }
            if (rt->layer_attn_out_enabled) {
                if (create_storage_pipeline(rt, "gated_residual_add_f32.comp", 4, sizeof(WorldVulkanGatedResidualPush),
                            &rt->runtime_gated_residual_shader, &rt->runtime_gated_residual_set_layout,
                            &rt->runtime_gated_residual_pipeline_layout, &rt->runtime_gated_residual_pipeline)) return 1;
                if (rt->layer_ctrl_enabled) {
                    if (create_storage_pipeline(rt, "add_channel_silu_f32.comp", 3, sizeof(WorldVulkanAddChannelSiluPush),
                                &rt->runtime_add_channel_silu_shader, &rt->runtime_add_channel_silu_set_layout,
                                &rt->runtime_add_channel_silu_pipeline_layout, &rt->runtime_add_channel_silu_pipeline)) return 1;
                    if (create_storage_pipeline(rt, "add_f32.comp", 3, sizeof(WorldVulkanSiluPush),
                                &rt->runtime_add_shader, &rt->runtime_add_set_layout,
                                &rt->runtime_add_pipeline_layout, &rt->runtime_add_pipeline)) return 1;
                }
            }
        }
    }
    if (create_storage_pipeline(rt, "out_norm_silu_f32.comp", 3, sizeof(WorldVulkanOutNormSiluPush),
                &rt->runtime_out_norm_silu_shader, &rt->runtime_out_norm_silu_set_layout,
                &rt->runtime_out_norm_silu_pipeline_layout, &rt->runtime_out_norm_silu_pipeline)) return 1;
    if (create_storage_pipeline(rt, "latent_update_f32.comp", 2, sizeof(WorldVulkanLatentUpdatePush),
                &rt->runtime_latent_update_shader, &rt->runtime_latent_update_set_layout,
                &rt->runtime_latent_update_pipeline_layout, &rt->runtime_latent_update_pipeline)) return 1;
    if (rt->cfg.value_residual && rt->layer_qkv_enabled) {
        if (create_storage_pipeline(rt, "lerp_inplace_f32.comp", 2, sizeof(WorldVulkanLerpPush),
                    &rt->runtime_lerp_shader, &rt->runtime_lerp_set_layout,
                    &rt->runtime_lerp_pipeline_layout, &rt->runtime_lerp_pipeline)) return 1;
    }
    if (create_storage_pipeline(rt, "patchify_f32.comp", 3, sizeof(WorldVulkanPatchifyPush),
                &rt->patchify_shader, &rt->patchify_set_layout,
                &rt->patchify_pipeline_layout, &rt->patchify_pipeline)) return 1;
    if (create_storage_pipeline(rt, "unpatchify_orig_f32.comp", 4, sizeof(WorldVulkanUnpatchifyOrigPush),
                &rt->unpatch_orig_shader, &rt->unpatch_orig_set_layout,
                &rt->unpatch_orig_pipeline_layout, &rt->unpatch_orig_pipeline)) return 1;
    if (create_storage_pipeline(rt, "latent_to_rgba.comp", 3, sizeof(WorldVulkanLatentRgbaPush),
                &rt->latent_rgba_shader, &rt->latent_rgba_set_layout,
                &rt->latent_rgba_pipeline_layout, &rt->latent_rgba_pipeline)) return 1;

    {
        VkBuffer buffers[4] = {
            rt->control_buffer, rt->ctrl_fc1_weight_buffer, rt->dummy_bias_buffer, rt->ctrl_hidden_buffer
        };
        VkDeviceSize sizes[4] = {control_bytes, ctrl_fc1_weight_bytes, sizeof(float), ctrl_hidden_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, NULL,
                    &rt->ctrl_fc1_descriptor_pool, &rt->ctrl_fc1_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[2] = {rt->ctrl_hidden_buffer, rt->ctrl_hidden_buffer};
        VkDeviceSize sizes[2] = {ctrl_hidden_bytes, ctrl_hidden_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_silu_set_layout, 2, buffers, sizes, NULL,
                    &rt->ctrl_silu_descriptor_pool, &rt->ctrl_silu_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[4] = {
            rt->ctrl_hidden_buffer, rt->ctrl_fc2_weight_buffer, rt->dummy_bias_buffer, rt->ctrl_emb_buffer
        };
        VkDeviceSize sizes[4] = {ctrl_hidden_bytes, ctrl_fc2_weight_bytes, sizeof(float), ctrl_emb_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, NULL,
                    &rt->ctrl_fc2_descriptor_pool, &rt->ctrl_fc2_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[3] = {rt->ctrl_emb_buffer, rt->rms_weight_buffer, rt->ctrl_emb_norm_buffer};
        VkDeviceSize sizes[3] = {ctrl_emb_bytes, ctrl_emb_bytes, ctrl_emb_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_rms_set_layout, 3, buffers, sizes, NULL,
                    &rt->ctrl_rms_descriptor_pool, &rt->ctrl_rms_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[4] = {
            rt->noise_buffer, rt->denoise_fc1_weight_buffer, rt->dummy_bias_buffer, rt->noise_hidden_buffer
        };
        VkDeviceSize sizes[4] = {noise_bytes, denoise_fc1_weight_bytes, sizeof(float), ctrl_hidden_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, NULL,
                    &rt->denoise_fc1_descriptor_pool, &rt->denoise_fc1_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[2] = {rt->noise_hidden_buffer, rt->noise_hidden_buffer};
        VkDeviceSize sizes[2] = {ctrl_hidden_bytes, ctrl_hidden_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_silu_set_layout, 2, buffers, sizes, NULL,
                    &rt->denoise_silu_descriptor_pool, &rt->denoise_silu_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[4] = {
            rt->noise_hidden_buffer, rt->denoise_fc2_weight_buffer, rt->dummy_bias_buffer, rt->cond_buffer
        };
        VkDeviceSize sizes[4] = {ctrl_hidden_bytes, denoise_fc2_weight_bytes, sizeof(float), ctrl_emb_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, NULL,
                    &rt->denoise_fc2_descriptor_pool, &rt->denoise_fc2_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[2] = {rt->cond_buffer, rt->cond_act_buffer};
        VkDeviceSize sizes[2] = {ctrl_emb_bytes, ctrl_emb_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_silu_set_layout, 2, buffers, sizes, NULL,
                    &rt->denoise_cond_silu_descriptor_pool, &rt->denoise_cond_silu_descriptor_set)) return 1;
    }
    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        VkBuffer buffers[4] = {
            rt->cond_act_buffer, rt->out_norm_weight_buffer, rt->dummy_bias_buffer, rt->out_mod_table_buffer
        };
        VkDeviceSize sizes[4] = {ctrl_emb_bytes, out_norm_weight_bytes, sizeof(float), out_mod_pass_bytes};
        VkDeviceSize offsets[4] = {0, 0, 0, (VkDeviceSize)((size_t)pass_idx * out_mod_pass_bytes)};
        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                    &rt->out_norm_descriptor_pool[pass_idx], &rt->out_norm_descriptor_set[pass_idx])) return 1;
    }
    if (rt->layers_to_run > 0) {
        size_t layer_bias_bytes = (size_t)rt->D * sizeof(float);
        size_t layer_weight_bytes = (size_t)6 * rt->D * rt->D * sizeof(float);
        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            VkBuffer buffers[3] = {rt->cond_buffer, rt->layer_cond_bias_buffer, rt->cond_act_buffer};
            VkDeviceSize sizes[3] = {ctrl_emb_bytes, layer_bias_bytes, ctrl_emb_bytes};
            VkDeviceSize offsets[3] = {0, (VkDeviceSize)((size_t)layer_idx * layer_bias_bytes), 0};
            if (create_storage_descriptor_set(rt, rt->runtime_add_bias_silu_set_layout, 3, buffers, sizes, offsets,
                        &rt->layer_bias_silu_descriptor_pools[layer_idx],
                        &rt->layer_bias_silu_descriptor_sets[layer_idx])) return 1;
        }
        for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                int table_idx = pass_idx * rt->layers_to_run + layer_idx;
                VkBuffer buffers[4] = {
                    rt->cond_act_buffer, rt->layer_cond_proj_weight_buffer,
                    rt->dummy_bias_buffer, rt->layer_mod_table_buffer
                };
                VkDeviceSize sizes[4] = {ctrl_emb_bytes, layer_weight_bytes, sizeof(float), layer_mod_pass_layer_bytes};
                VkDeviceSize offsets[4] = {
                    0,
                    (VkDeviceSize)((size_t)layer_idx * layer_weight_bytes),
                    0,
                    (VkDeviceSize)((size_t)table_idx * layer_mod_pass_layer_bytes)
                };
                if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                            &rt->layer_mod_descriptor_pools[table_idx],
                            &rt->layer_mod_descriptor_sets[table_idx])) return 1;
            }
        }
    }
    if (rt->layer_qkv_enabled) {
        for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                int table_idx = pass_idx * rt->layers_to_run + layer_idx;
                VkBuffer tokens_in = (rt->layer_mlp_enabled && (layer_idx % 2) != 0) ?
                    rt->tokens_after_mlp_buffer : rt->tokens_buffer;
                VkBuffer buffers[4] = {
                    tokens_in, rt->layer_mod_table_buffer, rt->layer_mod_table_buffer, rt->norm_buffer
                };
                VkDeviceSize sizes[4] = {
                    token_bytes, ctrl_emb_bytes, ctrl_emb_bytes, token_bytes
                };
                VkDeviceSize base = (VkDeviceSize)((size_t)table_idx * layer_mod_pass_layer_bytes);
                VkDeviceSize offsets[4] = {
                    0,
                    base,
                    base + (VkDeviceSize)ctrl_emb_bytes,
                    0
                };
                if (create_storage_descriptor_set(rt, rt->runtime_ada_rms_set_layout, 4, buffers, sizes, offsets,
                            &rt->attn_ada_descriptor_pools[table_idx],
                            &rt->attn_ada_descriptor_sets[table_idx])) return 1;
            }
        }
        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
            VkBuffer buffers[4] = {
                rt->norm_buffer, rt->qkv_proj_weight_buffer, rt->dummy_bias_buffer, rt->qkv_raw_buffer
            };
            VkDeviceSize sizes[4] = {token_bytes, qkv_proj_weight_layer_bytes, sizeof(float), qkv_raw_bytes};
            VkDeviceSize offsets[4] = {0, (VkDeviceSize)((size_t)layer_idx * qkv_proj_weight_layer_bytes), 0, 0};
            if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                        &rt->qkv_proj_descriptor_pools[layer_idx],
                        &rt->qkv_proj_descriptor_sets[layer_idx])) return 1;
        }
        if (rt->runtime_linear_wf16_coopmat_enabled &&
                rt->qkv_proj_weight_f16_buffer) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                VkBuffer buffers[4] = {
                    rt->norm_buffer, rt->qkv_proj_weight_f16_buffer, rt->dummy_bias_buffer, rt->qkv_raw_buffer
                };
                VkDeviceSize sizes[4] = {token_bytes, qkv_proj_weight_layer_f16_bytes, sizeof(float), qkv_raw_bytes};
                VkDeviceSize offsets[4] = {0, (VkDeviceSize)((size_t)layer_idx * qkv_proj_weight_layer_f16_bytes), 0, 0};
                if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                            &rt->qkv_proj_wf16_descriptor_pools[layer_idx],
                            &rt->qkv_proj_wf16_descriptor_sets[layer_idx])) return 1;
            }
        }
        {
            VkBuffer buffers[9] = {
                rt->qkv_raw_buffer, rt->x_pos_buffer, rt->y_pos_buffer, rt->t_pos_buffer,
                rt->xy_buffer, rt->inv_t_buffer, rt->q_buffer, rt->k_buffer, rt->v_buffer
            };
            VkDeviceSize sizes[9] = {
                qkv_raw_bytes, pos_bytes, pos_bytes, pos_bytes, xy_bytes, inv_t_bytes,
                q_rope_bytes, kv_rope_bytes, kv_rope_bytes
            };
            if (create_storage_descriptor_set(rt, rt->runtime_qkv_rms_rope_set_layout, 9, buffers, sizes, NULL,
                        &rt->qkv_rms_rope_descriptor_pool,
                        &rt->qkv_rms_rope_descriptor_set)) return 1;
        }
        if (rt->cfg.value_residual && rt->v_first_buffer) {
            VkBuffer buffers[9] = {
                rt->qkv_raw_buffer, rt->x_pos_buffer, rt->y_pos_buffer, rt->t_pos_buffer,
                rt->xy_buffer, rt->inv_t_buffer, rt->q_buffer, rt->k_buffer, rt->v_first_buffer
            };
            VkDeviceSize sizes[9] = {
                qkv_raw_bytes, pos_bytes, pos_bytes, pos_bytes, xy_bytes, inv_t_bytes,
                q_rope_bytes, kv_rope_bytes, kv_rope_bytes
            };
            if (create_storage_descriptor_set(rt, rt->runtime_qkv_rms_rope_set_layout, 9, buffers, sizes, NULL,
                        &rt->qkv_rms_rope_first_descriptor_pool,
                        &rt->qkv_rms_rope_first_descriptor_set)) return 1;
            {
                VkBuffer lerp_buffers[2] = {rt->v_buffer, rt->v_first_buffer};
                VkDeviceSize lerp_sizes[2] = {kv_rope_bytes, kv_rope_bytes};
                if (create_storage_descriptor_set(rt, rt->runtime_lerp_set_layout, 2,
                            lerp_buffers, lerp_sizes, NULL,
                            &rt->value_residual_descriptor_pool,
                            &rt->value_residual_descriptor_set)) return 1;
            }
        }
        if (rt->layer_attention_enabled) {
            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                VkDeviceSize kv_offset = (VkDeviceSize)(rt->cache_kv_offsets[layer_idx] * sizeof(float));
                VkDeviceSize meta_offset = (VkDeviceSize)(rt->cache_meta_offsets[layer_idx] * sizeof(uint32_t));
                VkDeviceSize count_offset = (VkDeviceSize)((size_t)layer_idx * sizeof(uint32_t));
                VkDeviceSize layer_cache_kv_bytes = (VkDeviceSize)((size_t)rt->cfg.n_kv_heads *
                    (size_t)rt->cache_capacities[layer_idx] * (size_t)rt->d_head * sizeof(float));
                VkDeviceSize layer_cache_meta_bytes = (VkDeviceSize)((size_t)rt->cache_capacities[layer_idx] * sizeof(uint32_t));
                VkBuffer buffers[5] = {
                    rt->cache_k_buffer, rt->cache_v_buffer, rt->k_buffer,
                    (rt->cfg.value_residual && layer_idx == 0 && rt->v_first_buffer) ? rt->v_first_buffer : rt->v_buffer,
                    rt->cache_written_buffer
                };
                VkDeviceSize sizes[5] = {
                    layer_cache_kv_bytes, layer_cache_kv_bytes, kv_rope_bytes, kv_rope_bytes, layer_cache_meta_bytes
                };
                VkDeviceSize offsets[5] = {kv_offset, kv_offset, 0, 0, meta_offset};
                if (create_storage_descriptor_set(rt, rt->runtime_kv_upsert_set_layout, 5, buffers, sizes, offsets,
                            &rt->kv_upsert_descriptor_pools[layer_idx],
                            &rt->kv_upsert_descriptor_sets[layer_idx])) return 1;
                {
                    VkBuffer index_buffers[3] = {
                    rt->cache_written_buffer, rt->cache_indices_buffer, rt->cache_index_count_buffer
                    };
                    VkDeviceSize index_sizes[3] = {layer_cache_meta_bytes, layer_cache_meta_bytes, sizeof(uint32_t)};
                    VkDeviceSize index_offsets[3] = {meta_offset, meta_offset, count_offset};
                    if (create_storage_descriptor_set(rt, rt->runtime_cache_indices_set_layout, 3,
                                index_buffers, index_sizes, index_offsets,
                                &rt->cache_indices_descriptor_pools[layer_idx],
                                &rt->cache_indices_descriptor_sets[layer_idx])) return 1;
                }
                {
                    VkBuffer attn_buffers[6] = {
                        rt->q_buffer, rt->cache_k_buffer, rt->cache_v_buffer, rt->cache_indices_buffer,
                        rt->attn_buffer, rt->cache_index_count_buffer
                    };
                    VkDeviceSize attn_sizes[6] = {
                        q_rope_bytes, layer_cache_kv_bytes, layer_cache_kv_bytes,
                        layer_cache_meta_bytes, token_bytes, sizeof(uint32_t)
                    };
                    VkDeviceSize attn_offsets[6] = {0, kv_offset, kv_offset, meta_offset, 0, count_offset};
                    if (create_storage_descriptor_set(rt, rt->runtime_indexed_attention_set_layout, 6,
                                attn_buffers, attn_sizes, attn_offsets,
                                &rt->indexed_attention_descriptor_pools[layer_idx],
                                &rt->indexed_attention_descriptor_sets[layer_idx])) return 1;
                }
            }
            if (rt->layer_attn_out_enabled) {
                for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                    VkBuffer buffers[4] = {
                        rt->attn_buffer, rt->attn_out_proj_weight_buffer, rt->dummy_bias_buffer, rt->attn_proj_buffer
                    };
                    VkDeviceSize sizes[4] = {token_bytes, attn_out_proj_weight_layer_bytes, sizeof(float), token_bytes};
                    VkDeviceSize offsets[4] = {
                        0,
                        (VkDeviceSize)((size_t)layer_idx * attn_out_proj_weight_layer_bytes),
                        0,
                        0
                    };
                    if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                                &rt->attn_out_proj_descriptor_pools[layer_idx],
                                &rt->attn_out_proj_descriptor_sets[layer_idx])) return 1;
                }
                if (rt->runtime_linear_wf16_coopmat_enabled &&
                        rt->attn_out_proj_weight_f16_buffer) {
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        VkBuffer buffers[4] = {
                            rt->attn_buffer, rt->attn_out_proj_weight_f16_buffer, rt->dummy_bias_buffer, rt->attn_proj_buffer
                        };
                        VkDeviceSize sizes[4] = {token_bytes, attn_out_proj_weight_layer_f16_bytes, sizeof(float), token_bytes};
                        VkDeviceSize offsets[4] = {
                            0,
                            (VkDeviceSize)((size_t)layer_idx * attn_out_proj_weight_layer_f16_bytes),
                            0,
                            0
                        };
                        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4,
                                    buffers, sizes, offsets,
                                    &rt->attn_out_proj_wf16_descriptor_pools[layer_idx],
                                    &rt->attn_out_proj_wf16_descriptor_sets[layer_idx])) return 1;
                    }
                }
                for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        int table_idx = pass_idx * rt->layers_to_run + layer_idx;
                        VkDeviceSize base = (VkDeviceSize)((size_t)table_idx * layer_mod_pass_layer_bytes);
                        VkBuffer tokens_in = (rt->layer_mlp_enabled && (layer_idx % 2) != 0) ?
                            rt->tokens_after_mlp_buffer : rt->tokens_buffer;
                        VkBuffer buffers[4] = {
                            tokens_in, rt->attn_proj_buffer, rt->layer_mod_table_buffer, rt->tokens_after_attn_buffer
                        };
                        VkDeviceSize sizes[4] = {token_bytes, token_bytes, ctrl_emb_bytes, token_bytes};
                        VkDeviceSize offsets[4] = {0, 0, base + (VkDeviceSize)(2 * ctrl_emb_bytes), 0};
                        if (create_storage_descriptor_set(rt, rt->runtime_gated_residual_set_layout, 4, buffers, sizes, offsets,
                                    &rt->attn_residual_descriptor_pools[table_idx],
                                    &rt->attn_residual_descriptor_sets[table_idx])) return 1;
                    }
                }
                if (rt->layer_ctrl_enabled) {
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        if (!rt->layer_has_ctrl[layer_idx]) continue;
                        VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx * ctrl_layer_weight_bytes);
                        VkDeviceSize cond_offset = (VkDeviceSize)((size_t)layer_idx * ctrl_emb_bytes);
                        VkBuffer buffers[4] = {
                            rt->ctrl_emb_norm_buffer, rt->ctrl_fc1_c_weight_buffer, rt->dummy_bias_buffer, rt->ctrl_cond_buffer
                        };
                        VkDeviceSize sizes[4] = {ctrl_emb_bytes, ctrl_layer_weight_bytes, sizeof(float), ctrl_emb_bytes};
                        VkDeviceSize offsets[4] = {0, weight_offset, 0, cond_offset};
                        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                                    &rt->ctrl_cond_descriptor_pools[layer_idx],
                                    &rt->ctrl_cond_descriptor_sets[layer_idx])) return 1;
                    }
                    {
                        VkBuffer buffers[3] = {rt->tokens_after_attn_buffer, rt->rms_weight_buffer, rt->ctrl_norm_buffer};
                        VkDeviceSize sizes[3] = {token_bytes, ctrl_emb_bytes, token_bytes};
                        if (create_storage_descriptor_set(rt, rt->runtime_rms_set_layout, 3, buffers, sizes, NULL,
                                    &rt->ctrl_norm_descriptor_pool,
                                    &rt->ctrl_norm_descriptor_set)) return 1;
                    }
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        if (!rt->layer_has_ctrl[layer_idx]) continue;
                        VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx * ctrl_layer_weight_bytes);
                        VkDeviceSize cond_offset = (VkDeviceSize)((size_t)layer_idx * ctrl_emb_bytes);
                        VkBuffer buffers[4] = {
                            rt->ctrl_norm_buffer, rt->ctrl_fc1_x_weight_buffer, rt->dummy_bias_buffer, rt->ctrl_hidden_layer_buffer
                        };
                        VkDeviceSize sizes[4] = {token_bytes, ctrl_layer_weight_bytes, sizeof(float), token_bytes};
                        VkDeviceSize offsets[4] = {0, weight_offset, 0, 0};
                        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                                    &rt->ctrl_fc1_x_descriptor_pools[layer_idx],
                                    &rt->ctrl_fc1_x_descriptor_sets[layer_idx])) return 1;
                        {
                            VkBuffer silu_buffers[3] = {rt->ctrl_hidden_layer_buffer, rt->ctrl_cond_buffer, rt->ctrl_hidden_layer_buffer};
                            VkDeviceSize silu_sizes[3] = {token_bytes, ctrl_emb_bytes, token_bytes};
                            VkDeviceSize silu_offsets[3] = {0, cond_offset, 0};
                            if (create_storage_descriptor_set(rt, rt->runtime_add_channel_silu_set_layout, 3,
                                        silu_buffers, silu_sizes, silu_offsets,
                                        &rt->ctrl_add_silu_descriptor_pools[layer_idx],
                                        &rt->ctrl_add_silu_descriptor_sets[layer_idx])) return 1;
                        }
                        {
                            VkBuffer fc2_buffers[4] = {
                                rt->ctrl_hidden_layer_buffer, rt->ctrl_fc2_weight_buffer_layer, rt->dummy_bias_buffer, rt->ctrl_out_buffer
                            };
                            VkDeviceSize fc2_sizes[4] = {token_bytes, ctrl_layer_weight_bytes, sizeof(float), token_bytes};
                            VkDeviceSize fc2_offsets[4] = {0, weight_offset, 0, 0};
                            if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4,
                                        fc2_buffers, fc2_sizes, fc2_offsets,
                                        &rt->ctrl_fc2_descriptor_pools_layer[layer_idx],
                                        &rt->ctrl_fc2_descriptor_sets_layer[layer_idx])) return 1;
                        }
                    }
                    {
                        VkBuffer buffers[3] = {rt->tokens_after_attn_buffer, rt->ctrl_out_buffer, rt->tokens_after_ctrl_buffer};
                        VkDeviceSize sizes[3] = {token_bytes, token_bytes, token_bytes};
                        if (create_storage_descriptor_set(rt, rt->runtime_add_set_layout, 3, buffers, sizes, NULL,
                                    &rt->ctrl_add_descriptor_pool,
                                    &rt->ctrl_add_descriptor_set)) return 1;
                    }
                }
                if (rt->layer_mlp_enabled) {
                    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            int table_idx = pass_idx * rt->layers_to_run + layer_idx;
                            VkDeviceSize base = (VkDeviceSize)((size_t)table_idx * layer_mod_pass_layer_bytes);
                            VkBuffer mlp_input_buffer = (rt->layer_ctrl_enabled && rt->layer_has_ctrl[layer_idx]) ?
                                rt->tokens_after_ctrl_buffer : rt->tokens_after_attn_buffer;
                            VkBuffer buffers[4] = {
                                mlp_input_buffer, rt->layer_mod_table_buffer, rt->layer_mod_table_buffer, rt->mlp_in_buffer
                            };
                            VkDeviceSize sizes[4] = {token_bytes, ctrl_emb_bytes, ctrl_emb_bytes, token_bytes};
                            VkDeviceSize offsets[4] = {
                                0,
                                base + (VkDeviceSize)(3 * ctrl_emb_bytes),
                                base + (VkDeviceSize)(4 * ctrl_emb_bytes),
                                0
                            };
                            if (create_storage_descriptor_set(rt, rt->runtime_ada_rms_set_layout, 4, buffers, sizes, offsets,
                                        &rt->mlp_ada_descriptor_pools[table_idx],
                                        &rt->mlp_ada_descriptor_sets[table_idx])) return 1;
                        }
                    }
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx * dit_mlp_fc1_weight_layer_bytes);
                        VkBuffer buffers[4] = {
                            rt->mlp_in_buffer, rt->dit_mlp_fc1_weight_buffer, rt->dummy_bias_buffer, rt->mlp_hidden_buffer
                        };
                        VkDeviceSize sizes[4] = {token_bytes, dit_mlp_fc1_weight_layer_bytes, sizeof(float), mlp_hidden_token_bytes};
                        VkDeviceSize offsets[4] = {0, weight_offset, 0, 0};
                        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                                    &rt->mlp_fc1_descriptor_pools[layer_idx],
                                    &rt->mlp_fc1_descriptor_sets[layer_idx])) return 1;
                    }
                    if (rt->runtime_linear_wf16_coopmat_enabled &&
                            rt->dit_mlp_fc1_weight_f16_buffer) {
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx *
                                    dit_mlp_fc1_weight_layer_f16_bytes);
                            VkBuffer buffers[4] = {
                                rt->mlp_in_buffer, rt->dit_mlp_fc1_weight_f16_buffer, rt->dummy_bias_buffer, rt->mlp_hidden_buffer
                            };
                            VkDeviceSize sizes[4] = {token_bytes, dit_mlp_fc1_weight_layer_f16_bytes, sizeof(float), mlp_hidden_token_bytes};
                            VkDeviceSize offsets[4] = {0, weight_offset, 0, 0};
                            if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4,
                                        buffers, sizes, offsets,
                                        &rt->mlp_fc1_wf16_descriptor_pools[layer_idx],
                                        &rt->mlp_fc1_wf16_descriptor_sets[layer_idx])) return 1;
                        }
                    }
                    {
                        VkBuffer buffers[2] = {rt->mlp_hidden_buffer, rt->mlp_hidden_buffer};
                        VkDeviceSize sizes[2] = {mlp_hidden_token_bytes, mlp_hidden_token_bytes};
                        if (create_storage_descriptor_set(rt, rt->runtime_silu_set_layout, 2, buffers, sizes, NULL,
                                    &rt->mlp_silu_descriptor_pool,
                                    &rt->mlp_silu_descriptor_set)) return 1;
                    }
                    if (rt->runtime_silu_f16_pipeline &&
                            rt->mlp_hidden_f16_buffer) {
                        VkBuffer buffers[2] = {rt->mlp_hidden_buffer, rt->mlp_hidden_f16_buffer};
                        VkDeviceSize sizes[2] = {mlp_hidden_token_bytes, mlp_hidden_token_f16_bytes};
                        if (create_storage_descriptor_set(rt, rt->runtime_silu_f16_set_layout, 2, buffers, sizes, NULL,
                                    &rt->mlp_silu_f16_descriptor_pool,
                                    &rt->mlp_silu_f16_descriptor_set)) return 1;
                    }
                    for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                        VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx * dit_mlp_fc2_weight_layer_bytes);
                        VkBuffer buffers[4] = {
                            rt->mlp_hidden_buffer, rt->dit_mlp_fc2_weight_buffer, rt->dummy_bias_buffer, rt->mlp_out_buffer
                        };
                        VkDeviceSize sizes[4] = {mlp_hidden_token_bytes, dit_mlp_fc2_weight_layer_bytes, sizeof(float), token_bytes};
                        VkDeviceSize offsets[4] = {0, weight_offset, 0, 0};
                        if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4, buffers, sizes, offsets,
                                    &rt->mlp_fc2_descriptor_pools[layer_idx],
                                    &rt->mlp_fc2_descriptor_sets[layer_idx])) return 1;
                    }
                    if (rt->runtime_linear_wf16_coopmat_enabled &&
                            rt->dit_mlp_fc2_weight_f16_buffer) {
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx *
                                    dit_mlp_fc2_weight_layer_f16_bytes);
                            VkBuffer buffers[4] = {
                                rt->mlp_hidden_buffer, rt->dit_mlp_fc2_weight_f16_buffer, rt->dummy_bias_buffer, rt->mlp_out_buffer
                            };
                            VkDeviceSize sizes[4] = {mlp_hidden_token_bytes, dit_mlp_fc2_weight_layer_f16_bytes, sizeof(float), token_bytes};
                            VkDeviceSize offsets[4] = {0, weight_offset, 0, 0};
                            if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4,
                                        buffers, sizes, offsets,
                                        &rt->mlp_fc2_wf16_descriptor_pools[layer_idx],
                                        &rt->mlp_fc2_wf16_descriptor_sets[layer_idx])) return 1;
                        }
                        if (rt->mlp_hidden_f16_buffer &&
                                rt->runtime_linear_f16x_wf16_coopmat_pipeline) {
                            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                                VkDeviceSize weight_offset = (VkDeviceSize)((size_t)layer_idx *
                                        dit_mlp_fc2_weight_layer_f16_bytes);
                                VkBuffer buffers[4] = {
                                    rt->mlp_hidden_f16_buffer, rt->dit_mlp_fc2_weight_f16_buffer, rt->dummy_bias_buffer, rt->mlp_out_buffer
                                };
                                VkDeviceSize sizes[4] = {mlp_hidden_token_f16_bytes, dit_mlp_fc2_weight_layer_f16_bytes, sizeof(float), token_bytes};
                                VkDeviceSize offsets[4] = {0, weight_offset, 0, 0};
                                if (create_storage_descriptor_set(rt, rt->runtime_linear_set_layout, 4,
                                            buffers, sizes, offsets,
                                            &rt->mlp_fc2_f16x_descriptor_pools[layer_idx],
                                            &rt->mlp_fc2_f16x_descriptor_sets[layer_idx])) return 1;
                            }
                        }
                    }
                    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
                        for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                            int table_idx = pass_idx * rt->layers_to_run + layer_idx;
                            VkDeviceSize base = (VkDeviceSize)((size_t)table_idx * layer_mod_pass_layer_bytes);
                            VkBuffer mlp_input_buffer = (rt->layer_ctrl_enabled && rt->layer_has_ctrl[layer_idx]) ?
                                rt->tokens_after_ctrl_buffer : rt->tokens_after_attn_buffer;
                            VkBuffer mlp_output_buffer = (layer_idx % 2) == 0 ?
                                rt->tokens_after_mlp_buffer : rt->tokens_buffer;
                            VkBuffer buffers[4] = {
                                mlp_input_buffer, rt->mlp_out_buffer, rt->layer_mod_table_buffer, mlp_output_buffer
                            };
                            VkDeviceSize sizes[4] = {token_bytes, token_bytes, ctrl_emb_bytes, token_bytes};
                            VkDeviceSize offsets[4] = {0, 0, base + (VkDeviceSize)(5 * ctrl_emb_bytes), 0};
                            if (create_storage_descriptor_set(rt, rt->runtime_gated_residual_set_layout, 4, buffers, sizes, offsets,
                                        &rt->mlp_residual_descriptor_pools[table_idx],
                                        &rt->mlp_residual_descriptor_sets[table_idx])) return 1;
                        }
                    }
                }
            }
        }
    }
    {
        VkBuffer final_input_buffer;
        if (rt->layer_mlp_enabled && rt->layers_to_run > 0) {
            final_input_buffer = (rt->layers_to_run % 2) != 0 ?
                rt->tokens_after_mlp_buffer : rt->tokens_buffer;
        } else {
            final_input_buffer = rt->layer_ctrl_enabled ? rt->tokens_after_ctrl_buffer :
                (rt->layer_attn_out_enabled ? rt->tokens_after_attn_buffer : rt->tokens_buffer);
        }
        for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
            VkBuffer buffers[3] = {final_input_buffer, rt->out_mod_table_buffer, rt->final_tokens_buffer};
            VkDeviceSize sizes[3] = {token_bytes, out_mod_pass_bytes, token_bytes};
            VkDeviceSize offsets[3] = {0, (VkDeviceSize)((size_t)pass_idx * out_mod_pass_bytes), 0};
            if (create_storage_descriptor_set(rt, rt->runtime_out_norm_silu_set_layout, 3, buffers, sizes, offsets,
                        &rt->out_norm_silu_descriptor_pool[pass_idx],
                        &rt->out_norm_silu_descriptor_set[pass_idx])) return 1;
        }
    }
    {
        VkBuffer buffers[3] = {rt->latent_buffer, rt->patch_weight_buffer, rt->tokens_buffer};
        VkDeviceSize sizes[3] = {latent_bytes, patch_weight_bytes, token_bytes};
        if (create_storage_descriptor_set(rt, rt->patchify_set_layout, 3, buffers, sizes, NULL,
                    &rt->patchify_descriptor_pool, &rt->patchify_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[4] = {
            rt->final_tokens_buffer,
            rt->unpatch_weight_buffer, rt->unpatch_bias_buffer, rt->latent_out_buffer
        };
        VkDeviceSize sizes[4] = {token_bytes, patch_weight_bytes, unpatch_bias_bytes, latent_bytes};
        if (create_storage_descriptor_set(rt, rt->unpatch_orig_set_layout, 4, buffers, sizes, NULL,
                    &rt->unpatch_orig_descriptor_pool, &rt->unpatch_orig_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[2] = {rt->latent_buffer, rt->latent_out_buffer};
        VkDeviceSize sizes[2] = {latent_bytes, latent_bytes};
        if (create_storage_descriptor_set(rt, rt->runtime_latent_update_set_layout, 2, buffers, sizes, NULL,
                    &rt->latent_update_descriptor_pool, &rt->latent_update_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[3] = {rt->latent_buffer, rt->output_buffer, rt->ctrl_emb_norm_buffer};
        VkDeviceSize sizes[3] = {latent_bytes, rt->pixel_count * sizeof(uint32_t), ctrl_emb_bytes};
        if (create_storage_descriptor_set(rt, rt->latent_rgba_set_layout, 3, buffers, sizes, NULL,
                    &rt->latent_rgba_descriptor_pool, &rt->latent_rgba_descriptor_set)) return 1;
    }

    rt->ctrl_embedding_enabled = 1;
    rt->denoise_out_norm_enabled = 1;
    rt->layer_mod_enabled = rt->layers_to_run > 0;
    if (precompute_runtime_out_mods(rt)) return 1;
    rt->model_slice_enabled = 1;
    fprintf(stderr,
            "Vulkan resident latent slice enabled: C=%d H=%d W=%d T=%d D=%d layers=%d qkv=%d attn=%d attn_out=%d ctrl=%d mlp=%d ctrl_dim=%d hidden=%d passes=%d bytes(latent)=%.2f MiB\n",
            rt->C, rt->H, rt->W, rt->T, rt->D, rt->layers_to_run,
            rt->layer_qkv_enabled, rt->layer_attention_enabled, rt->layer_attn_out_enabled,
            rt->layer_ctrl_enabled, rt->layer_mlp_enabled, rt->ctrl_dim, rt->mlp_hidden, rt->total_passes,
            (double)latent_bytes / (1024.0 * 1024.0));
    return 0;
}

static int record_runtime_model_slice(
        WorldVulkanRuntime *rt,
        const float *control_input) {
#define PROFILE_DISPATCH(label, gx, gy, gz) do { \
    uint32_t _profile_event = profile_begin_dispatch(rt, (label)); \
    vkCmdDispatch(rt->command_buffer, (gx), (gy), (gz)); \
    profile_end_dispatch(rt, _profile_event); \
} while (0)
#define LINEAR_PIPELINE(push) runtime_linear_pipeline_for(rt, (push).rows, (push).cols, (push).inner, (push).has_bias)
#define LINEAR_LAYOUT(push) runtime_linear_layout_for(rt, (push).rows, (push).cols, (push).inner, (push).has_bias)
#define LINEAR_DISPATCH(label, push) do { \
    uint32_t _linear_tile_x = runtime_linear_tile_x_for(rt, (push).rows, (push).cols, (push).inner, (push).has_bias); \
    uint32_t _linear_tile_y = runtime_linear_tile_y_for(rt, (push).rows, (push).cols, (push).inner, (push).has_bias); \
    PROFILE_DISPATCH((label), ((push).cols + _linear_tile_x - 1u) / _linear_tile_x, \
            ((push).rows + _linear_tile_y - 1u) / _linear_tile_y, 1); \
} while (0)

    if (rt->ctrl_embedding_enabled) {
        WorldVulkanLinearPush fc1_push;
        memset(&fc1_push, 0, sizeof(fc1_push));
        fc1_push.rows = 1;
        fc1_push.cols = (uint32_t)rt->mlp_hidden;
        fc1_push.inner = (uint32_t)rt->ctrl_dim;
        fc1_push.has_bias = 0;

        WorldVulkanSiluPush silu_push;
        memset(&silu_push, 0, sizeof(silu_push));
        silu_push.n = (uint32_t)rt->mlp_hidden;

        WorldVulkanLinearPush fc2_push;
        memset(&fc2_push, 0, sizeof(fc2_push));
        fc2_push.rows = 1;
        fc2_push.cols = (uint32_t)rt->D;
        fc2_push.inner = (uint32_t)rt->mlp_hidden;
        fc2_push.has_bias = 0;

        WorldVulkanRmsNormPush rms_push;
        memset(&rms_push, 0, sizeof(rms_push));
        rms_push.rows = 1;
        rms_push.cols = (uint32_t)rt->D;
        rms_push.eps = rt->rms_eps;

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, LINEAR_PIPELINE(fc1_push));
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                LINEAR_LAYOUT(fc1_push), 0, 1, &rt->ctrl_fc1_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, LINEAR_LAYOUT(fc1_push), VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(fc1_push), &fc1_push);
        LINEAR_DISPATCH("linear.ctrl_emb_fc1", fc1_push);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_silu_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_silu_pipeline_layout, 0, 1, &rt->ctrl_silu_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_silu_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(silu_push), &silu_push);
        PROFILE_DISPATCH("silu.ctrl_emb", ((uint32_t)rt->mlp_hidden + 255u) / 256u, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, LINEAR_PIPELINE(fc2_push));
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                LINEAR_LAYOUT(fc2_push), 0, 1, &rt->ctrl_fc2_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, LINEAR_LAYOUT(fc2_push), VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(fc2_push), &fc2_push);
        LINEAR_DISPATCH("linear.ctrl_emb_fc2", fc2_push);
        cmd_shader_barrier(rt->command_buffer);

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_rms_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->runtime_rms_pipeline_layout, 0, 1, &rt->ctrl_rms_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->runtime_rms_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(rms_push), &rms_push);
        PROFILE_DISPATCH("rms.ctrl_emb", 1, 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        if (rt->layer_ctrl_enabled) {
            WorldVulkanLinearPush ctrl_cond_push;
            memset(&ctrl_cond_push, 0, sizeof(ctrl_cond_push));
            ctrl_cond_push.rows = 1;
            ctrl_cond_push.cols = (uint32_t)rt->D;
            ctrl_cond_push.inner = (uint32_t)rt->D;
            ctrl_cond_push.has_bias = 0;

            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                if (!rt->layer_has_ctrl[layer_idx]) continue;
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, LINEAR_PIPELINE(ctrl_cond_push));
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        LINEAR_LAYOUT(ctrl_cond_push), 0, 1,
                        &rt->ctrl_cond_descriptor_sets[layer_idx], 0, NULL);
                vkCmdPushConstants(rt->command_buffer, LINEAR_LAYOUT(ctrl_cond_push),
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ctrl_cond_push), &ctrl_cond_push);
                LINEAR_DISPATCH("linear.ctrl_cond", ctrl_cond_push);
                cmd_shader_barrier(rt->command_buffer);
            }
        }
    }

    WorldVulkanPatchifyPush patch_push;
    memset(&patch_push, 0, sizeof(patch_push));
    patch_push.B = 1;
    patch_push.C = (uint32_t)rt->C;
    patch_push.H = (uint32_t)rt->H;
    patch_push.W = (uint32_t)rt->W;
    patch_push.D = (uint32_t)rt->D;
    patch_push.ph = (uint32_t)rt->ph;
    patch_push.pw = (uint32_t)rt->pw;
    patch_push.Hp = (uint32_t)rt->cfg.height;
    patch_push.Wp = (uint32_t)rt->cfg.width;

    WorldVulkanUnpatchifyOrigPush unpatch_push;
    memset(&unpatch_push, 0, sizeof(unpatch_push));
    unpatch_push.T = (uint32_t)rt->T;
    unpatch_push.D = (uint32_t)rt->D;
    unpatch_push.C = (uint32_t)rt->C;
    unpatch_push.H = (uint32_t)rt->H;
    unpatch_push.W = (uint32_t)rt->W;
    unpatch_push.ph = (uint32_t)rt->ph;
    unpatch_push.pw = (uint32_t)rt->pw;
    unpatch_push.Wp = (uint32_t)rt->cfg.width;
    unpatch_push.out_dim = (uint32_t)rt->out_dim;

    WorldVulkanLatentRgbaPush rgba_push;
    memset(&rgba_push, 0, sizeof(rgba_push));
    rgba_push.out_width = (uint32_t)rt->width;
    rgba_push.out_height = (uint32_t)rt->height;
    rgba_push.frames = (uint32_t)rt->frames;
    rgba_push.latent_c = (uint32_t)rt->C;
    rgba_push.latent_h = (uint32_t)rt->H;
    rgba_push.latent_w = (uint32_t)rt->W;
    rgba_push.frame_ordinal = (uint32_t)rt->frame_ordinal;
    rgba_push.ctrl_dim = (uint32_t)rt->D;
    if (control_input) {
        rgba_push.control_x = control_input[0];
        rgba_push.control_y = control_input[1];
    }

    {
        const char *vae_only = getenv("WORLD_VULKAN_VAE_ONLY");
        if (rt->vae_enabled && vae_only && vae_only[0] && vae_only[0] != '0') {
            record_runtime_vae_decode(rt);
            return 0;
        }
    }

    for (int pass_idx = 0; pass_idx < rt->total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= rt->steps_to_run;
        int table_pass_idx = rt->forced_pass_table_idx >= 0 ? rt->forced_pass_table_idx : pass_idx;
        if (table_pass_idx < 0 || table_pass_idx >= rt->precomputed_total_passes) {
            fprintf(stderr,
                    "invalid Vulkan runtime pass table index %d for precomputed_total_passes=%d\n",
                    table_pass_idx, rt->precomputed_total_passes);
            return 1;
        }

        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->patchify_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->patchify_pipeline_layout, 0, 1, &rt->patchify_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->patchify_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(patch_push), &patch_push);
        PROFILE_DISPATCH("patchify", (uint32_t)(rt->T * rt->D), 1, 1);
        cmd_shader_barrier(rt->command_buffer);

        if (rt->layer_qkv_enabled) {
            WorldVulkanAdaRmsNormPush ada_push;
            memset(&ada_push, 0, sizeof(ada_push));
            ada_push.B = 1;
            ada_push.T = (uint32_t)rt->T;
            ada_push.N = 1;
            ada_push.D = (uint32_t)rt->D;
            ada_push.eps = rt->rms_eps;

            WorldVulkanLinearPush qkv_push;
            memset(&qkv_push, 0, sizeof(qkv_push));
            qkv_push.rows = (uint32_t)rt->T;
            qkv_push.cols = (uint32_t)rt->qkv_dim;
            qkv_push.inner = (uint32_t)rt->D;
            qkv_push.has_bias = 0;

            WorldVulkanQkvRmsRopePush rope_push;
            memset(&rope_push, 0, sizeof(rope_push));
            rope_push.B = 1;
            rope_push.T = (uint32_t)rt->T;
            rope_push.n_heads = (uint32_t)rt->cfg.n_heads;
            rope_push.n_kv_heads = (uint32_t)rt->cfg.n_kv_heads;
            rope_push.D = (uint32_t)rt->d_head;
            rope_push.width = (uint32_t)rt->cfg.width;
            rope_push.height = (uint32_t)rt->cfg.height;
            rope_push.eps = rt->rms_eps;

            for (int layer_idx = 0; layer_idx < rt->layers_to_run; ++layer_idx) {
                int table_idx = table_pass_idx * rt->layers_to_run + layer_idx;

            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_ada_rms_pipeline);
            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    rt->runtime_ada_rms_pipeline_layout, 0, 1,
                    &rt->attn_ada_descriptor_sets[table_idx], 0, NULL);
            vkCmdPushConstants(rt->command_buffer, rt->runtime_ada_rms_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ada_push), &ada_push);
            PROFILE_DISPATCH("ada_rms.attn", (uint32_t)rt->T, 1, 1);
            cmd_shader_barrier(rt->command_buffer);

            int use_qkv_wf16 =
                rt->qkv_proj_wf16_descriptor_sets &&
                rt->qkv_proj_wf16_descriptor_sets[layer_idx] &&
                use_runtime_linear_wf16_coopmat(rt,
                        qkv_push.rows, qkv_push.cols,
                        qkv_push.inner, qkv_push.has_bias);
            int use_qkv_wf16_n16 =
                use_qkv_wf16 &&
                use_runtime_linear_wf16_coopmat_n16(rt,
                        qkv_push.rows, qkv_push.cols,
                        qkv_push.inner, qkv_push.has_bias);
            VkPipeline qkv_pipeline = use_qkv_wf16_n16 ?
                rt->runtime_linear_wf16_coopmat_n16_pipeline :
                (use_qkv_wf16 ? rt->runtime_linear_wf16_coopmat_pipeline : LINEAR_PIPELINE(qkv_push));
            VkPipelineLayout qkv_layout = use_qkv_wf16_n16 ?
                rt->runtime_linear_wf16_coopmat_n16_pipeline_layout :
                (use_qkv_wf16 ? rt->runtime_linear_wf16_coopmat_pipeline_layout : LINEAR_LAYOUT(qkv_push));
            VkDescriptorSet qkv_set = use_qkv_wf16 ?
                rt->qkv_proj_wf16_descriptor_sets[layer_idx] : rt->qkv_proj_descriptor_sets[layer_idx];
            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, qkv_pipeline);
            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    qkv_layout, 0, 1,
                    &qkv_set, 0, NULL);
            vkCmdPushConstants(rt->command_buffer, qkv_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qkv_push), &qkv_push);
            if (use_qkv_wf16_n16) {
                PROFILE_DISPATCH("linear.qkv_wf16_n16",
                        (qkv_push.cols + 15u) / 16u,
                        (qkv_push.rows + 15u) / 16u,
                        1);
            } else if (use_qkv_wf16) {
                PROFILE_DISPATCH("linear.qkv_wf16",
                        (qkv_push.cols + 7u) / 8u,
                        (qkv_push.rows + 15u) / 16u,
                        1);
            } else {
                LINEAR_DISPATCH("linear.qkv", qkv_push);
            }
            cmd_shader_barrier(rt->command_buffer);

            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_qkv_rms_rope_pipeline);
            VkDescriptorSet qkv_rope_set =
                (rt->cfg.value_residual && layer_idx == 0 && rt->qkv_rms_rope_first_descriptor_set) ?
                rt->qkv_rms_rope_first_descriptor_set : rt->qkv_rms_rope_descriptor_set;
            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    rt->runtime_qkv_rms_rope_pipeline_layout, 0, 1,
                    &qkv_rope_set, 0, NULL);
            vkCmdPushConstants(rt->command_buffer, rt->runtime_qkv_rms_rope_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rope_push), &rope_push);
            PROFILE_DISPATCH("qkv_rms_rope",
                    (uint32_t)rt->T,
                    (uint32_t)(rt->cfg.n_heads + 2 * rt->cfg.n_kv_heads),
                    1);
            cmd_shader_barrier(rt->command_buffer);

            if (rt->cfg.value_residual && layer_idx > 0 && rt->runtime_lerp_pipeline) {
                WorldVulkanLerpPush lerp_push;
                memset(&lerp_push, 0, sizeof(lerp_push));
                lerp_push.n = (uint32_t)(rt->T * rt->kv_dim);
                lerp_push.weight = rt->layer_v_lamb ? rt->layer_v_lamb[layer_idx] : 0.0f;
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_lerp_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->runtime_lerp_pipeline_layout, 0, 1,
                        &rt->value_residual_descriptor_set, 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->runtime_lerp_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lerp_push), &lerp_push);
                PROFILE_DISPATCH("lerp.value_residual",
                        ((uint32_t)(rt->T * rt->kv_dim) + 255u) / 256u,
                        1, 1);
                cmd_shader_barrier(rt->command_buffer);
            }

            if (rt->layer_attention_enabled) {
                int cache_ring_length = rt->cache_ring_lengths[layer_idx];
                int cache_capacity = rt->cache_capacities[layer_idx];
                int cache_pinned_dilation = rt->cache_pinned_dilations[layer_idx];
                uint32_t pinned = (uint32_t)cache_pinned_dilation;
                uint32_t bucket = ((uint32_t)rt->frame_ordinal + pinned - 1u) / pinned;
                uint32_t num_buckets = (uint32_t)((cache_ring_length / rt->T) / cache_pinned_dilation);
                uint32_t base = num_buckets > 0u ? (bucket % num_buckets) * (uint32_t)rt->T : 0u;
                uint32_t write_step = ((uint32_t)rt->frame_ordinal % pinned) == 0u ? 1u : 0u;

                WorldVulkanKvCacheUpsertPush upsert_push;
                memset(&upsert_push, 0, sizeof(upsert_push));
                upsert_push.B = 1;
                upsert_push.H = (uint32_t)rt->cfg.n_kv_heads;
                upsert_push.T = (uint32_t)rt->T;
                upsert_push.D = (uint32_t)rt->d_head;
                upsert_push.L = (uint32_t)cache_ring_length;
                upsert_push.base = base;
                upsert_push.write_step = write_step;
                upsert_push.frozen = is_cache_pass ? 0u : 1u;

                WorldVulkanCacheFrameIndicesPush indices_push;
                memset(&indices_push, 0, sizeof(indices_push));
                indices_push.capacity = (uint32_t)cache_capacity;
                indices_push.T = (uint32_t)rt->T;
                indices_push.base = base;
                indices_push.write_step = write_step;

                WorldVulkanIndexedAttentionPush attn_push;
                memset(&attn_push, 0, sizeof(attn_push));
                attn_push.B = 1;
                attn_push.Hq = (uint32_t)rt->cfg.n_heads;
                attn_push.Hkv = (uint32_t)rt->cfg.n_kv_heads;
                attn_push.Tq = (uint32_t)rt->T;
                attn_push.Nkv = (uint32_t)rt->T;
                attn_push.Tk = (uint32_t)cache_capacity;
                attn_push.D = (uint32_t)rt->d_head;
                attn_push.scale = 1.0f / sqrtf((float)rt->d_head);

                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_kv_upsert_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->runtime_kv_upsert_pipeline_layout, 0, 1,
                        &rt->kv_upsert_descriptor_sets[layer_idx], 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->runtime_kv_upsert_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upsert_push), &upsert_push);
                PROFILE_DISPATCH("kv_upsert",
                        ((uint32_t)(rt->cfg.n_kv_heads * rt->T * rt->d_head) + 255u) / 256u,
                        1, 1);
                cmd_shader_barrier(rt->command_buffer);

                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_cache_indices_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        rt->runtime_cache_indices_pipeline_layout, 0, 1,
                        &rt->cache_indices_descriptor_sets[layer_idx], 0, NULL);
                vkCmdPushConstants(rt->command_buffer, rt->runtime_cache_indices_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(indices_push), &indices_push);
                PROFILE_DISPATCH("cache_indices", (uint32_t)(cache_capacity / rt->T), 1, 1);
                cmd_shader_barrier(rt->command_buffer);

                VkPipeline attn_pipeline = rt->runtime_indexed_attention_pipeline;
                VkPipelineLayout attn_pipeline_layout = rt->runtime_indexed_attention_pipeline_layout;
                const char *attn_label = "attention.generic";
                uint32_t attn_dispatch_x = (uint32_t)(rt->cfg.n_heads * rt->T);
                const char *disable_d64_attn = getenv("WORLD_VULKAN_DISABLE_D64_ATTENTION");
                int d64_allowed = !(disable_d64_attn && disable_d64_attn[0] && disable_d64_attn[0] != '0');
                if (rt->d_head == 64 && d64_allowed) {
                    uint32_t group = rt->cfg.n_kv_heads > 0 ?
                        (uint32_t)(rt->cfg.n_heads / rt->cfg.n_kv_heads) : 0u;
                    if (rt->runtime_indexed_attention_d64_flash96_enabled &&
                            rt->runtime_indexed_attention_d64_flash96_pipeline &&
                            group > 0u && group <= 16u &&
                            (uint32_t)rt->cfg.n_heads == group * (uint32_t)rt->cfg.n_kv_heads) {
                        uint32_t q_per_h = 16u / group;
                        if (q_per_h == 0u) q_per_h = 1u;
                        uint32_t q_blocks = ((uint32_t)rt->T + q_per_h - 1u) / q_per_h;
                        attn_pipeline = rt->runtime_indexed_attention_d64_flash96_pipeline;
                        attn_pipeline_layout = rt->runtime_indexed_attention_d64_flash96_pipeline_layout;
                        attn_label = "attention.d64_flash96";
                        attn_dispatch_x = (uint32_t)rt->cfg.n_kv_heads * q_blocks;
                    } else if (rt->runtime_indexed_attention_d64_flash_enabled &&
                            rt->runtime_indexed_attention_d64_flash_pipeline &&
                            group > 0u && group <= 16u &&
                            (uint32_t)rt->cfg.n_heads == group * (uint32_t)rt->cfg.n_kv_heads) {
                        uint32_t q_per_h = 16u / group;
                        if (q_per_h == 0u) q_per_h = 1u;
                        uint32_t q_blocks = ((uint32_t)rt->T + q_per_h - 1u) / q_per_h;
                        attn_pipeline = rt->runtime_indexed_attention_d64_flash_pipeline;
                        attn_pipeline_layout = rt->runtime_indexed_attention_d64_flash_pipeline_layout;
                        attn_label = "attention.d64_flash";
                        attn_dispatch_x = (uint32_t)rt->cfg.n_kv_heads * q_blocks;
                    } else if (rt->runtime_indexed_attention_d64_pipeline) {
                        attn_pipeline = rt->runtime_indexed_attention_d64_pipeline;
                        attn_pipeline_layout = rt->runtime_indexed_attention_d64_pipeline_layout;
                        attn_label = "attention.d64";
                    }
                }
                vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, attn_pipeline);
                vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        attn_pipeline_layout, 0, 1,
                        &rt->indexed_attention_descriptor_sets[layer_idx], 0, NULL);
                vkCmdPushConstants(rt->command_buffer, attn_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(attn_push), &attn_push);
                PROFILE_DISPATCH(attn_label, attn_dispatch_x, 1, 1);
                cmd_shader_barrier(rt->command_buffer);

                if (rt->layer_attn_out_enabled) {
                    WorldVulkanLinearPush out_proj_push;
                    memset(&out_proj_push, 0, sizeof(out_proj_push));
                    out_proj_push.rows = (uint32_t)rt->T;
                    out_proj_push.cols = (uint32_t)rt->D;
                    out_proj_push.inner = (uint32_t)rt->D;
                    out_proj_push.has_bias = 0;

                    WorldVulkanGatedResidualPush residual_push;
                    memset(&residual_push, 0, sizeof(residual_push));
                    residual_push.T = (uint32_t)rt->T;
                    residual_push.D = (uint32_t)rt->D;

                    int use_attn_out_wf16 =
                        rt->attn_out_proj_wf16_descriptor_sets &&
                        rt->attn_out_proj_wf16_descriptor_sets[layer_idx] &&
                        use_runtime_linear_wf16_coopmat(rt,
                                out_proj_push.rows, out_proj_push.cols,
                                out_proj_push.inner, out_proj_push.has_bias);
                    VkPipeline out_proj_pipeline = use_attn_out_wf16 ?
                        rt->runtime_linear_wf16_coopmat_pipeline : LINEAR_PIPELINE(out_proj_push);
                    VkPipelineLayout out_proj_layout = use_attn_out_wf16 ?
                        rt->runtime_linear_wf16_coopmat_pipeline_layout : LINEAR_LAYOUT(out_proj_push);
                    VkDescriptorSet out_proj_set = use_attn_out_wf16 ?
                        rt->attn_out_proj_wf16_descriptor_sets[layer_idx] : rt->attn_out_proj_descriptor_sets[layer_idx];
                    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, out_proj_pipeline);
                    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            out_proj_layout, 0, 1,
                            &out_proj_set, 0, NULL);
                    vkCmdPushConstants(rt->command_buffer, out_proj_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(out_proj_push), &out_proj_push);
                    if (use_attn_out_wf16) {
                        PROFILE_DISPATCH("linear.attn_out_wf16",
                                (out_proj_push.cols + 7u) / 8u,
                                (out_proj_push.rows + 15u) / 16u,
                                1);
                    } else {
                        LINEAR_DISPATCH("linear.attn_out", out_proj_push);
                    }
                    cmd_shader_barrier(rt->command_buffer);

                    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_gated_residual_pipeline);
                    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            rt->runtime_gated_residual_pipeline_layout, 0, 1,
                            &rt->attn_residual_descriptor_sets[table_idx], 0, NULL);
                    vkCmdPushConstants(rt->command_buffer, rt->runtime_gated_residual_pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(residual_push), &residual_push);
                    PROFILE_DISPATCH("gated.attn_residual",
                            ((uint32_t)(rt->T * rt->D) + 255u) / 256u,
                            1, 1);
                    cmd_shader_barrier(rt->command_buffer);

                    if (rt->layer_ctrl_enabled && rt->layer_has_ctrl[layer_idx]) {
                        WorldVulkanRmsNormPush ctrl_norm_push;
                        memset(&ctrl_norm_push, 0, sizeof(ctrl_norm_push));
                        ctrl_norm_push.rows = (uint32_t)rt->T;
                        ctrl_norm_push.cols = (uint32_t)rt->D;
                        ctrl_norm_push.eps = rt->rms_eps;

                        WorldVulkanLinearPush ctrl_fc_push;
                        memset(&ctrl_fc_push, 0, sizeof(ctrl_fc_push));
                        ctrl_fc_push.rows = (uint32_t)rt->T;
                        ctrl_fc_push.cols = (uint32_t)rt->D;
                        ctrl_fc_push.inner = (uint32_t)rt->D;
                        ctrl_fc_push.has_bias = 0;

                        WorldVulkanAddChannelSiluPush add_silu_push;
                        memset(&add_silu_push, 0, sizeof(add_silu_push));
                        add_silu_push.rows = (uint32_t)rt->T;
                        add_silu_push.D = (uint32_t)rt->D;

                        WorldVulkanSiluPush add_push;
                        memset(&add_push, 0, sizeof(add_push));
                        add_push.n = (uint32_t)(rt->T * rt->D);

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_rms_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                rt->runtime_rms_pipeline_layout, 0, 1,
                                &rt->ctrl_norm_descriptor_set, 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, rt->runtime_rms_pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ctrl_norm_push), &ctrl_norm_push);
                        PROFILE_DISPATCH("rms.ctrl_fusion", (uint32_t)rt->T, 1, 1);
                        cmd_shader_barrier(rt->command_buffer);

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, LINEAR_PIPELINE(ctrl_fc_push));
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                LINEAR_LAYOUT(ctrl_fc_push), 0, 1,
                                &rt->ctrl_fc1_x_descriptor_sets[layer_idx], 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, LINEAR_LAYOUT(ctrl_fc_push),
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ctrl_fc_push), &ctrl_fc_push);
                        LINEAR_DISPATCH("linear.ctrl_fc1_x", ctrl_fc_push);
                        cmd_shader_barrier(rt->command_buffer);

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_add_channel_silu_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                rt->runtime_add_channel_silu_pipeline_layout, 0, 1,
                                &rt->ctrl_add_silu_descriptor_sets[layer_idx], 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, rt->runtime_add_channel_silu_pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(add_silu_push), &add_silu_push);
                        PROFILE_DISPATCH("add_channel_silu.ctrl",
                                ((uint32_t)(rt->T * rt->D) + 255u) / 256u,
                                1, 1);
                        cmd_shader_barrier(rt->command_buffer);

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, LINEAR_PIPELINE(ctrl_fc_push));
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                LINEAR_LAYOUT(ctrl_fc_push), 0, 1,
                                &rt->ctrl_fc2_descriptor_sets_layer[layer_idx], 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, LINEAR_LAYOUT(ctrl_fc_push),
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ctrl_fc_push), &ctrl_fc_push);
                        LINEAR_DISPATCH("linear.ctrl_fc2", ctrl_fc_push);
                        cmd_shader_barrier(rt->command_buffer);

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_add_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                rt->runtime_add_pipeline_layout, 0, 1,
                                &rt->ctrl_add_descriptor_set, 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, rt->runtime_add_pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(add_push), &add_push);
                        PROFILE_DISPATCH("add.ctrl_residual",
                                ((uint32_t)(rt->T * rt->D) + 255u) / 256u,
                                1, 1);
                        cmd_shader_barrier(rt->command_buffer);
                    }

                    if (rt->layer_mlp_enabled) {
                        WorldVulkanAdaRmsNormPush mlp_ada_push;
                        memset(&mlp_ada_push, 0, sizeof(mlp_ada_push));
                        mlp_ada_push.B = 1;
                        mlp_ada_push.T = (uint32_t)rt->T;
                        mlp_ada_push.N = 1;
                        mlp_ada_push.D = (uint32_t)rt->D;
                        mlp_ada_push.eps = rt->rms_eps;

                        WorldVulkanLinearPush mlp_fc1_push;
                        memset(&mlp_fc1_push, 0, sizeof(mlp_fc1_push));
                        mlp_fc1_push.rows = (uint32_t)rt->T;
                        mlp_fc1_push.cols = (uint32_t)rt->mlp_hidden;
                        mlp_fc1_push.inner = (uint32_t)rt->D;
                        mlp_fc1_push.has_bias = 0;

                        WorldVulkanSiluPush mlp_silu_push;
                        memset(&mlp_silu_push, 0, sizeof(mlp_silu_push));
                        mlp_silu_push.n = (uint32_t)(rt->T * rt->mlp_hidden);

                        WorldVulkanLinearPush mlp_fc2_push;
                        memset(&mlp_fc2_push, 0, sizeof(mlp_fc2_push));
                        mlp_fc2_push.rows = (uint32_t)rt->T;
                        mlp_fc2_push.cols = (uint32_t)rt->D;
                        mlp_fc2_push.inner = (uint32_t)rt->mlp_hidden;
                        mlp_fc2_push.has_bias = 0;

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_ada_rms_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                rt->runtime_ada_rms_pipeline_layout, 0, 1,
                                &rt->mlp_ada_descriptor_sets[table_idx], 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, rt->runtime_ada_rms_pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mlp_ada_push), &mlp_ada_push);
                        PROFILE_DISPATCH("ada_rms.mlp", (uint32_t)rt->T, 1, 1);
                        cmd_shader_barrier(rt->command_buffer);

                        int use_mlp_fc1_wf16 =
                            rt->mlp_fc1_wf16_descriptor_sets &&
                            rt->mlp_fc1_wf16_descriptor_sets[layer_idx] &&
                            use_runtime_linear_wf16_coopmat(rt,
                                    mlp_fc1_push.rows, mlp_fc1_push.cols,
                                    mlp_fc1_push.inner, mlp_fc1_push.has_bias);
                        VkPipeline mlp_fc1_pipeline = use_mlp_fc1_wf16 ?
                            rt->runtime_linear_wf16_coopmat_pipeline : LINEAR_PIPELINE(mlp_fc1_push);
                        VkPipelineLayout mlp_fc1_layout = use_mlp_fc1_wf16 ?
                            rt->runtime_linear_wf16_coopmat_pipeline_layout : LINEAR_LAYOUT(mlp_fc1_push);
                        VkDescriptorSet mlp_fc1_set = use_mlp_fc1_wf16 ?
                            rt->mlp_fc1_wf16_descriptor_sets[layer_idx] : rt->mlp_fc1_descriptor_sets[layer_idx];
                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, mlp_fc1_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mlp_fc1_layout, 0, 1,
                                &mlp_fc1_set, 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, mlp_fc1_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mlp_fc1_push), &mlp_fc1_push);
                        if (use_mlp_fc1_wf16) {
                            PROFILE_DISPATCH("linear.mlp_fc1_wf16",
                                    (mlp_fc1_push.cols + 7u) / 8u,
                                    (mlp_fc1_push.rows + 15u) / 16u,
                                    1);
                        } else {
                            LINEAR_DISPATCH("linear.mlp_fc1", mlp_fc1_push);
                        }
                        cmd_shader_barrier(rt->command_buffer);

                        int use_mlp_fc2_f16x =
                            rt->mlp_silu_f16_descriptor_set &&
                            rt->mlp_fc2_f16x_descriptor_sets &&
                            rt->mlp_fc2_f16x_descriptor_sets[layer_idx] &&
                            use_runtime_linear_f16x_wf16_coopmat(rt,
                                    mlp_fc2_push.rows, mlp_fc2_push.cols,
                                    mlp_fc2_push.inner, mlp_fc2_push.has_bias);
                        if (use_mlp_fc2_f16x) {
                            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rt->runtime_silu_f16_pipeline);
                            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rt->runtime_silu_f16_pipeline_layout, 0, 1,
                                    &rt->mlp_silu_f16_descriptor_set, 0, NULL);
                            vkCmdPushConstants(rt->command_buffer, rt->runtime_silu_f16_pipeline_layout,
                                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mlp_silu_push), &mlp_silu_push);
                            PROFILE_DISPATCH("silu.mlp_f16",
                                    ((uint32_t)(rt->T * rt->mlp_hidden) + 255u) / 256u,
                                    1, 1);
                        } else {
                            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_silu_pipeline);
                            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rt->runtime_silu_pipeline_layout, 0, 1,
                                    &rt->mlp_silu_descriptor_set, 0, NULL);
                            vkCmdPushConstants(rt->command_buffer, rt->runtime_silu_pipeline_layout,
                                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mlp_silu_push), &mlp_silu_push);
                            PROFILE_DISPATCH("silu.mlp",
                                    ((uint32_t)(rt->T * rt->mlp_hidden) + 255u) / 256u,
                                    1, 1);
                        }
                        cmd_shader_barrier(rt->command_buffer);

                        int use_mlp_fc2_wf16 =
                            rt->mlp_fc2_wf16_descriptor_sets &&
                            rt->mlp_fc2_wf16_descriptor_sets[layer_idx] &&
                            use_runtime_linear_wf16_coopmat(rt,
                                    mlp_fc2_push.rows, mlp_fc2_push.cols,
                                    mlp_fc2_push.inner, mlp_fc2_push.has_bias);
                        VkPipeline mlp_fc2_pipeline = use_mlp_fc2_f16x ?
                            rt->runtime_linear_f16x_wf16_coopmat_pipeline :
                            (use_mlp_fc2_wf16 ? rt->runtime_linear_wf16_coopmat_pipeline : LINEAR_PIPELINE(mlp_fc2_push));
                        VkPipelineLayout mlp_fc2_layout = use_mlp_fc2_f16x ?
                            rt->runtime_linear_f16x_wf16_coopmat_pipeline_layout :
                            (use_mlp_fc2_wf16 ? rt->runtime_linear_wf16_coopmat_pipeline_layout : LINEAR_LAYOUT(mlp_fc2_push));
                        VkDescriptorSet mlp_fc2_set = use_mlp_fc2_f16x ?
                            rt->mlp_fc2_f16x_descriptor_sets[layer_idx] :
                            (use_mlp_fc2_wf16 ? rt->mlp_fc2_wf16_descriptor_sets[layer_idx] : rt->mlp_fc2_descriptor_sets[layer_idx]);
                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, mlp_fc2_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mlp_fc2_layout, 0, 1,
                                &mlp_fc2_set, 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, mlp_fc2_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mlp_fc2_push), &mlp_fc2_push);
                        if (use_mlp_fc2_f16x) {
                            PROFILE_DISPATCH("linear.mlp_fc2_f16x_wf16",
                                    (mlp_fc2_push.cols + 7u) / 8u,
                                    (mlp_fc2_push.rows + 15u) / 16u,
                                    1);
                        } else if (use_mlp_fc2_wf16) {
                            PROFILE_DISPATCH("linear.mlp_fc2_wf16",
                                    (mlp_fc2_push.cols + 7u) / 8u,
                                    (mlp_fc2_push.rows + 15u) / 16u,
                                    1);
                        } else {
                            LINEAR_DISPATCH("linear.mlp_fc2", mlp_fc2_push);
                        }
                        cmd_shader_barrier(rt->command_buffer);

                        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_gated_residual_pipeline);
                        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                rt->runtime_gated_residual_pipeline_layout, 0, 1,
                                &rt->mlp_residual_descriptor_sets[table_idx], 0, NULL);
                        vkCmdPushConstants(rt->command_buffer, rt->runtime_gated_residual_pipeline_layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(residual_push), &residual_push);
                        PROFILE_DISPATCH("gated.mlp_residual",
                                ((uint32_t)(rt->T * rt->D) + 255u) / 256u,
                                1, 1);
                        cmd_shader_barrier(rt->command_buffer);
                    }
                }
            }
            }
        }

        if (!is_cache_pass) {
            WorldVulkanOutNormSiluPush out_norm_push;
            memset(&out_norm_push, 0, sizeof(out_norm_push));
            out_norm_push.rows = (uint32_t)rt->T;
            out_norm_push.D = (uint32_t)rt->D;
            out_norm_push.eps = rt->rms_eps;

            float sigma_step = rt->cfg.scheduler_sigmas[pass_idx];
            float next_sigma = rt->cfg.scheduler_sigmas[pass_idx + 1];
            WorldVulkanLatentUpdatePush update_push;
            memset(&update_push, 0, sizeof(update_push));
            update_push.n = (uint32_t)rt->latent_elems;
            update_push.dsigma = next_sigma - sigma_step;

            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_out_norm_silu_pipeline);
            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    rt->runtime_out_norm_silu_pipeline_layout, 0, 1,
                    &rt->out_norm_silu_descriptor_set[table_pass_idx], 0, NULL);
            vkCmdPushConstants(rt->command_buffer, rt->runtime_out_norm_silu_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(out_norm_push), &out_norm_push);
            PROFILE_DISPATCH("out_norm_silu", (uint32_t)rt->T, 1, 1);
            cmd_shader_barrier(rt->command_buffer);

            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->unpatch_orig_pipeline);
            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    rt->unpatch_orig_pipeline_layout, 0, 1, &rt->unpatch_orig_descriptor_set, 0, NULL);
            vkCmdPushConstants(rt->command_buffer, rt->unpatch_orig_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(unpatch_push), &unpatch_push);
            PROFILE_DISPATCH("unpatchify", (uint32_t)(rt->T * rt->out_dim), 1, 1);
            cmd_shader_barrier(rt->command_buffer);

            vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->runtime_latent_update_pipeline);
            vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    rt->runtime_latent_update_pipeline_layout, 0, 1,
                    &rt->latent_update_descriptor_set, 0, NULL);
            vkCmdPushConstants(rt->command_buffer, rt->runtime_latent_update_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(update_push), &update_push);
            PROFILE_DISPATCH("latent_update", ((uint32_t)rt->latent_elems + 255u) / 256u, 1, 1);
            cmd_shader_barrier(rt->command_buffer);
        }
    }

    if (rt->vae_enabled) {
        record_runtime_vae_decode(rt);
    } else {
        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->latent_rgba_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->latent_rgba_pipeline_layout, 0, 1, &rt->latent_rgba_descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->latent_rgba_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(rgba_push), &rgba_push);
        PROFILE_DISPATCH("latent_to_rgba",
                ((uint32_t)rt->width + 15u) / 16u,
                ((uint32_t)rt->height + 15u) / 16u,
                (uint32_t)rt->frames);
    }
#undef LINEAR_DISPATCH
#undef LINEAR_LAYOUT
#undef LINEAR_PIPELINE
#undef PROFILE_DISPATCH
    return 0;
}

int world_vulkan_runtime_create(
        WorldVulkanRuntime **out,
        const WorldConfig *cfg,
        const WorldModelProbeWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frame_idx,
        unsigned int seed,
        int noise_mode,
        const WorldVaeDecoderWeights *vae) {
    if (!out || !cfg) return 1;
    *out = NULL;

    WorldVulkanRuntime *rt = (WorldVulkanRuntime *)calloc(1, sizeof(*rt));
    if (!rt) return 1;
    rt->cfg = *cfg;
    rt->width = cfg->width * cfg->patch_w * 16;
    rt->height = cfg->height * cfg->patch_h * 16;
    rt->frames = 4;
    rt->frame_ordinal = frame_idx;
    rt->seed = seed;
    rt->noise_mode = noise_mode;
    if (weights && weights->layers && layers_to_run > 0 && layers_to_run <= weights->n_layers) {
        rt->layers_to_run = layers_to_run;
    } else {
        rt->layers_to_run = 0;
        if (layers_to_run > 0) {
            fprintf(stderr,
                    "warning: Vulkan layer modulation disabled for layers_to_run=%d n_layers=%d\n",
                    layers_to_run, weights ? weights->n_layers : 0);
        }
    }
    {
        int max_steps = cfg->scheduler_sigmas_count > 1 ? cfg->scheduler_sigmas_count - 1 : 1;
        rt->steps_to_run = (steps_to_run > 0 && steps_to_run <= max_steps) ? steps_to_run : max_steps;
        rt->total_passes = rt->steps_to_run + 1;
        rt->precomputed_total_passes = rt->total_passes;
        rt->forced_pass_table_idx = -1;
        if (rt->total_passes > WORLD_VULKAN_MAX_PASSES) {
            fprintf(stderr, "invalid Vulkan total_passes=%d max=%d\n", rt->total_passes, WORLD_VULKAN_MAX_PASSES);
            goto fail;
        }
    }
    rt->pixel_count = (size_t)rt->width * (size_t)rt->height * (size_t)rt->frames;
    rt->rgb_bytes = rt->pixel_count * 3;
    rt->rgb_host = (unsigned char *)malloc(rt->rgb_bytes);
    if (!rt->rgb_host) goto fail;

    fprintf(stderr,
            "creating resident Vulkan runtime scaffold: RGB %dx%d frames=%d shader_dir=%s\n",
            rt->width, rt->height, rt->frames, WORLD_VULKAN_SHADER_DIR);
    fprintf(stderr,
            "warning: Vulkan backend is partial; transformer/VAE kernels are being ported incrementally\n");

    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "worldmodel.cu";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "worldmodel.cu";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    VK_CALL(vkCreateInstance(&instance_info, NULL, &rt->instance));

    if (pick_physical_device(rt)) goto fail;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = rt->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    int has_shader_f16_ext = physical_device_has_extension(rt->physical_device,
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    int has_storage_16bit_ext = physical_device_has_extension(rt->physical_device,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    int has_coop_matrix_ext = physical_device_has_extension(rt->physical_device,
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkPhysicalDeviceShaderFloat16Int8Features shader_f16_avail;
    memset(&shader_f16_avail, 0, sizeof(shader_f16_avail));
    shader_f16_avail.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage_16bit_avail;
    memset(&storage_16bit_avail, 0, sizeof(storage_16bit_avail));
    storage_16bit_avail.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_matrix_avail;
    memset(&coop_matrix_avail, 0, sizeof(coop_matrix_avail));
    coop_matrix_avail.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 avail_features;
    memset(&avail_features, 0, sizeof(avail_features));
    avail_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    void *avail_chain = NULL;
    if (has_coop_matrix_ext) {
        coop_matrix_avail.pNext = avail_chain;
        avail_chain = &coop_matrix_avail;
    }
    storage_16bit_avail.pNext = avail_chain;
    avail_chain = &storage_16bit_avail;
    shader_f16_avail.pNext = avail_chain;
    avail_chain = &shader_f16_avail;
    avail_features.pNext = avail_chain;
    vkGetPhysicalDeviceFeatures2(rt->physical_device, &avail_features);

    const char *device_extensions[8];
    uint32_t device_extension_count = 0;
    VkPhysicalDeviceShaderFloat16Int8Features shader_f16_enable;
    memset(&shader_f16_enable, 0, sizeof(shader_f16_enable));
    shader_f16_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage_16bit_enable;
    memset(&storage_16bit_enable, 0, sizeof(storage_16bit_enable));
    storage_16bit_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_matrix_enable;
    memset(&coop_matrix_enable, 0, sizeof(coop_matrix_enable));
    coop_matrix_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    void *enable_chain = NULL;
    if (has_shader_f16_ext && shader_f16_avail.shaderFloat16) {
        shader_f16_enable.shaderFloat16 = VK_TRUE;
        shader_f16_enable.pNext = enable_chain;
        enable_chain = &shader_f16_enable;
        device_extensions[device_extension_count++] = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
        rt->shader_float16_enabled = 1;
    }
    if (has_storage_16bit_ext && storage_16bit_avail.storageBuffer16BitAccess) {
        storage_16bit_enable.storageBuffer16BitAccess = VK_TRUE;
        storage_16bit_enable.uniformAndStorageBuffer16BitAccess =
            storage_16bit_avail.uniformAndStorageBuffer16BitAccess;
        storage_16bit_enable.pNext = enable_chain;
        enable_chain = &storage_16bit_enable;
        device_extensions[device_extension_count++] = VK_KHR_16BIT_STORAGE_EXTENSION_NAME;
        rt->storage_16bit_enabled = 1;
    }
    if (has_coop_matrix_ext && coop_matrix_avail.cooperativeMatrix) {
        coop_matrix_enable.cooperativeMatrix = VK_TRUE;
        coop_matrix_enable.pNext = enable_chain;
        enable_chain = &coop_matrix_enable;
        device_extensions[device_extension_count++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        rt->cooperative_matrix_enabled = 1;
    }
    VkPhysicalDeviceFeatures2 enabled_features;
    memset(&enabled_features, 0, sizeof(enabled_features));
    enabled_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabled_features.pNext = enable_chain;
    fprintf(stderr,
            "Vulkan optional compute features: shaderFloat16=%d storage16=%d cooperativeMatrix=%d\n",
            rt->shader_float16_enabled, rt->storage_16bit_enabled, rt->cooperative_matrix_enabled);

    VkDeviceCreateInfo device_info;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pNext = &enabled_features;
    device_info.enabledExtensionCount = device_extension_count;
    device_info.ppEnabledExtensionNames = device_extension_count ? device_extensions : NULL;
    VK_CALL(vkCreateDevice(rt->physical_device, &device_info, NULL, &rt->device));
    vkGetDeviceQueue(rt->device, rt->queue_family, 0, &rt->queue);

    if (create_output_buffer(rt)) goto fail;
    if (create_fill_pipeline(rt)) goto fail;
    if (create_descriptors(rt)) goto fail;
    if (create_commands(rt)) goto fail;
    if (create_profile_queries(rt)) goto fail;
    if (create_runtime_model_slice(rt, weights)) goto fail;
    if (create_runtime_vae(rt, vae)) goto fail;

    *out = rt;
    return 0;

fail:
    world_vulkan_runtime_destroy(rt);
    return 1;
}

static void convert_rgba_to_rgb(WorldVulkanRuntime *rt) {
    const uint32_t *rgba = (const uint32_t *)rt->output_mapped;
    unsigned char *rgb = rt->rgb_host;
    for (size_t i = 0; i < rt->pixel_count; ++i) {
        uint32_t p = rgba[i];
        rgb[i * 3 + 0] = (unsigned char)(p & 0xffu);
        rgb[i * 3 + 1] = (unsigned char)((p >> 8) & 0xffu);
        rgb[i * 3 + 2] = (unsigned char)((p >> 16) & 0xffu);
    }
}

static int world_vulkan_runtime_step_pixels(
        WorldVulkanRuntime *rt,
        const float *control_input,
        const unsigned char **pixels_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out,
        int rgba_output) {
    if (!rt || !pixels_out || !width_out || !height_out || !frames_out) return 1;
    double t0 = now_seconds();
    if (rt->model_slice_enabled) {
        if (rt->use_external_latent_once) {
            rt->use_external_latent_once = 0;
        } else {
            fill_runtime_latent(rt, control_input);
        }
        copy_runtime_control(rt, control_input);
        if (rt->layer_qkv_enabled) {
            fill_runtime_positions(rt, rt->frame_ordinal * rt->frame_stride);
        }
    }

    WorldVulkanFillPush push;
    memset(&push, 0, sizeof(push));
    push.width = (uint32_t)rt->width;
    push.height = (uint32_t)rt->height;
    push.frames = (uint32_t)rt->frames;
    push.frame_ordinal = (uint32_t)rt->frame_ordinal;
    if (control_input) {
        push.control_x = control_input[0];
        push.control_y = control_input[1];
    }

    VK_CALL_RET(vkResetFences(rt->device, 1, &rt->fence));
    VK_CALL_RET(vkResetCommandBuffer(rt->command_buffer, 0));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL_RET(vkBeginCommandBuffer(rt->command_buffer, &begin_info));
    profile_begin_frame(rt);
    if (rt->model_slice_enabled) {
        if (record_runtime_model_slice(rt, control_input)) return 1;
    } else {
        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->fill_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->pipeline_layout, 0, 1, &rt->descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(push), &push);
        vkCmdDispatch(rt->command_buffer,
                ((uint32_t)rt->width + 15u) / 16u,
                ((uint32_t)rt->height + 15u) / 16u,
                (uint32_t)rt->frames);
    }
    record_output_readback(rt);
    VK_CALL_RET(vkEndCommandBuffer(rt->command_buffer));

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    VK_CALL_RET(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
    VK_CALL_RET(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));
    profile_report_frame(rt);

    const char *dump_latent_path = getenv("WORLD_DUMP_RUNTIME_LATENT");
    if (rt->model_slice_enabled && dump_latent_path && dump_latent_path[0] && rt->latent_mapped) {
        FILE *f = fopen(dump_latent_path, rt->frame_ordinal == 0 ? "wb" : "ab");
        if (!f) {
            fprintf(stderr, "failed to open Vulkan latent dump %s: %s\n", dump_latent_path, strerror(errno));
            return 1;
        }
        size_t wrote = fwrite(rt->latent_mapped, sizeof(float), rt->latent_elems, f);
        fclose(f);
        if (wrote != rt->latent_elems) {
            fprintf(stderr, "failed to write Vulkan latent dump %s\n", dump_latent_path);
            return 1;
        }
    }

    const unsigned char *pixels = (const unsigned char *)rt->output_mapped;
    const char *pixel_label = "rgba";
    if (!rgba_output) {
        convert_rgba_to_rgb(rt);
        pixels = rt->rgb_host;
        pixel_label = "rgb";
    }
    rt->frame_ordinal += 1;

    *pixels_out = pixels;
    *width_out = rt->width;
    *height_out = rt->height;
    *frames_out = rt->frames;
    double elapsed = now_seconds() - t0;
    if (seconds_out) *seconds_out = (float)elapsed;
    fprintf(stderr, "Vulkan %s timing: total=%.3fms %s_fps=%.3f\n",
            rt->model_slice_enabled ? "resident-slice" : "scaffold",
            elapsed * 1000.0,
            pixel_label,
            (double)rt->frames / elapsed);
    return 0;
}

int world_vulkan_runtime_step_rgb(
        WorldVulkanRuntime *rt,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    return world_vulkan_runtime_step_pixels(
            rt, control_input, rgb_out, width_out, height_out, frames_out, seconds_out, 0);
}

int world_vulkan_runtime_step_rgba(
        WorldVulkanRuntime *rt,
        const float *control_input,
        const unsigned char **rgba_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    return world_vulkan_runtime_step_pixels(
            rt, control_input, rgba_out, width_out, height_out, frames_out, seconds_out, 1);
}

int world_vulkan_runtime_seed_latent_rgb(
        WorldVulkanRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    int old_steps_to_run = rt ? rt->steps_to_run : 0;
    int old_total_passes = rt ? rt->total_passes : 0;
    int old_forced_pass_table_idx = rt ? rt->forced_pass_table_idx : -1;
    if (rt && latent && rt->model_slice_enabled && rt->latent_mapped) {
        memcpy(rt->latent_mapped, latent, rt->latent_elems * sizeof(float));
        rt->use_external_latent_once = 1;
        rt->steps_to_run = 0;
        rt->total_passes = 1;
        rt->forced_pass_table_idx = old_steps_to_run;
    }
    int rc = world_vulkan_runtime_step_rgb(rt, control_input, rgb_out, width_out, height_out, frames_out, seconds_out);
    if (rt) {
        rt->steps_to_run = old_steps_to_run;
        rt->total_passes = old_total_passes;
        rt->forced_pass_table_idx = old_forced_pass_table_idx;
    }
    return rc;
}

int world_vulkan_runtime_seed_latent_rgba(
        WorldVulkanRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgba_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    int old_steps_to_run = rt ? rt->steps_to_run : 0;
    int old_total_passes = rt ? rt->total_passes : 0;
    int old_forced_pass_table_idx = rt ? rt->forced_pass_table_idx : -1;
    if (rt && latent && rt->model_slice_enabled && rt->latent_mapped) {
        memcpy(rt->latent_mapped, latent, rt->latent_elems * sizeof(float));
        rt->use_external_latent_once = 1;
        rt->steps_to_run = 0;
        rt->total_passes = 1;
        rt->forced_pass_table_idx = old_steps_to_run;
    }
    int rc = world_vulkan_runtime_step_rgba(rt, control_input, rgba_out, width_out, height_out, frames_out, seconds_out);
    if (rt) {
        rt->steps_to_run = old_steps_to_run;
        rt->total_passes = old_total_passes;
        rt->forced_pass_table_idx = old_forced_pass_table_idx;
    }
    return rc;
}

static float probe_value(int i, float scale) {
    int v = (i * 37 + 17) % 29;
    return ((float)v - 14.0f) * scale;
}

static uint16_t probe_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t rounded = (mant + (1u << (shift - 1u))) >> shift;
        return (uint16_t)(sign | rounded);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    mant += 0x1000u;
    if (mant & 0x800000u) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float probe_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static int run_probe_compute(
        WorldVulkanRuntime *rt,
        const char *shader_name,
        uint32_t binding_count,
        size_t push_size,
        VkBuffer *buffers,
        VkDeviceSize *sizes,
        const void *push,
        uint32_t gx,
        uint32_t gy,
        uint32_t gz) {
    int rc = 1;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    if (create_storage_pipeline(rt, shader_name, binding_count, push_size,
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    if (create_storage_descriptor_set(rt, set_layout, binding_count, buffers, sizes, NULL,
                &descriptor_pool, &descriptor_set)) goto cleanup;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, push, push_size, gx, gy, gz)) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    return rc;
}

static void fill_probe_rope_tables(float *xy, float *inv_t, int D, int height, int width) {
    int d_t = D / 4;
    int d_xy = D / 8;
    float max_freq = (float)(height < width ? height : width) * 0.8f;
    int n = (d_xy + 1) / 2;
    for (int i = 0; i < d_xy; ++i) {
        int k = i / 2;
        float a = n > 1 ? (float)k / (float)(n - 1) : 0.0f;
        xy[i] = (1.0f + (max_freq * 0.5f - 1.0f) * a) * (float)M_PI;
    }
    for (int i = 0; i < d_t; ++i) {
        int even = (i / 2) * 2;
        float exponent = (float)even / (float)d_t;
        inv_t[i] = 1.0f / powf(10000.0f, exponent);
    }
}

static float probe_rope_phase(
        int pair_id,
        uint32_t x_pos,
        uint32_t y_pos,
        uint32_t t_pos,
        const float *xy,
        const float *inv_t,
        int D,
        int width,
        int height) {
    int d_xy = D / 8;
    if (pair_id < d_xy) {
        float x = (2.0f * (float)x_pos + 1.0f) / (float)width - 1.0f;
        return x * xy[pair_id];
    }
    if (pair_id < 2 * d_xy) {
        float y = (2.0f * (float)y_pos + 1.0f) / (float)height - 1.0f;
        return y * xy[pair_id - d_xy];
    }
    return (float)t_pos * inv_t[pair_id - 2 * d_xy];
}

int world_vulkan_linear_f32_probe(void) {
    enum { rows = 7, cols = 11, inner = 13 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;

    size_t x_bytes = (size_t)rows * inner * sizeof(float);
    size_t w_bytes = (size_t)cols * inner * sizeof(float);
    size_t b_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    float *w = (float *)w_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * inner; ++i) x[i] = probe_value(i, 0.03125f);
    for (int i = 0; i < cols * inner; ++i) w[i] = probe_value(i + 101, 0.0234375f);
    for (int i = 0; i < cols; ++i) b[i] = probe_value(i + 211, 0.015625f);
    memset(y, 0, y_bytes);

    if (create_shader_module_from_name(rt, "linear_f32.comp", &shader)) goto cleanup;

    VkDescriptorSetLayoutBinding bindings[4];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set_info;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 4;
    set_info.pBindings = bindings;
    VK_CALL(vkCreateDescriptorSetLayout(rt->device, &set_info, NULL, &set_layout));

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(WorldVulkanLinearPush);

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VK_CALL(vkCreatePipelineLayout(rt->device, &layout_info, NULL, &pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    VK_CALL(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 4;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CALL(vkCreateDescriptorPool(rt->device, &pool_info, NULL, &descriptor_pool));

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;
    VK_CALL(vkAllocateDescriptorSets(rt->device, &alloc_info, &descriptor_set));

    VkDescriptorBufferInfo buffer_infos[4];
    memset(buffer_infos, 0, sizeof(buffer_infos));
    buffer_infos[0].buffer = x_buffer;
    buffer_infos[0].range = x_bytes;
    buffer_infos[1].buffer = w_buffer;
    buffer_infos[1].range = w_bytes;
    buffer_infos[2].buffer = b_buffer;
    buffer_infos[2].range = b_bytes;
    buffer_infos[3].buffer = y_buffer;
    buffer_infos[3].range = y_bytes;

    VkWriteDescriptorSet writes[4];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(rt->device, 4, writes, 0, NULL);

    WorldVulkanLinearPush push;
    push.rows = rows;
    push.cols = cols;
    push.inner = inner;
    push.has_bias = 1;

    VK_CALL(vkResetFences(rt->device, 1, &rt->fence));
    VK_CALL(vkResetCommandBuffer(rt->command_buffer, 0));
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(rt->command_buffer, &begin_info));
    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdPushConstants(rt->command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(push), &push);
    vkCmdDispatch(rt->command_buffer, (cols + 15) / 16, (rows + 15) / 16, 1);
    VK_CALL(vkEndCommandBuffer(rt->command_buffer));

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    VK_CALL(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
    VK_CALL(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float ref = b[c];
            for (int k = 0; k < inner; ++k) {
                ref += x[r * inner + k] * w[c * inner + k];
            }
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan linear_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 1.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;

fail:
    rc = 1;
    goto cleanup;
}

int world_vulkan_linear_f16_coopmat_probe(void) {
    enum { rows = 32, cols = 16, inner = 32 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (!rt->shader_float16_enabled || !rt->storage_16bit_enabled || !rt->cooperative_matrix_enabled) {
        fprintf(stderr, "vulkan linear_f16_coopmat probe: skipped; required optional features unavailable\n");
        rc = 0;
        goto cleanup;
    }

    size_t x_bytes = (size_t)rows * inner * sizeof(uint16_t);
    size_t w_bytes = (size_t)cols * inner * sizeof(uint16_t);
    size_t y_bytes = (size_t)rows * cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    uint16_t *x = (uint16_t *)x_mapped;
    uint16_t *w = (uint16_t *)w_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * inner; ++i) x[i] = probe_f32_to_f16(probe_value(i + 1703, 0.03125f));
    for (int i = 0; i < cols * inner; ++i) w[i] = probe_f32_to_f16(probe_value(i + 1901, 0.0234375f));

    WorldVulkanLinearPush push;
    memset(&push, 0, sizeof(push));
    push.rows = rows;
    push.cols = cols;
    push.inner = inner;
    push.has_bias = 0;

    VkBuffer buffers[3] = {x_buffer, w_buffer, y_buffer};
    VkDeviceSize sizes[3] = {x_bytes, w_bytes, y_bytes};
    if (run_probe_compute(rt, "linear_f16_coopmat.comp", 3, sizeof(push), buffers, sizes, &push,
                (uint32_t)(cols / 8), (uint32_t)(rows / 16), 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float ref = 0.0f;
            for (int k = 0; k < inner; ++k) {
                ref += probe_f16_to_f32(x[r * inner + k]) * probe_f16_to_f32(w[c * inner + k]);
            }
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan linear_f16_coopmat probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-4f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_linear_f32_coopmat_probe(void) {
    enum { rows = 32, cols = 16, inner = 32 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (!rt->shader_float16_enabled || !rt->cooperative_matrix_enabled) {
        fprintf(stderr, "vulkan linear_f32_coopmat probe: skipped; required optional features unavailable\n");
        rc = 0;
        goto cleanup;
    }

    size_t x_bytes = (size_t)rows * inner * sizeof(float);
    size_t w_bytes = (size_t)cols * inner * sizeof(float);
    size_t b_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    float *w = (float *)w_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * inner; ++i) x[i] = probe_value(i + 2011, 0.03125f);
    for (int i = 0; i < cols * inner; ++i) w[i] = probe_value(i + 2203, 0.0234375f);
    memset(b, 0, b_bytes);
    memset(y, 0, y_bytes);

    WorldVulkanLinearPush push;
    memset(&push, 0, sizeof(push));
    push.rows = rows;
    push.cols = cols;
    push.inner = inner;
    push.has_bias = 0;

    VkBuffer buffers[4] = {x_buffer, w_buffer, b_buffer, y_buffer};
    VkDeviceSize sizes[4] = {x_bytes, w_bytes, b_bytes, y_bytes};
    if (run_probe_compute(rt, "linear_f32_coopmat.comp", 4, sizeof(push), buffers, sizes, &push,
                (uint32_t)(cols / 8), (uint32_t)(rows / 16), 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float ref = 0.0f;
            for (int k = 0; k < inner; ++k) {
                float xh = probe_f16_to_f32(probe_f32_to_f16(x[r * inner + k]));
                float wh = probe_f16_to_f32(probe_f32_to_f16(w[c * inner + k]));
                ref += xh * wh;
            }
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan linear_f32_coopmat probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-4f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_linear_f32_wf16_coopmat_probe(void) {
    enum { rows = 32, cols = 16, inner = 32 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (!rt->shader_float16_enabled || !rt->storage_16bit_enabled || !rt->cooperative_matrix_enabled) {
        fprintf(stderr, "vulkan linear_f32_wf16_coopmat probe: skipped; required optional features unavailable\n");
        rc = 0;
        goto cleanup;
    }

    size_t x_bytes = (size_t)rows * inner * sizeof(float);
    size_t w_bytes = (size_t)cols * inner * sizeof(uint16_t);
    size_t b_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    uint16_t *w = (uint16_t *)w_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * inner; ++i) x[i] = probe_value(i + 2309, 0.03125f);
    for (int i = 0; i < cols * inner; ++i) {
        w[i] = probe_f32_to_f16(probe_value(i + 2503, 0.0234375f));
    }
    memset(b, 0, b_bytes);
    memset(y, 0, y_bytes);

    WorldVulkanLinearPush push;
    memset(&push, 0, sizeof(push));
    push.rows = rows;
    push.cols = cols;
    push.inner = inner;
    push.has_bias = 0;

    VkBuffer buffers[4] = {x_buffer, w_buffer, b_buffer, y_buffer};
    VkDeviceSize sizes[4] = {x_bytes, w_bytes, b_bytes, y_bytes};
    const char *shader_names[2] = {
        "linear_f32_wf16_coopmat.comp",
        "linear_f32_wf16_coopmat_n16.comp",
    };
    uint32_t col_tiles[2] = {8u, 16u};
    for (int variant = 0; variant < 2; ++variant) {
        memset(y, 0, y_bytes);
        if (run_probe_compute(rt, shader_names[variant], 4, sizeof(push), buffers, sizes, &push,
                    (uint32_t)(cols / col_tiles[variant]), (uint32_t)(rows / 16), 1)) goto cleanup;

        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float ref = 0.0f;
                for (int k = 0; k < inner; ++k) {
                    float xh = probe_f16_to_f32(probe_f32_to_f16(x[r * inner + k]));
                    float wh = probe_f16_to_f32(w[c * inner + k]);
                    ref += xh * wh;
                }
                float diff = fabsf(y[r * cols + c] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
        mean_abs /= (float)(rows * cols);
        fprintf(stderr, "vulkan %s probe: max_abs=%g mean_abs=%g\n",
                shader_names[variant], max_abs, mean_abs);
        if (max_abs > 2.0e-4f) goto cleanup;
    }
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_linear_f16x_wf16_coopmat_probe(void) {
    enum { rows = 32, cols = 16, inner = 32 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (!rt->shader_float16_enabled || !rt->storage_16bit_enabled || !rt->cooperative_matrix_enabled) {
        fprintf(stderr, "vulkan linear_f16x_wf16_coopmat probe: skipped; required optional features unavailable\n");
        rc = 0;
        goto cleanup;
    }

    size_t x_bytes = (size_t)rows * inner * sizeof(uint16_t);
    size_t w_bytes = (size_t)cols * inner * sizeof(uint16_t);
    size_t b_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    uint16_t *x = (uint16_t *)x_mapped;
    uint16_t *w = (uint16_t *)w_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * inner; ++i) {
        x[i] = probe_f32_to_f16(probe_value(i + 2707, 0.03125f));
    }
    for (int i = 0; i < cols * inner; ++i) {
        w[i] = probe_f32_to_f16(probe_value(i + 2909, 0.0234375f));
    }
    memset(b, 0, b_bytes);
    memset(y, 0, y_bytes);

    WorldVulkanLinearPush push;
    memset(&push, 0, sizeof(push));
    push.rows = rows;
    push.cols = cols;
    push.inner = inner;
    push.has_bias = 0;

    VkBuffer buffers[4] = {x_buffer, w_buffer, b_buffer, y_buffer};
    VkDeviceSize sizes[4] = {x_bytes, w_bytes, b_bytes, y_bytes};
    if (run_probe_compute(rt, "linear_f16x_wf16_coopmat.comp", 4, sizeof(push), buffers, sizes, &push,
                (uint32_t)(cols / 8), (uint32_t)(rows / 16), 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float ref = 0.0f;
            for (int k = 0; k < inner; ++k) {
                ref += probe_f16_to_f32(x[r * inner + k]) * probe_f16_to_f32(w[c * inner + k]);
            }
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan linear_f16x_wf16_coopmat probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-4f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_silu_f32_to_f16_probe(void) {
    enum { n = 513 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (!rt->shader_float16_enabled || !rt->storage_16bit_enabled) {
        fprintf(stderr, "vulkan silu_f32_to_f16 probe: skipped; required optional features unavailable\n");
        rc = 0;
        goto cleanup;
    }

    size_t x_bytes = (size_t)n * sizeof(float);
    size_t y_bytes = (size_t)n * sizeof(uint16_t);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    uint16_t *y = (uint16_t *)y_mapped;
    for (int i = 0; i < n; ++i) {
        x[i] = probe_value(i + 3109, 0.0625f);
    }
    memset(y, 0, y_bytes);

    WorldVulkanSiluPush push;
    memset(&push, 0, sizeof(push));
    push.n = n;

    VkBuffer buffers[2] = {x_buffer, y_buffer};
    VkDeviceSize sizes[2] = {x_bytes, y_bytes};
    if (run_probe_compute(rt, "silu_f32_to_f16.comp", 2, sizeof(push), buffers, sizes, &push,
                (uint32_t)((n + 255) / 256), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = x[i] / (1.0f + expf(-x[i]));
        ref = probe_f16_to_f32(probe_f32_to_f16(ref));
        float got = probe_f16_to_f32(y[i]);
        float diff = fabsf(got - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan silu_f32_to_f16 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 1.0e-3f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_silu_f32_probe(void) {
    enum { n = 513 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t bytes = (size_t)n * sizeof(float);
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < n; ++i) x[i] = probe_value(i + 31, 0.125f);
    memset(y, 0, bytes);

    if (create_storage_pipeline(rt, "silu_f32.comp", 2, sizeof(WorldVulkanSiluPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[2] = {x_buffer, y_buffer};
    VkDeviceSize sizes[2] = {bytes, bytes};
    if (create_storage_descriptor_set(rt, set_layout, 2, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanSiluPush push;
    push.n = n;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (n + 255u) / 256u, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = x[i] / (1.0f + expf(-x[i]));
        float diff = fabsf(y[i] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan silu_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-6f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_add_bias_silu_f32_probe(void) {
    enum { n = 4099 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t bytes = (size_t)n * sizeof(float);
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < n; ++i) {
        x[i] = probe_value(i + 41, 0.09375f);
        b[i] = probe_value(i + 73, 0.0625f);
    }
    memset(y, 0, bytes);

    if (create_storage_pipeline(rt, "add_bias_silu_f32.comp", 3, sizeof(WorldVulkanSiluPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, b_buffer, y_buffer};
    VkDeviceSize sizes[3] = {bytes, bytes, bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanSiluPush push;
    push.n = n;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (n + 255u) / 256u, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = x[i] + b[i];
        float ref = v / (1.0f + expf(-v));
        float diff = fabsf(y[i] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan add_bias_silu_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-6f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_add_channel_silu_f32_probe(void) {
    enum { rows = 9, D = 257, n = rows * D };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer bias_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory bias_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *bias_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)n * sizeof(float);
    size_t bias_bytes = (size_t)D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, bias_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bias_buffer, &bias_memory, &bias_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *bias = (float *)bias_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < n; ++i) x[i] = probe_value(i + 83, 0.0546875f);
    for (int d = 0; d < D; ++d) bias[d] = probe_value(d + 131, 0.03125f);
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "add_channel_silu_f32.comp", 3, sizeof(WorldVulkanAddChannelSiluPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, bias_buffer, y_buffer};
    VkDeviceSize sizes[3] = {x_bytes, bias_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanAddChannelSiluPush push;
    push.rows = rows;
    push.D = D;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (uint32_t)((n + 255) / 256), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = x[i] + bias[i % D];
        float ref = v / (1.0f + expf(-v));
        float diff = fabsf(y[i] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan add_channel_silu_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-6f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, bias_buffer, bias_memory, bias_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_add_f32_probe(void) {
    enum { n = 4099 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer a_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory a_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *a_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t bytes = (size_t)n * sizeof(float);
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &a_buffer, &a_memory, &a_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *a = (float *)a_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < n; ++i) {
        a[i] = probe_value(i + 193, 0.0625f);
        b[i] = probe_value(i + 251, 0.046875f);
    }
    memset(y, 0, bytes);

    if (create_storage_pipeline(rt, "add_f32.comp", 3, sizeof(WorldVulkanSiluPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {a_buffer, b_buffer, y_buffer};
    VkDeviceSize sizes[3] = {bytes, bytes, bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanSiluPush push;
    push.n = n;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (uint32_t)((n + 255) / 256), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = a[i] + b[i];
        float diff = fabsf(y[i] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan add_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs != 0.0f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, a_buffer, a_memory, a_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_out_norm_silu_f32_probe(void) {
    enum { rows = 5, D = 96 };
    const float eps = 1.0e-6f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer mod_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory mod_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *mod_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)rows * D * sizeof(float);
    size_t mod_bytes = (size_t)2 * D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, mod_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mod_buffer, &mod_memory, &mod_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *mod = (float *)mod_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * D; ++i) x[i] = probe_value(i + 271, 0.09375f);
    for (int i = 0; i < 2 * D; ++i) mod[i] = probe_value(i + 331, 0.03125f);
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "out_norm_silu_f32.comp", 3, sizeof(WorldVulkanOutNormSiluPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, mod_buffer, y_buffer};
    VkDeviceSize sizes[3] = {x_bytes, mod_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanOutNormSiluPush push;
    memset(&push, 0, sizeof(push));
    push.rows = rows;
    push.D = D;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                rows, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = x[r * D + d];
            sum += v * v;
        }
        float inv = 1.0f / sqrtf(sum / (float)D + eps);
        for (int d = 0; d < D; ++d) {
            float v = x[r * D + d] * inv * (1.0f + mod[d]) + mod[D + d];
            float ref = v / (1.0f + expf(-v));
            float diff = fabsf(y[r * D + d] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * D);
    fprintf(stderr, "vulkan out_norm_silu_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-6f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, mod_buffer, mod_memory, mod_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_latent_update_f32_probe(void) {
    enum { n = 1031 };
    const float dsigma = -0.1875f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer latent_buffer = VK_NULL_HANDLE;
    VkBuffer velocity_buffer = VK_NULL_HANDLE;
    VkDeviceMemory latent_memory = VK_NULL_HANDLE;
    VkDeviceMemory velocity_memory = VK_NULL_HANDLE;
    void *latent_mapped = NULL;
    void *velocity_mapped = NULL;
    float ref[n];

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t bytes = (size_t)n * sizeof(float);
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &latent_buffer, &latent_memory, &latent_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &velocity_buffer, &velocity_memory, &velocity_mapped)) goto cleanup;
    float *latent = (float *)latent_mapped;
    float *velocity = (float *)velocity_mapped;
    for (int i = 0; i < n; ++i) {
        latent[i] = probe_value(i + 419, 0.125f);
        velocity[i] = probe_value(i + 557, 0.0625f);
        ref[i] = latent[i] + dsigma * velocity[i];
    }

    if (create_storage_pipeline(rt, "latent_update_f32.comp", 2, sizeof(WorldVulkanLatentUpdatePush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[2] = {latent_buffer, velocity_buffer};
    VkDeviceSize sizes[2] = {bytes, bytes};
    if (create_storage_descriptor_set(rt, set_layout, 2, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanLatentUpdatePush push;
    memset(&push, 0, sizeof(push));
    push.n = n;
    push.dsigma = dsigma;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (uint32_t)((n + 255) / 256), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(latent[i] - ref[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan latent_update_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs != 0.0f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, latent_buffer, latent_memory, latent_mapped);
        destroy_host_buffer(rt, velocity_buffer, velocity_memory, velocity_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_lerp_inplace_f32_probe(void) {
    enum { n = 1543 };
    const float weight = 0.3125f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer end_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory end_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *end_mapped = NULL;
    float ref[n];

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t bytes = (size_t)n * sizeof(float);
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &end_buffer, &end_memory, &end_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *end = (float *)end_mapped;
    for (int i = 0; i < n; ++i) {
        x[i] = probe_value(i + 613, 0.078125f);
        end[i] = probe_value(i + 701, 0.09375f);
        ref[i] = x[i] + weight * (end[i] - x[i]);
    }

    if (create_storage_pipeline(rt, "lerp_inplace_f32.comp", 2, sizeof(WorldVulkanLerpPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[2] = {x_buffer, end_buffer};
    VkDeviceSize sizes[2] = {bytes, bytes};
    if (create_storage_descriptor_set(rt, set_layout, 2, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanLerpPush push;
    memset(&push, 0, sizeof(push));
    push.n = n;
    push.weight = weight;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (uint32_t)((n + 255) / 256), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(x[i] - ref[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan lerp_inplace_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs != 0.0f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, end_buffer, end_memory, end_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_taehv_primitives_probe(void) {
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) return 1;

    {
        enum { C = 3, H = 2, W = 5 };
        int ok = 1;
        VkBuffer latent_buffer = VK_NULL_HANDLE, out_buffer = VK_NULL_HANDLE;
        VkDeviceMemory latent_memory = VK_NULL_HANDLE, out_memory = VK_NULL_HANDLE;
        void *latent_mapped = NULL, *out_mapped = NULL;
        size_t latent_bytes = (size_t)C * H * W * sizeof(float);
        size_t out_bytes = (size_t)4 * C * H * W * sizeof(float);
        if (create_host_buffer(rt, latent_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &latent_buffer, &latent_memory, &latent_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) ok = 0;
        if (ok) {
            float *latent = (float *)latent_mapped;
            float *out = (float *)out_mapped;
            for (int i = 0; i < C * H * W; ++i) latent[i] = probe_value(i + 11, 0.75f);
            memset(out, 0, out_bytes);
            VkBuffer buffers[2] = {latent_buffer, out_buffer};
            VkDeviceSize sizes[2] = {latent_bytes, out_bytes};
            WorldVulkanTaehvRepeatClampPush push = {C, H, W};
            if (run_probe_compute(rt, "taehv_repeat_clamp_f32.comp", 2, sizeof(push),
                        buffers, sizes, &push, (uint32_t)((4 * C * H * W + 255) / 256), 1, 1)) ok = 0;
            float max_abs = 0.0f, mean_abs = 0.0f;
            for (int i = 0; ok && i < 4 * C * H * W; ++i) {
                float ref = tanhf(latent[i % (C * H * W)] / 3.0f) * 3.0f;
                float diff = fabsf(out[i] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
            mean_abs /= (float)(4 * C * H * W);
            fprintf(stderr, "vulkan taehv_repeat_clamp_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
            if (max_abs > 2.0e-6f) ok = 0;
        }
        destroy_host_buffer(rt, latent_buffer, latent_memory, latent_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
        if (!ok) goto cleanup;
    }

    {
        enum { N = 211 };
        int ok = 1;
        VkBuffer x_buffer = VK_NULL_HANDLE, a_buffer = VK_NULL_HANDLE, b_buffer = VK_NULL_HANDLE, out_buffer = VK_NULL_HANDLE;
        VkDeviceMemory x_memory = VK_NULL_HANDLE, a_memory = VK_NULL_HANDLE, b_memory = VK_NULL_HANDLE, out_memory = VK_NULL_HANDLE;
        void *x_mapped = NULL, *a_mapped = NULL, *b_mapped = NULL, *out_mapped = NULL;
        size_t bytes = (size_t)N * sizeof(float);
        if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &a_buffer, &a_memory, &a_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) ok = 0;
        if (ok) {
            float *x = (float *)x_mapped;
            float *a = (float *)a_mapped;
            float *b = (float *)b_mapped;
            float *out = (float *)out_mapped;
            for (int i = 0; i < N; ++i) {
                x[i] = probe_value(i + 23, 0.125f);
                a[i] = probe_value(i + 31, 0.25f);
                b[i] = probe_value(i + 47, 0.1875f);
                out[i] = 0.0f;
            }
            VkBuffer relu_buffers[1] = {x_buffer};
            VkDeviceSize relu_sizes[1] = {bytes};
            WorldVulkanSiluPush push = {N};
            if (run_probe_compute(rt, "taehv_relu_f32.comp", 1, sizeof(push),
                        relu_buffers, relu_sizes, &push, (uint32_t)((N + 255) / 256), 1, 1)) ok = 0;
            float relu_max = 0.0f;
            for (int i = 0; ok && i < N; ++i) {
                float ref = probe_value(i + 23, 0.125f);
                if (ref < 0.0f) ref = 0.0f;
                float diff = fabsf(x[i] - ref);
                if (diff > relu_max) relu_max = diff;
            }
            VkBuffer add_buffers[3] = {a_buffer, b_buffer, out_buffer};
            VkDeviceSize add_sizes[3] = {bytes, bytes, bytes};
            if (ok && run_probe_compute(rt, "taehv_add_relu_f32.comp", 3, sizeof(push),
                        add_buffers, add_sizes, &push, (uint32_t)((N + 255) / 256), 1, 1)) ok = 0;
            float add_max = 0.0f;
            for (int i = 0; ok && i < N; ++i) {
                float ref = a[i] + b[i];
                if (ref < 0.0f) ref = 0.0f;
                float diff = fabsf(out[i] - ref);
                if (diff > add_max) add_max = diff;
            }
            fprintf(stderr, "vulkan taehv_relu/add_relu_f32 probe: relu_max=%g add_relu_max=%g\n", relu_max, add_max);
            if (relu_max != 0.0f || add_max != 0.0f) ok = 0;
        }
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, a_buffer, a_memory, a_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
        if (!ok) goto cleanup;
    }

    {
        enum { N = 2, C_IN = 3, C_OUT = 4, H = 5, W = 4, K = 3 };
        int ok = 1;
        VkBuffer x_buffer = VK_NULL_HANDLE, w_buffer = VK_NULL_HANDLE, b_buffer = VK_NULL_HANDLE, y_buffer = VK_NULL_HANDLE, y_c4_buffer = VK_NULL_HANDLE;
        VkDeviceMemory x_memory = VK_NULL_HANDLE, w_memory = VK_NULL_HANDLE, b_memory = VK_NULL_HANDLE, y_memory = VK_NULL_HANDLE, y_c4_memory = VK_NULL_HANDLE;
        void *x_mapped = NULL, *w_mapped = NULL, *b_mapped = NULL, *y_mapped = NULL, *y_c4_mapped = NULL;
        size_t x_bytes = (size_t)N * C_IN * H * W * sizeof(float);
        size_t w_bytes = (size_t)C_OUT * C_IN * K * K * sizeof(float);
        size_t b_bytes = (size_t)C_OUT * sizeof(float);
        size_t y_bytes = (size_t)N * C_OUT * H * W * sizeof(float);
        if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_c4_buffer, &y_c4_memory, &y_c4_mapped)) ok = 0;
        if (ok) {
            float *x = (float *)x_mapped;
            float *w = (float *)w_mapped;
            float *b = (float *)b_mapped;
            float *y = (float *)y_mapped;
            float *y_c4 = (float *)y_c4_mapped;
            for (int i = 0; i < N * C_IN * H * W; ++i) x[i] = probe_value(i + 59, 0.03125f);
            for (int i = 0; i < C_OUT * C_IN * K * K; ++i) w[i] = probe_value(i + 73, 0.0234375f);
            for (int i = 0; i < C_OUT; ++i) b[i] = probe_value(i + 89, 0.015625f);
            memset(y, 0, y_bytes);
            memset(y_c4, 0, y_bytes);
            VkBuffer buffers[4] = {x_buffer, w_buffer, b_buffer, y_buffer};
            VkDeviceSize sizes[4] = {x_bytes, w_bytes, b_bytes, y_bytes};
            WorldVulkanTaehvConv2dPush push = {N, C_IN, C_OUT, H, W, K, 1};
            if (run_probe_compute(rt, "taehv_conv2d_nchw_f32.comp", 4, sizeof(push),
                        buffers, sizes, &push, (uint32_t)((N * C_OUT * H * W + 255) / 256), 1, 1)) ok = 0;
            VkBuffer c4_buffers[4] = {x_buffer, w_buffer, b_buffer, y_c4_buffer};
            if (ok && run_probe_compute(rt, "taehv_conv2d_nchw_c4_f32.comp", 4, sizeof(push),
                        c4_buffers, sizes, &push, (uint32_t)((N * ((C_OUT + 3) / 4) * H * W + 255) / 256), 1, 1)) ok = 0;
            float max_abs = 0.0f, mean_abs = 0.0f, c4_max_abs = 0.0f, c4_mean_abs = 0.0f;
            for (int n = 0; ok && n < N; ++n) {
                for (int co = 0; co < C_OUT; ++co) {
                    for (int yy = 0; yy < H; ++yy) {
                        for (int xx = 0; xx < W; ++xx) {
                            float ref = b[co];
                            for (int ci = 0; ci < C_IN; ++ci) {
                                for (int ky = 0; ky < K; ++ky) {
                                    int iy = yy + ky - K / 2;
                                    if (iy < 0 || iy >= H) continue;
                                    for (int kx = 0; kx < K; ++kx) {
                                        int ix = xx + kx - K / 2;
                                        if (ix < 0 || ix >= W) continue;
                                        ref += x[((n * C_IN + ci) * H + iy) * W + ix] *
                                            w[(((co * C_IN + ci) * K + ky) * K + kx)];
                                    }
                                }
                            }
                            int idx = ((n * C_OUT + co) * H + yy) * W + xx;
                            float diff = fabsf(y[idx] - ref);
                            float c4_diff = fabsf(y_c4[idx] - ref);
                            if (diff > max_abs) max_abs = diff;
                            if (c4_diff > c4_max_abs) c4_max_abs = c4_diff;
                            mean_abs += diff;
                            c4_mean_abs += c4_diff;
                        }
                    }
                }
            }
            mean_abs /= (float)(N * C_OUT * H * W);
            c4_mean_abs /= (float)(N * C_OUT * H * W);
            fprintf(stderr, "vulkan taehv_conv2d_nchw_f32 probe: max_abs=%g mean_abs=%g c4_max_abs=%g c4_mean_abs=%g\n",
                    max_abs, mean_abs, c4_max_abs, c4_mean_abs);
            if (max_abs > 2.0e-6f || c4_max_abs > 2.0e-6f) ok = 0;
            if (ok) {
                for (int i = 0; i < C_OUT * C_IN; ++i) w[i] = probe_value(i + 97, 0.02734375f);
                memset(y, 0, y_bytes);
                memset(y_c4, 0, y_bytes);
                WorldVulkanTaehvConv2dPush push1 = {N, C_IN, C_OUT, H, W, 1, 1};
                if (run_probe_compute(rt, "taehv_conv2d_nchw_f32.comp", 4, sizeof(push1),
                            buffers, sizes, &push1, (uint32_t)((N * C_OUT * H * W + 255) / 256), 1, 1)) ok = 0;
                if (ok && run_probe_compute(rt, "taehv_conv2d_nchw_1x1_c4_f32.comp", 4, sizeof(push1),
                            c4_buffers, sizes, &push1, (uint32_t)((N * ((C_OUT + 3) / 4) * H * W + 255) / 256), 1, 1)) ok = 0;
                float k1_max_abs = 0.0f;
                float k1_c4_max_abs = 0.0f;
                for (int n = 0; ok && n < N; ++n) {
                    for (int co = 0; co < C_OUT; ++co) {
                        for (int yy = 0; yy < H; ++yy) {
                            for (int xx = 0; xx < W; ++xx) {
                                float ref = b[co];
                                for (int ci = 0; ci < C_IN; ++ci) {
                                    ref += x[((n * C_IN + ci) * H + yy) * W + xx] *
                                        w[co * C_IN + ci];
                                }
                                int idx = ((n * C_OUT + co) * H + yy) * W + xx;
                                float diff = fabsf(y[idx] - ref);
                                float c4_diff = fabsf(y_c4[idx] - ref);
                                if (diff > k1_max_abs) k1_max_abs = diff;
                                if (c4_diff > k1_c4_max_abs) k1_c4_max_abs = c4_diff;
                            }
                        }
                    }
                }
                fprintf(stderr, "vulkan taehv_conv2d_nchw_k1_f32 probe: max_abs=%g c4_max_abs=%g\n",
                        k1_max_abs, k1_c4_max_abs);
                if (k1_max_abs > 2.0e-6f || k1_c4_max_abs > 2.0e-6f) ok = 0;
            }
        }
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
        destroy_host_buffer(rt, y_c4_buffer, y_c4_memory, y_c4_mapped);
        if (!ok) goto cleanup;
    }

    {
        enum { N = 4, C = 2, H = 3, W = 5 };
        int ok = 1;
        VkBuffer x_buffer = VK_NULL_HANDLE, y_buffer = VK_NULL_HANDLE;
        VkDeviceMemory x_memory = VK_NULL_HANDLE, y_memory = VK_NULL_HANDLE;
        void *x_mapped = NULL, *y_mapped = NULL;
        size_t x_bytes = (size_t)N * C * H * W * sizeof(float);
        size_t y_bytes = (size_t)N * 2 * C * H * W * sizeof(float);
        if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) ok = 0;
        if (ok) {
            float *x = (float *)x_mapped;
            float *y = (float *)y_mapped;
            for (int i = 0; i < N * C * H * W; ++i) x[i] = probe_value(i + 97, 0.046875f);
            memset(y, 0, y_bytes);
            VkBuffer buffers[2] = {x_buffer, y_buffer};
            VkDeviceSize sizes[2] = {x_bytes, y_bytes};
            WorldVulkanTaehvNchwPush push = {N, C, H, W};
            if (run_probe_compute(rt, "taehv_concat_past_nchw_f32.comp", 2, sizeof(push),
                        buffers, sizes, &push, (uint32_t)((N * 2 * C * H * W + 255) / 256), 1, 1)) ok = 0;
            float max_abs = 0.0f;
            for (int n = 0; ok && n < N; ++n) {
                for (int c2 = 0; c2 < 2 * C; ++c2) {
                    for (int yy = 0; yy < H; ++yy) {
                        for (int xx = 0; xx < W; ++xx) {
                            float ref = 0.0f;
                            if (c2 < C) ref = x[((n * C + c2) * H + yy) * W + xx];
                            else if (n > 0) ref = x[(((n - 1) * C + (c2 - C)) * H + yy) * W + xx];
                            int idx = ((n * 2 * C + c2) * H + yy) * W + xx;
                            float diff = fabsf(y[idx] - ref);
                            if (diff > max_abs) max_abs = diff;
                        }
                    }
                }
            }
            fprintf(stderr, "vulkan taehv_concat_past_nchw_f32 probe: max_abs=%g\n", max_abs);
            if (max_abs != 0.0f) ok = 0;
        }
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
        if (!ok) goto cleanup;
    }

    {
        enum { N = 2, C = 3, H = 2, W = 4, STRIDE = 2 };
        int ok = 1;
        VkBuffer x_buffer = VK_NULL_HANDLE, up_buffer = VK_NULL_HANDLE, tgrow_in_buffer = VK_NULL_HANDLE, tgrow_out_buffer = VK_NULL_HANDLE;
        VkDeviceMemory x_memory = VK_NULL_HANDLE, up_memory = VK_NULL_HANDLE, tgrow_in_memory = VK_NULL_HANDLE, tgrow_out_memory = VK_NULL_HANDLE;
        void *x_mapped = NULL, *up_mapped = NULL, *tgrow_in_mapped = NULL, *tgrow_out_mapped = NULL;
        size_t x_bytes = (size_t)N * C * H * W * sizeof(float);
        size_t up_bytes = (size_t)N * C * H * 2 * W * 2 * sizeof(float);
        size_t tgrow_in_bytes = (size_t)N * C * STRIDE * H * W * sizeof(float);
        size_t tgrow_out_bytes = (size_t)N * STRIDE * C * H * W * sizeof(float);
        if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, up_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &up_buffer, &up_memory, &up_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, tgrow_in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tgrow_in_buffer, &tgrow_in_memory, &tgrow_in_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, tgrow_out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tgrow_out_buffer, &tgrow_out_memory, &tgrow_out_mapped)) ok = 0;
        if (ok) {
            float *x = (float *)x_mapped;
            float *up = (float *)up_mapped;
            float *tgrow_in = (float *)tgrow_in_mapped;
            float *tgrow_out = (float *)tgrow_out_mapped;
            for (int i = 0; i < N * C * H * W; ++i) x[i] = probe_value(i + 113, 0.0625f);
            for (int i = 0; i < N * C * STRIDE * H * W; ++i) tgrow_in[i] = probe_value(i + 127, 0.0390625f);
            memset(up, 0, up_bytes);
            memset(tgrow_out, 0, tgrow_out_bytes);
            VkBuffer up_buffers[2] = {x_buffer, up_buffer};
            VkDeviceSize up_sizes[2] = {x_bytes, up_bytes};
            WorldVulkanTaehvNchwPush up_push = {N, C, H, W};
            if (run_probe_compute(rt, "taehv_upsample2_nchw_f32.comp", 2, sizeof(up_push),
                        up_buffers, up_sizes, &up_push, (uint32_t)((N * C * H * 2 * W * 2 + 255) / 256), 1, 1)) ok = 0;
            float up_max = 0.0f;
            for (int n = 0; ok && n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int yy = 0; yy < H * 2; ++yy) {
                        for (int xx = 0; xx < W * 2; ++xx) {
                            int idx = ((n * C + c) * (H * 2) + yy) * (W * 2) + xx;
                            float ref = x[((n * C + c) * H + yy / 2) * W + xx / 2];
                            float diff = fabsf(up[idx] - ref);
                            if (diff > up_max) up_max = diff;
                        }
                    }
                }
            }
            VkBuffer tgrow_buffers[2] = {tgrow_in_buffer, tgrow_out_buffer};
            VkDeviceSize tgrow_sizes[2] = {tgrow_in_bytes, tgrow_out_bytes};
            WorldVulkanTaehvTgrowPush tgrow_push = {N, C, H, W, STRIDE};
            if (ok && run_probe_compute(rt, "taehv_tgrow_reshape_nchw_f32.comp", 2, sizeof(tgrow_push),
                        tgrow_buffers, tgrow_sizes, &tgrow_push, (uint32_t)((N * STRIDE * C * H * W + 255) / 256), 1, 1)) ok = 0;
            float tgrow_max = 0.0f;
            for (int n = 0; ok && n < N; ++n) {
                for (int s = 0; s < STRIDE; ++s) {
                    for (int c = 0; c < C; ++c) {
                        for (int yy = 0; yy < H; ++yy) {
                            for (int xx = 0; xx < W; ++xx) {
                                int out_idx = (((n * STRIDE + s) * C + c) * H + yy) * W + xx;
                                int in_idx = ((n * (C * STRIDE) + (s * C + c)) * H + yy) * W + xx;
                                float diff = fabsf(tgrow_out[out_idx] - tgrow_in[in_idx]);
                                if (diff > tgrow_max) tgrow_max = diff;
                            }
                        }
                    }
                }
            }
            fprintf(stderr, "vulkan taehv_upsample2/tgrow_f32 probe: upsample_max=%g tgrow_max=%g\n", up_max, tgrow_max);
            if (up_max != 0.0f || tgrow_max != 0.0f) ok = 0;
        }
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, up_buffer, up_memory, up_mapped);
        destroy_host_buffer(rt, tgrow_in_buffer, tgrow_in_memory, tgrow_in_mapped);
        destroy_host_buffer(rt, tgrow_out_buffer, tgrow_out_memory, tgrow_out_mapped);
        if (!ok) goto cleanup;
    }

    {
        enum { N = 6, H = 3, W = 4 };
        int ok = 1;
        VkBuffer x_buffer = VK_NULL_HANDLE, rgba_buffer = VK_NULL_HANDLE;
        VkDeviceMemory x_memory = VK_NULL_HANDLE, rgba_memory = VK_NULL_HANDLE;
        void *x_mapped = NULL, *rgba_mapped = NULL;
        size_t x_bytes = (size_t)N * 12 * H * W * sizeof(float);
        size_t rgba_bytes = (size_t)4 * H * 2 * W * 2 * sizeof(uint32_t);
        if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, rgba_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &rgba_buffer, &rgba_memory, &rgba_mapped)) ok = 0;
        if (ok) {
            float *x = (float *)x_mapped;
            uint32_t *rgba = (uint32_t *)rgba_mapped;
            for (int i = 0; i < N * 12 * H * W; ++i) x[i] = probe_value(i + 149, 0.125f) + 0.5f;
            memset(rgba, 0, rgba_bytes);
            VkBuffer buffers[2] = {x_buffer, rgba_buffer};
            VkDeviceSize sizes[2] = {x_bytes, rgba_bytes};
            WorldVulkanTaehvPixelShufflePush push = {N, H, W};
            if (run_probe_compute(rt, "taehv_pixel_shuffle_rgba_f32.comp", 2, sizeof(push),
                        buffers, sizes, &push, (uint32_t)((4 * H * 2 * W * 2 + 255) / 256), 1, 1)) ok = 0;
            uint32_t mismatches = 0;
            for (int f = 0; ok && f < 4; ++f) {
                int n = N - 4 + f;
                for (int oy = 0; oy < H * 2; ++oy) {
                    for (int ox = 0; ox < W * 2; ++ox) {
                        uint32_t comps[3];
                        for (int c = 0; c < 3; ++c) {
                            int ic = c * 4 + (oy & 1) * 2 + (ox & 1);
                            float v = x[((n * 12 + ic) * H + oy / 2) * W + ox / 2];
                            if (v < 0.0f) v = 0.0f;
                            if (v > 1.0f) v = 1.0f;
                            comps[c] = (uint32_t)(v * 255.0f);
                        }
                        uint32_t ref = 0xff000000u | comps[0] | (comps[1] << 8) | (comps[2] << 16);
                        uint32_t got = rgba[(f * H * 2 + oy) * (W * 2) + ox];
                        if (got != ref) ++mismatches;
                    }
                }
            }
            fprintf(stderr, "vulkan taehv_pixel_shuffle_rgba_f32 probe: mismatches=%u\n", mismatches);
            if (mismatches != 0) ok = 0;
        }
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, rgba_buffer, rgba_memory, rgba_mapped);
        if (!ok) goto cleanup;
    }

    {
        enum { N = 6, C_IN = 5, C_OUT = 12, H = 3, W = 4, K = 3 };
        int ok = 1;
        VkBuffer x_buffer = VK_NULL_HANDLE, w_buffer = VK_NULL_HANDLE, b_buffer = VK_NULL_HANDLE;
        VkBuffer y_buffer = VK_NULL_HANDLE, rgba_ref_buffer = VK_NULL_HANDLE, rgba_fused_buffer = VK_NULL_HANDLE;
        VkDeviceMemory x_memory = VK_NULL_HANDLE, w_memory = VK_NULL_HANDLE, b_memory = VK_NULL_HANDLE;
        VkDeviceMemory y_memory = VK_NULL_HANDLE, rgba_ref_memory = VK_NULL_HANDLE, rgba_fused_memory = VK_NULL_HANDLE;
        void *x_mapped = NULL, *w_mapped = NULL, *b_mapped = NULL;
        void *y_mapped = NULL, *rgba_ref_mapped = NULL, *rgba_fused_mapped = NULL;
        size_t x_bytes = (size_t)N * C_IN * H * W * sizeof(float);
        size_t w_bytes = (size_t)C_OUT * C_IN * K * K * sizeof(float);
        size_t b_bytes = (size_t)C_OUT * sizeof(float);
        size_t y_bytes = (size_t)N * C_OUT * H * W * sizeof(float);
        size_t rgba_bytes = (size_t)4 * H * 2 * W * 2 * sizeof(uint32_t);
        if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, rgba_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &rgba_ref_buffer, &rgba_ref_memory, &rgba_ref_mapped)) ok = 0;
        if (ok && create_host_buffer(rt, rgba_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &rgba_fused_buffer, &rgba_fused_memory, &rgba_fused_mapped)) ok = 0;
        if (ok) {
            float *x = (float *)x_mapped;
            float *w = (float *)w_mapped;
            float *b = (float *)b_mapped;
            float *y = (float *)y_mapped;
            uint32_t *rgba_ref = (uint32_t *)rgba_ref_mapped;
            uint32_t *rgba_fused = (uint32_t *)rgba_fused_mapped;
            for (int i = 0; i < N * C_IN * H * W; ++i) x[i] = probe_value(i + 181, 0.041015625f);
            for (int i = 0; i < C_OUT * C_IN * K * K; ++i) w[i] = probe_value(i + 211, 0.01953125f);
            for (int i = 0; i < C_OUT; ++i) b[i] = probe_value(i + 241, 0.03125f) + 0.5f;
            memset(y, 0, y_bytes);
            memset(rgba_ref, 0, rgba_bytes);
            memset(rgba_fused, 0, rgba_bytes);

            VkBuffer conv_buffers[4] = {x_buffer, w_buffer, b_buffer, y_buffer};
            VkDeviceSize conv_sizes[4] = {x_bytes, w_bytes, b_bytes, y_bytes};
            WorldVulkanTaehvConv2dPush conv_push = {N, C_IN, C_OUT, H, W, K, 1};
            if (run_probe_compute(rt, "taehv_conv2d_nchw_f32.comp", 4, sizeof(conv_push),
                        conv_buffers, conv_sizes, &conv_push, (uint32_t)((N * C_OUT * H * W + 255) / 256), 1, 1)) ok = 0;

            VkBuffer shuffle_buffers[2] = {y_buffer, rgba_ref_buffer};
            VkDeviceSize shuffle_sizes[2] = {y_bytes, rgba_bytes};
            WorldVulkanTaehvPixelShufflePush shuffle_push = {N, H, W};
            if (ok && run_probe_compute(rt, "taehv_pixel_shuffle_rgba_f32.comp", 2, sizeof(shuffle_push),
                        shuffle_buffers, shuffle_sizes, &shuffle_push, (uint32_t)((4 * H * 2 * W * 2 + 255) / 256), 1, 1)) ok = 0;

            VkBuffer fused_buffers[4] = {x_buffer, w_buffer, b_buffer, rgba_fused_buffer};
            VkDeviceSize fused_sizes[4] = {x_bytes, w_bytes, b_bytes, rgba_bytes};
            if (ok && run_probe_compute(rt, "taehv_conv_out_rgba_f32.comp", 4, sizeof(conv_push),
                        fused_buffers, fused_sizes, &conv_push, (uint32_t)((4 * H * 2 * W * 2 + 127) / 128), 1, 1)) ok = 0;

            uint32_t mismatches = 0;
            for (size_t i = 0; ok && i < rgba_bytes / sizeof(uint32_t); ++i) {
                if (rgba_ref[i] != rgba_fused[i]) {
                    ++mismatches;
                }
            }
            fprintf(stderr, "vulkan taehv_conv_out_rgba_f32 probe: mismatches=%u\n", mismatches);
            if (mismatches != 0) ok = 0;
        }
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
        destroy_host_buffer(rt, rgba_ref_buffer, rgba_ref_memory, rgba_ref_mapped);
        destroy_host_buffer(rt, rgba_fused_buffer, rgba_fused_memory, rgba_fused_mapped);
        if (!ok) goto cleanup;
    }

    rc = 0;

cleanup:
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_rms_norm_f32_probe(void) {
    enum { rows = 5, cols = 257 };
    const float eps = 1.0e-6f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)rows * cols * sizeof(float);
    size_t w_bytes = (size_t)cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *w = (float *)w_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * cols; ++i) x[i] = probe_value(i + 71, 0.0625f);
    for (int i = 0; i < cols; ++i) w[i] = 1.0f + probe_value(i + 151, 0.01171875f);
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "rms_norm_f32.comp", 3, sizeof(WorldVulkanRmsNormPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, w_buffer, y_buffer};
    VkDeviceSize sizes[3] = {x_bytes, w_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanRmsNormPush push;
    push.rows = rows;
    push.cols = cols;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                rows, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sum += v * v;
        }
        float scale = 1.0f / sqrtf(sum / (float)cols + eps);
        for (int c = 0; c < cols; ++c) {
            float ref = x[r * cols + c] * scale * w[c];
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan rms_norm_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_control_embedding_f32_probe(void) {
    enum { ctrl_dim = 9, hidden = 23, D = 17 };
    const float eps = 1.0e-6f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule linear_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout linear_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout linear_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline linear_pipeline = VK_NULL_HANDLE;
    VkShaderModule silu_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout silu_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout silu_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline silu_pipeline = VK_NULL_HANDLE;
    VkShaderModule rms_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout rms_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout rms_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline rms_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool fc1_pool = VK_NULL_HANDLE;
    VkDescriptorPool silu_pool = VK_NULL_HANDLE;
    VkDescriptorPool fc2_pool = VK_NULL_HANDLE;
    VkDescriptorPool rms_pool = VK_NULL_HANDLE;
    VkDescriptorSet fc1_set = VK_NULL_HANDLE;
    VkDescriptorSet silu_set = VK_NULL_HANDLE;
    VkDescriptorSet fc2_set = VK_NULL_HANDLE;
    VkDescriptorSet rms_set = VK_NULL_HANDLE;
    VkBuffer control_buffer = VK_NULL_HANDLE;
    VkBuffer fc1_w_buffer = VK_NULL_HANDLE;
    VkBuffer fc2_w_buffer = VK_NULL_HANDLE;
    VkBuffer dummy_buffer = VK_NULL_HANDLE;
    VkBuffer hidden_buffer = VK_NULL_HANDLE;
    VkBuffer emb_buffer = VK_NULL_HANDLE;
    VkBuffer rms_w_buffer = VK_NULL_HANDLE;
    VkBuffer norm_buffer = VK_NULL_HANDLE;
    VkDeviceMemory control_memory = VK_NULL_HANDLE;
    VkDeviceMemory fc1_w_memory = VK_NULL_HANDLE;
    VkDeviceMemory fc2_w_memory = VK_NULL_HANDLE;
    VkDeviceMemory dummy_memory = VK_NULL_HANDLE;
    VkDeviceMemory hidden_memory = VK_NULL_HANDLE;
    VkDeviceMemory emb_memory = VK_NULL_HANDLE;
    VkDeviceMemory rms_w_memory = VK_NULL_HANDLE;
    VkDeviceMemory norm_memory = VK_NULL_HANDLE;
    void *control_mapped = NULL;
    void *fc1_w_mapped = NULL;
    void *fc2_w_mapped = NULL;
    void *dummy_mapped = NULL;
    void *hidden_mapped = NULL;
    void *emb_mapped = NULL;
    void *rms_w_mapped = NULL;
    void *norm_mapped = NULL;
    float *ref_hidden = NULL;
    float *ref_emb = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;

    size_t control_bytes = (size_t)ctrl_dim * sizeof(float);
    size_t hidden_bytes = (size_t)hidden * sizeof(float);
    size_t emb_bytes = (size_t)D * sizeof(float);
    size_t fc1_w_bytes = (size_t)hidden * ctrl_dim * sizeof(float);
    size_t fc2_w_bytes = (size_t)D * hidden * sizeof(float);
    if (create_host_buffer(rt, control_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &control_buffer, &control_memory, &control_mapped)) goto cleanup;
    if (create_host_buffer(rt, fc1_w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &fc1_w_buffer, &fc1_w_memory, &fc1_w_mapped)) goto cleanup;
    if (create_host_buffer(rt, fc2_w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &fc2_w_buffer, &fc2_w_memory, &fc2_w_mapped)) goto cleanup;
    if (create_host_buffer(rt, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &dummy_buffer, &dummy_memory, &dummy_mapped)) goto cleanup;
    if (create_host_buffer(rt, hidden_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &hidden_buffer, &hidden_memory, &hidden_mapped)) goto cleanup;
    if (create_host_buffer(rt, emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &emb_buffer, &emb_memory, &emb_mapped)) goto cleanup;
    if (create_host_buffer(rt, emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rms_w_buffer, &rms_w_memory, &rms_w_mapped)) goto cleanup;
    if (create_host_buffer(rt, emb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &norm_buffer, &norm_memory, &norm_mapped)) goto cleanup;

    float *control = (float *)control_mapped;
    float *fc1_w = (float *)fc1_w_mapped;
    float *fc2_w = (float *)fc2_w_mapped;
    float *dummy = (float *)dummy_mapped;
    float *hidden_y = (float *)hidden_mapped;
    float *emb_y = (float *)emb_mapped;
    float *rms_w = (float *)rms_w_mapped;
    float *norm_y = (float *)norm_mapped;
    for (int i = 0; i < ctrl_dim; ++i) control[i] = probe_value(i + 11, 0.125f);
    for (int i = 0; i < hidden * ctrl_dim; ++i) fc1_w[i] = probe_value(i + 101, 0.03125f);
    for (int i = 0; i < D * hidden; ++i) fc2_w[i] = probe_value(i + 503, 0.0234375f);
    for (int i = 0; i < D; ++i) rms_w[i] = 1.0f + probe_value(i + 907, 0.015625f);
    dummy[0] = 0.0f;
    memset(hidden_y, 0, hidden_bytes);
    memset(emb_y, 0, emb_bytes);
    memset(norm_y, 0, emb_bytes);

    ref_hidden = (float *)calloc((size_t)hidden, sizeof(float));
    ref_emb = (float *)calloc((size_t)D, sizeof(float));
    if (!ref_hidden || !ref_emb) goto cleanup;

    if (create_storage_pipeline(rt, "linear_f32.comp", 4, sizeof(WorldVulkanLinearPush),
                &linear_shader, &linear_set_layout, &linear_pipeline_layout, &linear_pipeline)) goto cleanup;
    if (create_storage_pipeline(rt, "silu_f32.comp", 2, sizeof(WorldVulkanSiluPush),
                &silu_shader, &silu_set_layout, &silu_pipeline_layout, &silu_pipeline)) goto cleanup;
    if (create_storage_pipeline(rt, "rms_norm_f32.comp", 3, sizeof(WorldVulkanRmsNormPush),
                &rms_shader, &rms_set_layout, &rms_pipeline_layout, &rms_pipeline)) goto cleanup;

    {
        VkBuffer buffers[4] = {control_buffer, fc1_w_buffer, dummy_buffer, hidden_buffer};
        VkDeviceSize sizes[4] = {control_bytes, fc1_w_bytes, sizeof(float), hidden_bytes};
        if (create_storage_descriptor_set(rt, linear_set_layout, 4, buffers, sizes, NULL, &fc1_pool, &fc1_set)) goto cleanup;
    }
    {
        VkBuffer buffers[2] = {hidden_buffer, hidden_buffer};
        VkDeviceSize sizes[2] = {hidden_bytes, hidden_bytes};
        if (create_storage_descriptor_set(rt, silu_set_layout, 2, buffers, sizes, NULL, &silu_pool, &silu_set)) goto cleanup;
    }
    {
        VkBuffer buffers[4] = {hidden_buffer, fc2_w_buffer, dummy_buffer, emb_buffer};
        VkDeviceSize sizes[4] = {hidden_bytes, fc2_w_bytes, sizeof(float), emb_bytes};
        if (create_storage_descriptor_set(rt, linear_set_layout, 4, buffers, sizes, NULL, &fc2_pool, &fc2_set)) goto cleanup;
    }
    {
        VkBuffer buffers[3] = {emb_buffer, rms_w_buffer, norm_buffer};
        VkDeviceSize sizes[3] = {emb_bytes, emb_bytes, emb_bytes};
        if (create_storage_descriptor_set(rt, rms_set_layout, 3, buffers, sizes, NULL, &rms_pool, &rms_set)) goto cleanup;
    }

    WorldVulkanLinearPush fc1_push;
    fc1_push.rows = 1;
    fc1_push.cols = hidden;
    fc1_push.inner = ctrl_dim;
    fc1_push.has_bias = 0;
    if (submit_compute(rt, linear_pipeline, linear_pipeline_layout, fc1_set, &fc1_push, sizeof(fc1_push),
                (hidden + 15) / 16, 1, 1)) goto cleanup;

    WorldVulkanSiluPush silu_push;
    silu_push.n = hidden;
    if (submit_compute(rt, silu_pipeline, silu_pipeline_layout, silu_set, &silu_push, sizeof(silu_push),
                (hidden + 255) / 256, 1, 1)) goto cleanup;

    WorldVulkanLinearPush fc2_push;
    fc2_push.rows = 1;
    fc2_push.cols = D;
    fc2_push.inner = hidden;
    fc2_push.has_bias = 0;
    if (submit_compute(rt, linear_pipeline, linear_pipeline_layout, fc2_set, &fc2_push, sizeof(fc2_push),
                (D + 15) / 16, 1, 1)) goto cleanup;

    WorldVulkanRmsNormPush rms_push;
    rms_push.rows = 1;
    rms_push.cols = D;
    rms_push.eps = eps;
    if (submit_compute(rt, rms_pipeline, rms_pipeline_layout, rms_set, &rms_push, sizeof(rms_push),
                1, 1, 1)) goto cleanup;

    for (int h = 0; h < hidden; ++h) {
        float acc = 0.0f;
        for (int k = 0; k < ctrl_dim; ++k) {
            acc += control[k] * fc1_w[h * ctrl_dim + k];
        }
        ref_hidden[h] = acc / (1.0f + expf(-acc));
    }
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            acc += ref_hidden[h] * fc2_w[d * hidden + h];
        }
        ref_emb[d] = acc;
        sum += acc * acc;
    }
    float scale = 1.0f / sqrtf(sum / (float)D + eps);
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int d = 0; d < D; ++d) {
        float ref = ref_emb[d] * scale * rms_w[d];
        float diff = fabsf(norm_y[d] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)D;
    fprintf(stderr, "vulkan control_embedding_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (linear_pipeline) vkDestroyPipeline(rt->device, linear_pipeline, NULL);
    if (silu_pipeline) vkDestroyPipeline(rt->device, silu_pipeline, NULL);
    if (rms_pipeline) vkDestroyPipeline(rt->device, rms_pipeline, NULL);
    if (linear_pipeline_layout) vkDestroyPipelineLayout(rt->device, linear_pipeline_layout, NULL);
    if (silu_pipeline_layout) vkDestroyPipelineLayout(rt->device, silu_pipeline_layout, NULL);
    if (rms_pipeline_layout) vkDestroyPipelineLayout(rt->device, rms_pipeline_layout, NULL);
    if (fc1_pool) vkDestroyDescriptorPool(rt->device, fc1_pool, NULL);
    if (silu_pool) vkDestroyDescriptorPool(rt->device, silu_pool, NULL);
    if (fc2_pool) vkDestroyDescriptorPool(rt->device, fc2_pool, NULL);
    if (rms_pool) vkDestroyDescriptorPool(rt->device, rms_pool, NULL);
    if (linear_set_layout) vkDestroyDescriptorSetLayout(rt->device, linear_set_layout, NULL);
    if (silu_set_layout) vkDestroyDescriptorSetLayout(rt->device, silu_set_layout, NULL);
    if (rms_set_layout) vkDestroyDescriptorSetLayout(rt->device, rms_set_layout, NULL);
    if (linear_shader) vkDestroyShaderModule(rt->device, linear_shader, NULL);
    if (silu_shader) vkDestroyShaderModule(rt->device, silu_shader, NULL);
    if (rms_shader) vkDestroyShaderModule(rt->device, rms_shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, control_buffer, control_memory, control_mapped);
        destroy_host_buffer(rt, fc1_w_buffer, fc1_w_memory, fc1_w_mapped);
        destroy_host_buffer(rt, fc2_w_buffer, fc2_w_memory, fc2_w_mapped);
        destroy_host_buffer(rt, dummy_buffer, dummy_memory, dummy_mapped);
        destroy_host_buffer(rt, hidden_buffer, hidden_memory, hidden_mapped);
        destroy_host_buffer(rt, emb_buffer, emb_memory, emb_mapped);
        destroy_host_buffer(rt, rms_w_buffer, rms_w_memory, rms_w_mapped);
        destroy_host_buffer(rt, norm_buffer, norm_memory, norm_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    free(ref_hidden);
    free(ref_emb);
    return rc;
}

int world_vulkan_denoise_out_norm_f32_probe(void) {
    enum { C = 2, D = 16, mlp_ratio = 2, hidden = D * mlp_ratio, n_buttons = 6, ctrl_dim = n_buttons + 3 };
    enum { layer_count = 2 };
    enum { steps_to_run = 2, total_passes = steps_to_run + 1 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = C;
    cfg.d_model = D;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.n_layers = 1;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;
    cfg.n_buttons = n_buttons;
    cfg.mlp_ratio = mlp_ratio;
    cfg.scheduler_sigmas[0] = 1.0f;
    cfg.scheduler_sigmas[1] = 0.35f;
    cfg.scheduler_sigmas[2] = 0.0f;
    cfg.scheduler_sigmas_count = 3;

    size_t patch_elems = (size_t)D * C;
    size_t ctrl_fc1_elems = (size_t)hidden * ctrl_dim;
    size_t ctrl_fc2_elems = (size_t)D * hidden;
    size_t denoise_fc1_elems = (size_t)hidden * 512u;
    size_t denoise_fc2_elems = (size_t)D * hidden;
    size_t out_norm_elems = (size_t)2 * D * D;
    size_t layer_bias_elems = (size_t)layer_count * D;
    size_t layer_proj_elems = (size_t)layer_count * 6 * D * D;
    float *patch = NULL;
    float *ctrl_fc1 = NULL;
    float *ctrl_fc2 = NULL;
    float *denoise_fc1 = NULL;
    float *denoise_fc2 = NULL;
    float *out_norm = NULL;
    float *layer_bias_storage = NULL;
    float *layer_proj_storage = NULL;
    float *unpatch = NULL;
    float *unpatch_bias = NULL;
    float *noise = NULL;
    float *hidden_ref = NULL;
    float *cond = NULL;
    float *cond_act = NULL;
    float *layer_cond_act = NULL;
    WorldLayerWeights layers[layer_count];
    WorldVulkanRuntime *rt = NULL;
    memset(layers, 0, sizeof(layers));

    patch = (float *)malloc(patch_elems * sizeof(float));
    ctrl_fc1 = (float *)malloc(ctrl_fc1_elems * sizeof(float));
    ctrl_fc2 = (float *)malloc(ctrl_fc2_elems * sizeof(float));
    denoise_fc1 = (float *)malloc(denoise_fc1_elems * sizeof(float));
    denoise_fc2 = (float *)malloc(denoise_fc2_elems * sizeof(float));
    out_norm = (float *)malloc(out_norm_elems * sizeof(float));
    layer_bias_storage = (float *)malloc(layer_bias_elems * sizeof(float));
    layer_proj_storage = (float *)malloc(layer_proj_elems * sizeof(float));
    unpatch = (float *)malloc(patch_elems * sizeof(float));
    unpatch_bias = (float *)malloc((size_t)C * sizeof(float));
    noise = (float *)malloc(512u * sizeof(float));
    hidden_ref = (float *)malloc((size_t)hidden * sizeof(float));
    cond = (float *)malloc((size_t)D * sizeof(float));
    cond_act = (float *)malloc((size_t)D * sizeof(float));
    layer_cond_act = (float *)malloc((size_t)D * sizeof(float));
    if (!patch || !ctrl_fc1 || !ctrl_fc2 || !denoise_fc1 || !denoise_fc2 || !out_norm ||
            !layer_bias_storage || !layer_proj_storage || !unpatch || !unpatch_bias || !noise ||
            !hidden_ref || !cond || !cond_act || !layer_cond_act) {
        goto cleanup;
    }

    for (size_t i = 0; i < patch_elems; ++i) {
        patch[i] = probe_value((int)i + 301, 0.015625f);
        unpatch[i] = probe_value((int)i + 409, 0.01953125f);
    }
    for (int i = 0; i < C; ++i) unpatch_bias[i] = probe_value(i + 503, 0.01171875f);
    for (size_t i = 0; i < ctrl_fc1_elems; ++i) ctrl_fc1[i] = probe_value((int)i + 601, 0.0078125f);
    for (size_t i = 0; i < ctrl_fc2_elems; ++i) ctrl_fc2[i] = probe_value((int)i + 701, 0.0068359375f);
    for (size_t i = 0; i < denoise_fc1_elems; ++i) denoise_fc1[i] = probe_value((int)i + 809, 0.005859375f);
    for (size_t i = 0; i < denoise_fc2_elems; ++i) denoise_fc2[i] = probe_value((int)i + 907, 0.0078125f);
    for (size_t i = 0; i < out_norm_elems; ++i) out_norm[i] = probe_value((int)i + 1009, 0.009765625f);
    for (size_t i = 0; i < layer_bias_elems; ++i) {
        layer_bias_storage[i] = probe_value((int)i + 1103, 0.03125f);
    }
    for (size_t i = 0; i < layer_proj_elems; ++i) {
        layer_proj_storage[i] = probe_value((int)i + 1201, 0.0048828125f);
    }
    for (int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
        float *base = layer_proj_storage + (size_t)layer_idx * 6 * D * D;
        layers[layer_idx].cond_bias = layer_bias_storage + (size_t)layer_idx * D;
        layers[layer_idx].attn_cond_s_weight = base + 0 * D * D;
        layers[layer_idx].attn_cond_b_weight = base + 1 * D * D;
        layers[layer_idx].attn_cond_g_weight = base + 2 * D * D;
        layers[layer_idx].mlp_cond_s_weight = base + 3 * D * D;
        layers[layer_idx].mlp_cond_b_weight = base + 4 * D * D;
        layers[layer_idx].mlp_cond_g_weight = base + 5 * D * D;
    }

    WorldModelProbeWeights weights;
    memset(&weights, 0, sizeof(weights));
    weights.patchify_weight = patch;
    weights.denoise_fc1_weight = denoise_fc1;
    weights.denoise_fc2_weight = denoise_fc2;
    weights.ctrl_emb_fc1_weight = ctrl_fc1;
    weights.ctrl_emb_fc2_weight = ctrl_fc2;
    weights.layers = layers;
    weights.n_layers = layer_count;
    weights.out_norm_fc_weight = out_norm;
    weights.unpatchify_weight = unpatch;
    weights.unpatchify_bias = unpatch_bias;

    if (world_vulkan_runtime_create(&rt, &cfg, &weights, layer_count, steps_to_run, 0, 1234, WORLD_NOISE_NORMAL, NULL)) {
        goto cleanup;
    }

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float layer_max_abs = 0.0f;
    float layer_mean_abs = 0.0f;
    const float *got = (const float *)rt->out_mod_table_mapped;
    const float *layer_got = (const float *)rt->layer_mod_table_mapped;
    for (int pass_idx = 0; pass_idx < total_passes; ++pass_idx) {
        int is_cache_pass = pass_idx >= steps_to_run;
        float sigma = is_cache_pass ? 0.0f : cfg.scheduler_sigmas[pass_idx];
        fill_noise_embedding(noise, sigma);
        for (int h = 0; h < hidden; ++h) {
            float acc = 0.0f;
            for (int k = 0; k < 512; ++k) {
                acc += noise[k] * denoise_fc1[h * 512 + k];
            }
            hidden_ref[h] = acc / (1.0f + expf(-acc));
        }
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int h = 0; h < hidden; ++h) {
                acc += hidden_ref[h] * denoise_fc2[d * hidden + h];
            }
            cond[d] = acc;
            cond_act[d] = acc / (1.0f + expf(-acc));
        }
        for (int o = 0; o < 2 * D; ++o) {
            float ref = 0.0f;
            for (int d = 0; d < D; ++d) {
                ref += cond_act[d] * out_norm[o * D + d];
            }
            float diff = fabsf(got[pass_idx * 2 * D + o] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
        for (int layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
            const float *bias = layer_bias_storage + (size_t)layer_idx * D;
            const float *proj = layer_proj_storage + (size_t)layer_idx * 6 * D * D;
            for (int d = 0; d < D; ++d) {
                float v = cond[d] + bias[d];
                layer_cond_act[d] = v / (1.0f + expf(-v));
            }
            for (int o = 0; o < 6 * D; ++o) {
                float ref = 0.0f;
                for (int d = 0; d < D; ++d) {
                    ref += layer_cond_act[d] * proj[(size_t)o * D + d];
                }
                size_t got_idx = ((size_t)pass_idx * layer_count + layer_idx) * 6 * D + o;
                float diff = fabsf(layer_got[got_idx] - ref);
                if (diff > layer_max_abs) layer_max_abs = diff;
                layer_mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(total_passes * 2 * D);
    layer_mean_abs /= (float)(total_passes * layer_count * 6 * D);
    fprintf(stderr, "vulkan denoise_out_norm_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    fprintf(stderr, "vulkan layer_mod_f32 probe: max_abs=%g mean_abs=%g\n", layer_max_abs, layer_mean_abs);
    if (max_abs > 8.0e-5f) goto cleanup;
    if (layer_max_abs > 1.0e-4f) goto cleanup;
    rc = 0;

cleanup:
    world_vulkan_runtime_destroy(rt);
    free(patch);
    free(ctrl_fc1);
    free(ctrl_fc2);
    free(denoise_fc1);
    free(denoise_fc2);
    free(out_norm);
    free(layer_bias_storage);
    free(layer_proj_storage);
    free(unpatch);
    free(unpatch_bias);
    free(noise);
    free(hidden_ref);
    free(cond);
    free(cond_act);
    free(layer_cond_act);
    return rc;
}

int world_vulkan_ada_rms_norm_f32_probe(void) {
    enum { B = 2, N = 3, M = 5, T = N * M, D = 257 };
    const float eps = 1.0e-6f;
    const int rows = B * T;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer scale_buffer = VK_NULL_HANDLE;
    VkBuffer bias_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory scale_memory = VK_NULL_HANDLE;
    VkDeviceMemory bias_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *scale_mapped = NULL;
    void *bias_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)rows * D * sizeof(float);
    size_t sb_bytes = (size_t)B * N * D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &scale_buffer, &scale_memory, &scale_mapped)) goto cleanup;
    if (create_host_buffer(rt, sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bias_buffer, &bias_memory, &bias_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *scale = (float *)scale_mapped;
    float *bias = (float *)bias_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * D; ++i) x[i] = probe_value(i + 211, 0.046875f);
    for (int i = 0; i < B * N * D; ++i) {
        scale[i] = probe_value(i + 307, 0.0078125f);
        bias[i] = probe_value(i + 409, 0.01171875f);
    }
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "ada_rms_norm_f32.comp", 4, sizeof(WorldVulkanAdaRmsNormPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[4] = {x_buffer, scale_buffer, bias_buffer, y_buffer};
    VkDeviceSize sizes[4] = {x_bytes, sb_bytes, sb_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 4, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanAdaRmsNormPush push;
    push.B = B;
    push.T = T;
    push.N = N;
    push.D = D;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                rows, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int row = b * T + t;
            int n = t / M;
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v = x[row * D + d];
                sum += v * v;
            }
            float inv = 1.0f / sqrtf(sum / (float)D + eps);
            for (int d = 0; d < D; ++d) {
                int idx = row * D + d;
                int sb = (b * N + n) * D + d;
                float ref = x[idx] * inv * (1.0f + scale[sb]) + bias[sb];
                float diff = fabsf(y[idx] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(rows * D);
    fprintf(stderr, "vulkan ada_rms_norm_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, scale_buffer, scale_memory, scale_mapped);
        destroy_host_buffer(rt, bias_buffer, bias_memory, bias_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_ortho_rope_f32_probe(void) {
    enum { B = 2, H = 3, T = 11, D = 128, width = 20, height = 18 };
    enum { half_d = D / 2, d_xy = D / 8, d_t = D / 4 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer x_pos_buffer = VK_NULL_HANDLE;
    VkBuffer y_pos_buffer = VK_NULL_HANDLE;
    VkBuffer t_pos_buffer = VK_NULL_HANDLE;
    VkBuffer xy_buffer = VK_NULL_HANDLE;
    VkBuffer inv_t_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory x_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory t_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory xy_memory = VK_NULL_HANDLE;
    VkDeviceMemory inv_t_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *x_pos_mapped = NULL;
    void *y_pos_mapped = NULL;
    void *t_pos_mapped = NULL;
    void *xy_mapped = NULL;
    void *inv_t_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)B * H * T * D * sizeof(float);
    size_t pos_bytes = (size_t)T * sizeof(uint32_t);
    size_t xy_bytes = (size_t)d_xy * sizeof(float);
    size_t inv_t_bytes = (size_t)d_t * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_pos_buffer, &x_pos_memory, &x_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_pos_buffer, &y_pos_memory, &y_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &t_pos_buffer, &t_pos_memory, &t_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, xy_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &xy_buffer, &xy_memory, &xy_mapped)) goto cleanup;
    if (create_host_buffer(rt, inv_t_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &inv_t_buffer, &inv_t_memory, &inv_t_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    uint32_t *x_pos = (uint32_t *)x_pos_mapped;
    uint32_t *y_pos = (uint32_t *)y_pos_mapped;
    uint32_t *t_pos = (uint32_t *)t_pos_mapped;
    float *xy = (float *)xy_mapped;
    float *inv_t = (float *)inv_t_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < B * H * T * D; ++i) x[i] = probe_value(i + 503, 0.03125f);
    for (int t = 0; t < T; ++t) {
        y_pos[t] = (uint32_t)(t % height);
        x_pos[t] = (uint32_t)((t * 3) % width);
        t_pos[t] = (uint32_t)(t * 7);
    }
    fill_probe_rope_tables(xy, inv_t, D, height, width);
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "ortho_rope_f32.comp", 7, sizeof(WorldVulkanOrthoRopePush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[7] = {
        x_buffer, x_pos_buffer, y_pos_buffer, t_pos_buffer, xy_buffer, inv_t_buffer, y_buffer
    };
    VkDeviceSize sizes[7] = {
        x_bytes, pos_bytes, pos_bytes, pos_bytes, xy_bytes, inv_t_bytes, x_bytes
    };
    if (create_storage_descriptor_set(rt, set_layout, 7, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanOrthoRopePush push;
    push.B = B;
    push.H = H;
    push.T = T;
    push.D = D;
    push.width = width;
    push.height = height;
    uint32_t total = (uint32_t)(B * H * T * half_d);
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (total + 255u) / 256u, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                size_t base = (size_t)(((b * H + h) * T + t) * D);
                for (int p = 0; p < half_d; ++p) {
                    float phase = probe_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, D, width, height);
                    float c = cosf(phase);
                    float s = sinf(phase);
                    float x0 = x[base + 2 * p];
                    float x1 = x[base + 2 * p + 1];
                    float ref0 = x0 * c - x1 * s;
                    float ref1 = x1 * c + x0 * s;
                    float diff0 = fabsf(y[base + p] - ref0);
                    float diff1 = fabsf(y[base + half_d + p] - ref1);
                    if (diff0 > max_abs) max_abs = diff0;
                    if (diff1 > max_abs) max_abs = diff1;
                    mean_abs += diff0 + diff1;
                }
            }
        }
    }
    mean_abs /= (float)(B * H * T * D);
    fprintf(stderr, "vulkan ortho_rope_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 5.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, x_pos_buffer, x_pos_memory, x_pos_mapped);
        destroy_host_buffer(rt, y_pos_buffer, y_pos_memory, y_pos_mapped);
        destroy_host_buffer(rt, t_pos_buffer, t_pos_memory, t_pos_mapped);
        destroy_host_buffer(rt, xy_buffer, xy_memory, xy_mapped);
        destroy_host_buffer(rt, inv_t_buffer, inv_t_memory, inv_t_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_qkv_rms_rope_f32_probe(void) {
    enum { B = 2, T = 13, n_heads = 6, n_kv_heads = 2, D = 128, width = 19, height = 16 };
    enum { total_heads = n_heads + 2 * n_kv_heads, half_d = D / 2, d_xy = D / 8, d_t = D / 4 };
    const float eps = 1.0e-6f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer qkv_buffer = VK_NULL_HANDLE;
    VkBuffer x_pos_buffer = VK_NULL_HANDLE;
    VkBuffer y_pos_buffer = VK_NULL_HANDLE;
    VkBuffer t_pos_buffer = VK_NULL_HANDLE;
    VkBuffer xy_buffer = VK_NULL_HANDLE;
    VkBuffer inv_t_buffer = VK_NULL_HANDLE;
    VkBuffer q_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkDeviceMemory qkv_memory = VK_NULL_HANDLE;
    VkDeviceMemory x_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory t_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory xy_memory = VK_NULL_HANDLE;
    VkDeviceMemory inv_t_memory = VK_NULL_HANDLE;
    VkDeviceMemory q_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    void *qkv_mapped = NULL;
    void *x_pos_mapped = NULL;
    void *y_pos_mapped = NULL;
    void *t_pos_mapped = NULL;
    void *xy_mapped = NULL;
    void *inv_t_mapped = NULL;
    void *q_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t qkv_bytes = (size_t)B * T * total_heads * D * sizeof(float);
    size_t q_bytes = (size_t)B * n_heads * T * D * sizeof(float);
    size_t kv_bytes = (size_t)B * n_kv_heads * T * D * sizeof(float);
    size_t pos_bytes = (size_t)T * sizeof(uint32_t);
    size_t xy_bytes = (size_t)d_xy * sizeof(float);
    size_t inv_t_bytes = (size_t)d_t * sizeof(float);
    if (create_host_buffer(rt, qkv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &qkv_buffer, &qkv_memory, &qkv_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_pos_buffer, &x_pos_memory, &x_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_pos_buffer, &y_pos_memory, &y_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &t_pos_buffer, &t_pos_memory, &t_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, xy_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &xy_buffer, &xy_memory, &xy_mapped)) goto cleanup;
    if (create_host_buffer(rt, inv_t_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &inv_t_buffer, &inv_t_memory, &inv_t_mapped)) goto cleanup;
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &q_buffer, &q_memory, &q_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v_buffer, &v_memory, &v_mapped)) goto cleanup;

    float *qkv = (float *)qkv_mapped;
    uint32_t *x_pos = (uint32_t *)x_pos_mapped;
    uint32_t *y_pos = (uint32_t *)y_pos_mapped;
    uint32_t *t_pos = (uint32_t *)t_pos_mapped;
    float *xy = (float *)xy_mapped;
    float *inv_t = (float *)inv_t_mapped;
    float *q = (float *)q_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    for (int i = 0; i < B * T * total_heads * D; ++i) qkv[i] = probe_value(i + 601, 0.02734375f);
    for (int t = 0; t < T; ++t) {
        y_pos[t] = (uint32_t)(t % height);
        x_pos[t] = (uint32_t)((t * 5) % width);
        t_pos[t] = (uint32_t)(t * 2);
    }
    fill_probe_rope_tables(xy, inv_t, D, height, width);
    memset(q, 0, q_bytes);
    memset(k, 0, kv_bytes);
    memset(v, 0, kv_bytes);

    if (create_storage_pipeline(rt, "qkv_rms_rope_f32.comp", 9, sizeof(WorldVulkanQkvRmsRopePush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[9] = {
        qkv_buffer, x_pos_buffer, y_pos_buffer, t_pos_buffer, xy_buffer, inv_t_buffer, q_buffer, k_buffer, v_buffer
    };
    VkDeviceSize sizes[9] = {
        qkv_bytes, pos_bytes, pos_bytes, pos_bytes, xy_bytes, inv_t_bytes, q_bytes, kv_bytes, kv_bytes
    };
    if (create_storage_descriptor_set(rt, set_layout, 9, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanQkvRmsRopePush push;
    push.B = B;
    push.T = T;
    push.n_heads = n_heads;
    push.n_kv_heads = n_kv_heads;
    push.D = D;
    push.width = width;
    push.height = height;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * T, total_heads, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int compared = 0;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int role = 0; role < n_heads + n_kv_heads; ++role) {
                int is_k = role >= n_heads;
                int h = is_k ? role - n_heads : role;
                int src_head = is_k ? n_heads + h : h;
                const float *src = qkv + ((b * T + t) * total_heads + src_head) * D;
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float z = src[d];
                    sum += z * z;
                }
                float inv = 1.0f / sqrtf(sum / (float)D + eps);
                float *dst = is_k
                    ? k + ((b * n_kv_heads + h) * T + t) * D
                    : q + ((b * n_heads + h) * T + t) * D;
                for (int p = 0; p < half_d; ++p) {
                    float phase = probe_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, D, width, height);
                    float c = cosf(phase);
                    float s = sinf(phase);
                    float a = src[2 * p] * inv;
                    float bb = src[2 * p + 1] * inv;
                    float ref0 = a * c - bb * s;
                    float ref1 = bb * c + a * s;
                    float diff0 = fabsf(dst[p] - ref0);
                    float diff1 = fabsf(dst[half_d + p] - ref1);
                    if (diff0 > max_abs) max_abs = diff0;
                    if (diff1 > max_abs) max_abs = diff1;
                    mean_abs += diff0 + diff1;
                    compared += 2;
                }
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                const float *src = qkv + ((b * T + t) * total_heads + n_heads + n_kv_heads + h) * D;
                const float *dst = v + ((b * n_kv_heads + h) * T + t) * D;
                for (int d = 0; d < D; ++d) {
                    float diff = fabsf(dst[d] - src[d]);
                    if (diff > max_abs) max_abs = diff;
                    mean_abs += diff;
                    compared += 1;
                }
            }
        }
    }
    mean_abs /= (float)compared;
    fprintf(stderr, "vulkan qkv_rms_rope_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 5.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, qkv_buffer, qkv_memory, qkv_mapped);
        destroy_host_buffer(rt, x_pos_buffer, x_pos_memory, x_pos_mapped);
        destroy_host_buffer(rt, y_pos_buffer, y_pos_memory, y_pos_mapped);
        destroy_host_buffer(rt, t_pos_buffer, t_pos_memory, t_pos_mapped);
        destroy_host_buffer(rt, xy_buffer, xy_memory, xy_mapped);
        destroy_host_buffer(rt, inv_t_buffer, inv_t_memory, inv_t_mapped);
        destroy_host_buffer(rt, q_buffer, q_memory, q_mapped);
        destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
        destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_runtime_layer0_qkv_f32_probe(void) {
    enum {
        C = 2,
        D = 128,
        n_heads = 4,
        n_kv_heads = 2,
        d_head = D / n_heads,
        kv_dim = n_kv_heads * d_head,
        qkv_dim = D + 2 * kv_dim,
        height = 2,
        width = 2,
        T = height * width,
        mlp_ratio = 2,
        hidden = D * mlp_ratio,
        n_buttons = 5,
        ctrl_dim = n_buttons + 3
    };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = C;
    cfg.d_model = D;
    cfg.n_heads = n_heads;
    cfg.n_kv_heads = n_kv_heads;
    cfg.n_layers = 1;
    cfg.height = height;
    cfg.width = width;
    cfg.patch_h = 1;
    cfg.patch_w = 1;
    cfg.n_buttons = n_buttons;
    cfg.mlp_ratio = mlp_ratio;
    cfg.base_fps = 15;
    cfg.inference_fps = 60;
    cfg.local_window = 2;
    cfg.global_window = 2;
    cfg.global_pinned_dilation = 1;
    cfg.global_attn_period = 4;
    cfg.global_attn_offset = -1;
    cfg.scheduler_sigmas[0] = 1.0f;
    cfg.scheduler_sigmas[1] = 0.0f;
    cfg.scheduler_sigmas_count = 2;

    size_t latent_elems = (size_t)C * height * width;
    size_t patch_elems = (size_t)D * C;
    size_t ctrl_fc1_elems = (size_t)hidden * ctrl_dim;
    size_t ctrl_fc2_elems = (size_t)D * hidden;
    size_t denoise_fc1_elems = (size_t)hidden * 512u;
    size_t denoise_fc2_elems = (size_t)D * hidden;
    size_t out_norm_elems = (size_t)2 * D * D;
    size_t cond_proj_elems = (size_t)6 * D * D;
    size_t q_elems = (size_t)D * D;
    size_t kv_elems = (size_t)kv_dim * D;
    size_t dit_mlp_fc1_elems = (size_t)hidden * D;
    size_t dit_mlp_fc2_elems = (size_t)D * hidden;
    size_t mlp_hidden_token_elems = (size_t)T * hidden;
    float *latent = NULL;
    float *control = NULL;
    float *patch = NULL;
    float *ctrl_fc1 = NULL;
    float *ctrl_fc2 = NULL;
    float *denoise_fc1 = NULL;
    float *denoise_fc2 = NULL;
    float *out_norm = NULL;
    float *unpatch = NULL;
    float *unpatch_bias = NULL;
    float *cond_bias = NULL;
    float *cond_proj = NULL;
    float *q_w = NULL;
    float *k_w = NULL;
    float *v_w = NULL;
    float *out_w = NULL;
    float *ctrl_fc1_c_w = NULL;
    float *ctrl_fc1_x_w = NULL;
    float *ctrl_fc2_w = NULL;
    float *dit_mlp_fc1_w = NULL;
    float *dit_mlp_fc2_w = NULL;
    float *tokens_ref = NULL;
    float *norm_ref = NULL;
    float *qkv_ref = NULL;
    float *attn_proj_ref = NULL;
    float *tokens_after_attn_ref = NULL;
    float *ctrl_cond_ref = NULL;
    float *ctrl_norm_ref = NULL;
    float *ctrl_hidden_ref = NULL;
    float *ctrl_out_ref = NULL;
    float *tokens_after_ctrl_ref = NULL;
    float *mlp_in_ref = NULL;
    float *mlp_hidden_ref = NULL;
    float *mlp_out_ref = NULL;
    float *tokens_after_mlp_ref = NULL;
    float *final_tokens_ref = NULL;
    float *latent_out_ref = NULL;
    float *latent_final_ref = NULL;
    float *norm_got = NULL;
    float *qkv_got = NULL;
    float *q_got = NULL;
    float *k_got = NULL;
    float *v_got = NULL;
    float *cache_k = NULL;
    float *cache_v = NULL;
    uint32_t *written = NULL;
    uint32_t *indices = NULL;
    uint32_t *cache_index_count = NULL;
    float *attn = NULL;
    float *attn_proj = NULL;
    float *tokens_after_attn = NULL;
    float *ctrl_norm = NULL;
    float *ctrl_hidden = NULL;
    float *ctrl_out = NULL;
    float *tokens_after_ctrl = NULL;
    float *mlp_in = NULL;
    float *mlp_hidden = NULL;
    float *mlp_out = NULL;
    float *tokens_after_mlp = NULL;
    float *final_tokens = NULL;
    float *latent_out = NULL;
    WorldVulkanRuntime *rt = NULL;
    WorldLayerWeights layer;
    memset(&layer, 0, sizeof(layer));

    latent = (float *)malloc(latent_elems * sizeof(float));
    control = (float *)malloc((size_t)ctrl_dim * sizeof(float));
    patch = (float *)malloc(patch_elems * sizeof(float));
    ctrl_fc1 = (float *)malloc(ctrl_fc1_elems * sizeof(float));
    ctrl_fc2 = (float *)malloc(ctrl_fc2_elems * sizeof(float));
    denoise_fc1 = (float *)malloc(denoise_fc1_elems * sizeof(float));
    denoise_fc2 = (float *)malloc(denoise_fc2_elems * sizeof(float));
    out_norm = (float *)malloc(out_norm_elems * sizeof(float));
    unpatch = (float *)malloc(patch_elems * sizeof(float));
    unpatch_bias = (float *)malloc((size_t)C * sizeof(float));
    cond_bias = (float *)malloc((size_t)D * sizeof(float));
    cond_proj = (float *)malloc(cond_proj_elems * sizeof(float));
    q_w = (float *)malloc(q_elems * sizeof(float));
    k_w = (float *)malloc(kv_elems * sizeof(float));
    v_w = (float *)malloc(kv_elems * sizeof(float));
    out_w = (float *)malloc(q_elems * sizeof(float));
    ctrl_fc1_c_w = (float *)malloc(q_elems * sizeof(float));
    ctrl_fc1_x_w = (float *)malloc(q_elems * sizeof(float));
    ctrl_fc2_w = (float *)malloc(q_elems * sizeof(float));
    dit_mlp_fc1_w = (float *)malloc(dit_mlp_fc1_elems * sizeof(float));
    dit_mlp_fc2_w = (float *)malloc(dit_mlp_fc2_elems * sizeof(float));
    tokens_ref = (float *)malloc((size_t)T * D * sizeof(float));
    norm_ref = (float *)malloc((size_t)T * D * sizeof(float));
    qkv_ref = (float *)malloc((size_t)T * qkv_dim * sizeof(float));
    attn_proj_ref = (float *)malloc((size_t)T * D * sizeof(float));
    tokens_after_attn_ref = (float *)malloc((size_t)T * D * sizeof(float));
    ctrl_cond_ref = (float *)malloc((size_t)D * sizeof(float));
    ctrl_norm_ref = (float *)malloc((size_t)T * D * sizeof(float));
    ctrl_hidden_ref = (float *)malloc((size_t)T * D * sizeof(float));
    ctrl_out_ref = (float *)malloc((size_t)T * D * sizeof(float));
    tokens_after_ctrl_ref = (float *)malloc((size_t)T * D * sizeof(float));
    mlp_in_ref = (float *)malloc((size_t)T * D * sizeof(float));
    mlp_hidden_ref = (float *)malloc(mlp_hidden_token_elems * sizeof(float));
    mlp_out_ref = (float *)malloc((size_t)T * D * sizeof(float));
    tokens_after_mlp_ref = (float *)malloc((size_t)T * D * sizeof(float));
    final_tokens_ref = (float *)malloc((size_t)T * D * sizeof(float));
    latent_out_ref = (float *)malloc(latent_elems * sizeof(float));
    latent_final_ref = (float *)malloc(latent_elems * sizeof(float));
    if (!latent || !control || !patch || !ctrl_fc1 || !ctrl_fc2 || !denoise_fc1 ||
            !denoise_fc2 || !out_norm || !unpatch || !unpatch_bias || !cond_bias ||
            !cond_proj || !q_w || !k_w || !v_w || !out_w || !ctrl_fc1_c_w ||
            !ctrl_fc1_x_w || !ctrl_fc2_w || !dit_mlp_fc1_w || !dit_mlp_fc2_w ||
            !tokens_ref || !norm_ref || !qkv_ref ||
            !attn_proj_ref || !tokens_after_attn_ref || !ctrl_cond_ref || !ctrl_norm_ref ||
            !ctrl_hidden_ref || !ctrl_out_ref || !tokens_after_ctrl_ref ||
            !mlp_in_ref || !mlp_hidden_ref || !mlp_out_ref || !tokens_after_mlp_ref ||
            !final_tokens_ref || !latent_out_ref || !latent_final_ref) {
        goto cleanup;
    }

    for (size_t i = 0; i < latent_elems; ++i) latent[i] = probe_value((int)i + 31, 0.25f);
    for (int i = 0; i < ctrl_dim; ++i) control[i] = probe_value(i + 43, 0.0625f);
    for (size_t i = 0; i < patch_elems; ++i) {
        patch[i] = probe_value((int)i + 101, 0.0234375f);
        unpatch[i] = probe_value((int)i + 151, 0.017578125f);
    }
    for (int i = 0; i < C; ++i) unpatch_bias[i] = probe_value(i + 191, 0.01171875f);
    for (size_t i = 0; i < ctrl_fc1_elems; ++i) ctrl_fc1[i] = probe_value((int)i + 211, 0.001953125f);
    for (size_t i = 0; i < ctrl_fc2_elems; ++i) ctrl_fc2[i] = probe_value((int)i + 307, 0.001953125f);
    for (size_t i = 0; i < denoise_fc1_elems; ++i) denoise_fc1[i] = probe_value((int)i + 401, 0.001953125f);
    for (size_t i = 0; i < denoise_fc2_elems; ++i) denoise_fc2[i] = probe_value((int)i + 503, 0.001953125f);
    for (size_t i = 0; i < out_norm_elems; ++i) out_norm[i] = probe_value((int)i + 607, 0.00146484375f);
    for (int i = 0; i < D; ++i) cond_bias[i] = probe_value(i + 701, 0.015625f);
    for (size_t i = 0; i < cond_proj_elems; ++i) cond_proj[i] = probe_value((int)i + 809, 0.0009765625f);
    for (size_t i = 0; i < q_elems; ++i) q_w[i] = probe_value((int)i + 907, 0.0029296875f);
    for (size_t i = 0; i < q_elems; ++i) out_w[i] = probe_value((int)i + 1201, 0.00244140625f);
    for (size_t i = 0; i < q_elems; ++i) {
        ctrl_fc1_c_w[i] = probe_value((int)i + 1301, 0.001953125f);
        ctrl_fc1_x_w[i] = probe_value((int)i + 1409, 0.00244140625f);
        ctrl_fc2_w[i] = probe_value((int)i + 1511, 0.00244140625f);
    }
    for (size_t i = 0; i < dit_mlp_fc1_elems; ++i) {
        dit_mlp_fc1_w[i] = probe_value((int)i + 1601, 0.001953125f);
    }
    for (size_t i = 0; i < dit_mlp_fc2_elems; ++i) {
        dit_mlp_fc2_w[i] = probe_value((int)i + 1709, 0.001708984375f);
    }
    for (size_t i = 0; i < kv_elems; ++i) {
        k_w[i] = probe_value((int)i + 1009, 0.0029296875f);
        v_w[i] = probe_value((int)i + 1103, 0.00390625f);
    }

    layer.cond_bias = cond_bias;
    layer.attn_cond_s_weight = cond_proj + 0 * D * D;
    layer.attn_cond_b_weight = cond_proj + 1 * D * D;
    layer.attn_cond_g_weight = cond_proj + 2 * D * D;
    layer.mlp_cond_s_weight = cond_proj + 3 * D * D;
    layer.mlp_cond_b_weight = cond_proj + 4 * D * D;
    layer.mlp_cond_g_weight = cond_proj + 5 * D * D;
    layer.q_proj_weight = q_w;
    layer.k_proj_weight = k_w;
    layer.v_proj_weight = v_w;
    layer.out_proj_weight = out_w;
    layer.ctrl_fc1_c_weight = ctrl_fc1_c_w;
    layer.ctrl_fc1_x_weight = ctrl_fc1_x_w;
    layer.ctrl_fc2_weight = ctrl_fc2_w;
    layer.dit_mlp_fc1_weight = dit_mlp_fc1_w;
    layer.dit_mlp_fc2_weight = dit_mlp_fc2_w;
    layer.has_ctrl = 1;

    WorldModelProbeWeights weights;
    memset(&weights, 0, sizeof(weights));
    weights.patchify_weight = patch;
    weights.denoise_fc1_weight = denoise_fc1;
    weights.denoise_fc2_weight = denoise_fc2;
    weights.ctrl_emb_fc1_weight = ctrl_fc1;
    weights.ctrl_emb_fc2_weight = ctrl_fc2;
    weights.layers = &layer;
    weights.n_layers = 1;
    weights.out_norm_fc_weight = out_norm;
    weights.unpatchify_weight = unpatch;
    weights.unpatchify_bias = unpatch_bias;

    if (world_vulkan_runtime_create(&rt, &cfg, &weights, 1, 1, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (!rt->layer_qkv_enabled) goto cleanup;
    if (!rt->layer_attention_enabled) goto cleanup;
    rt->total_passes = 1;
    {
        const unsigned char *rgb = NULL;
        int rgb_w = 0, rgb_h = 0, rgb_frames = 0;
        if (rt->latent_mapped) {
            memcpy(rt->latent_mapped, latent, latent_elems * sizeof(float));
            rt->use_external_latent_once = 1;
        }
        if (world_vulkan_runtime_step_rgb(rt, control, &rgb, &rgb_w, &rgb_h, &rgb_frames, NULL)) {
            goto cleanup;
        }
    }

#define READBACK_ALLOC(ptr, count, type) do { \
    (ptr) = (type *)malloc((size_t)(count) * sizeof(type)); \
    if (!(ptr)) goto cleanup; \
} while (0)
#define READBACK_BUFFER(ptr, buffer, bytes) do { \
    if (download_from_device_buffer(rt, (buffer), (ptr), (VkDeviceSize)(bytes))) goto cleanup; \
} while (0)
    {
        size_t token_bytes = (size_t)T * D * sizeof(float);
        size_t qkv_bytes = (size_t)T * qkv_dim * sizeof(float);
        size_t kv_rope_bytes = (size_t)T * kv_dim * sizeof(float);
        size_t cache_kv_bytes = rt->cache_total_kv_elems * sizeof(float);
        size_t cache_meta_bytes = rt->cache_total_meta_elems * sizeof(uint32_t);
        size_t cache_count_bytes = (size_t)rt->layers_to_run * sizeof(uint32_t);
        READBACK_ALLOC(norm_got, (size_t)T * D, float);
        READBACK_ALLOC(qkv_got, (size_t)T * qkv_dim, float);
        READBACK_ALLOC(q_got, (size_t)T * D, float);
        READBACK_ALLOC(k_got, (size_t)T * kv_dim, float);
        READBACK_ALLOC(v_got, (size_t)T * kv_dim, float);
        READBACK_ALLOC(cache_k, rt->cache_total_kv_elems, float);
        READBACK_ALLOC(cache_v, rt->cache_total_kv_elems, float);
        READBACK_ALLOC(written, rt->cache_total_meta_elems, uint32_t);
        READBACK_ALLOC(indices, rt->cache_total_meta_elems, uint32_t);
        READBACK_ALLOC(cache_index_count, rt->layers_to_run, uint32_t);
        READBACK_ALLOC(attn, (size_t)T * D, float);
        READBACK_ALLOC(attn_proj, (size_t)T * D, float);
        READBACK_ALLOC(tokens_after_attn, (size_t)T * D, float);
        READBACK_ALLOC(ctrl_norm, (size_t)T * D, float);
        READBACK_ALLOC(ctrl_hidden, (size_t)T * D, float);
        READBACK_ALLOC(ctrl_out, (size_t)T * D, float);
        READBACK_ALLOC(tokens_after_ctrl, (size_t)T * D, float);
        READBACK_ALLOC(mlp_in, (size_t)T * D, float);
        READBACK_ALLOC(mlp_hidden, mlp_hidden_token_elems, float);
        READBACK_ALLOC(mlp_out, (size_t)T * D, float);
        READBACK_ALLOC(tokens_after_mlp, (size_t)T * D, float);
        READBACK_ALLOC(final_tokens, (size_t)T * D, float);
        READBACK_ALLOC(latent_out, latent_elems, float);
        READBACK_BUFFER(norm_got, rt->norm_buffer, token_bytes);
        READBACK_BUFFER(qkv_got, rt->qkv_raw_buffer, qkv_bytes);
        READBACK_BUFFER(q_got, rt->q_buffer, token_bytes);
        READBACK_BUFFER(k_got, rt->k_buffer, kv_rope_bytes);
        READBACK_BUFFER(v_got, rt->v_buffer, kv_rope_bytes);
        READBACK_BUFFER(cache_k, rt->cache_k_buffer, cache_kv_bytes);
        READBACK_BUFFER(cache_v, rt->cache_v_buffer, cache_kv_bytes);
        READBACK_BUFFER(written, rt->cache_written_buffer, cache_meta_bytes);
        READBACK_BUFFER(indices, rt->cache_indices_buffer, cache_meta_bytes);
        READBACK_BUFFER(cache_index_count, rt->cache_index_count_buffer, cache_count_bytes);
        READBACK_BUFFER(attn, rt->attn_buffer, token_bytes);
        READBACK_BUFFER(attn_proj, rt->attn_proj_buffer, token_bytes);
        READBACK_BUFFER(tokens_after_attn, rt->tokens_after_attn_buffer, token_bytes);
        READBACK_BUFFER(ctrl_norm, rt->ctrl_norm_buffer, token_bytes);
        READBACK_BUFFER(ctrl_hidden, rt->ctrl_hidden_layer_buffer, token_bytes);
        READBACK_BUFFER(ctrl_out, rt->ctrl_out_buffer, token_bytes);
        READBACK_BUFFER(tokens_after_ctrl, rt->tokens_after_ctrl_buffer, token_bytes);
        READBACK_BUFFER(mlp_in, rt->mlp_in_buffer, token_bytes);
        READBACK_BUFFER(mlp_hidden, rt->mlp_hidden_buffer, mlp_hidden_token_elems * sizeof(float));
        READBACK_BUFFER(mlp_out, rt->mlp_out_buffer, token_bytes);
        READBACK_BUFFER(tokens_after_mlp, rt->tokens_after_mlp_buffer, token_bytes);
        READBACK_BUFFER(final_tokens, rt->final_tokens_buffer, token_bytes);
        READBACK_BUFFER(latent_out, rt->latent_out_buffer, latent_elems * sizeof(float));
    }
#undef READBACK_BUFFER
#undef READBACK_ALLOC

    for (int t = 0; t < T; ++t) {
        int y = t / width;
        int x = t - y * width;
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < C; ++c) {
                acc += latent[(c * height + y) * width + x] * patch[d * C + c];
            }
            tokens_ref[t * D + d] = acc;
        }
    }

    const float *layer_mod = (const float *)rt->layer_mod_table_mapped;
    const float *scale = layer_mod;
    const float *bias = layer_mod + D;
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = tokens_ref[t * D + d];
            sum += v * v;
        }
        float inv = 1.0f / sqrtf(sum / (float)D + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
            norm_ref[t * D + d] = tokens_ref[t * D + d] * inv * (1.0f + scale[d]) + bias[d];
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < qkv_dim; ++o) {
            const float *w = o < D ? q_w + (size_t)o * D :
                (o < D + kv_dim ? k_w + (size_t)(o - D) * D : v_w + (size_t)(o - D - kv_dim) * D);
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                acc += norm_ref[t * D + d] * w[d];
            }
            qkv_ref[t * qkv_dim + o] = acc;
        }
    }

    float norm_max = 0.0f;
    float qkv_max = 0.0f;
    float rope_max = 0.0f;
    float norm_mean = 0.0f;
    float qkv_mean = 0.0f;
    float rope_mean = 0.0f;
    for (int i = 0; i < T * D; ++i) {
        float diff = fabsf(norm_got[i] - norm_ref[i]);
        if (diff > norm_max) norm_max = diff;
        norm_mean += diff;
    }
    for (int i = 0; i < T * qkv_dim; ++i) {
        float diff = fabsf(qkv_got[i] - qkv_ref[i]);
        if (diff > qkv_max) qkv_max = diff;
        qkv_mean += diff;
    }

    const uint32_t *x_pos = (const uint32_t *)rt->x_pos_mapped;
    const uint32_t *y_pos = (const uint32_t *)rt->y_pos_mapped;
    const uint32_t *t_pos = (const uint32_t *)rt->t_pos_mapped;
    const float *xy = (const float *)rt->xy_mapped;
    const float *inv_t = (const float *)rt->inv_t_mapped;
    int rope_count = 0;
    for (int t = 0; t < T; ++t) {
        for (int role = 0; role < n_heads + n_kv_heads; ++role) {
            int is_k = role >= n_heads;
            int h = is_k ? role - n_heads : role;
            int src_head = is_k ? n_heads + h : h;
            const float *src = qkv_ref + t * qkv_dim + src_head * d_head;
            float sum = 0.0f;
            for (int d = 0; d < d_head; ++d) {
                sum += src[d] * src[d];
            }
            float inv = 1.0f / sqrtf(sum / (float)d_head + 1.0e-6f);
            const float *dst = is_k ? k_got + (h * T + t) * d_head : q_got + (h * T + t) * d_head;
            for (int p = 0; p < d_head / 2; ++p) {
                float phase = probe_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, d_head, width, height);
                float c = cosf(phase);
                float s = sinf(phase);
                float a = src[2 * p] * inv;
                float bb = src[2 * p + 1] * inv;
                float ref0 = a * c - bb * s;
                float ref1 = bb * c + a * s;
                float diff0 = fabsf(dst[p] - ref0);
                float diff1 = fabsf(dst[d_head / 2 + p] - ref1);
                if (diff0 > rope_max) rope_max = diff0;
                if (diff1 > rope_max) rope_max = diff1;
                rope_mean += diff0 + diff1;
                rope_count += 2;
            }
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            const float *src = qkv_ref + t * qkv_dim + (n_heads + n_kv_heads + h) * d_head;
            const float *dst = v_got + (h * T + t) * d_head;
            for (int d = 0; d < d_head; ++d) {
                float diff = fabsf(dst[d] - src[d]);
                if (diff > rope_max) rope_max = diff;
                rope_mean += diff;
                rope_count += 1;
            }
        }
    }
    float cache_max = 0.0f;
    float attn_max = 0.0f;
    float attn_proj_max = 0.0f;
    float tokens_after_attn_max = 0.0f;
    float ctrl_cond_max = 0.0f;
    float ctrl_norm_max = 0.0f;
    float ctrl_hidden_max = 0.0f;
    float ctrl_out_max = 0.0f;
    float tokens_after_ctrl_max = 0.0f;
    float mlp_in_max = 0.0f;
    float mlp_hidden_max = 0.0f;
    float mlp_out_max = 0.0f;
    float tokens_after_mlp_max = 0.0f;
    float final_tokens_max = 0.0f;
    float latent_out_max = 0.0f;
    float latent_final_max = 0.0f;
    float attn_mean = 0.0f;
    float attn_proj_mean = 0.0f;
    float tokens_after_attn_mean = 0.0f;
    float tokens_after_ctrl_mean = 0.0f;
    float mlp_in_mean = 0.0f;
    float mlp_hidden_mean = 0.0f;
    float mlp_out_mean = 0.0f;
    float tokens_after_mlp_mean = 0.0f;
    float final_tokens_mean = 0.0f;
    float latent_out_mean = 0.0f;
    float latent_final_mean = 0.0f;
    uint32_t cache_mismatches = 0;
    const uint32_t index_count = cache_index_count[0];
    const float *ctrl_cond = (const float *)rt->ctrl_cond_mapped;
    const float *latent_final = (const float *)rt->latent_mapped;
    if (index_count != (uint32_t)T) {
        ++cache_mismatches;
    }
    for (int i = 0; i < rt->cache_capacity; ++i) {
        uint32_t ref_written = i >= rt->cache_ring_length ? 1u : 0u;
        if (written[i] != ref_written) ++cache_mismatches;
    }
    for (int t = 0; t < T; ++t) {
        uint32_t ref_idx = (uint32_t)(rt->cache_ring_length + t);
        if (indices[t] != ref_idx) ++cache_mismatches;
    }
    for (int h = 0; h < n_kv_heads; ++h) {
        for (int slot = 0; slot < rt->cache_capacity; ++slot) {
            int tail_t = slot - rt->cache_ring_length;
            for (int d = 0; d < d_head; ++d) {
                size_t cache_idx = (size_t)(h * rt->cache_capacity + slot) * d_head + d;
                float ref_k = 0.0f;
                float ref_v = 0.0f;
                if (tail_t >= 0 && tail_t < T) {
                    ref_k = k_got[(h * T + tail_t) * d_head + d];
                    ref_v = v_got[(h * T + tail_t) * d_head + d];
                }
                float dk = fabsf(cache_k[cache_idx] - ref_k);
                float dv = fabsf(cache_v[cache_idx] - ref_v);
                if (dk > cache_max) cache_max = dk;
                if (dv > cache_max) cache_max = dv;
                if (dk != 0.0f || dv != 0.0f) ++cache_mismatches;
            }
        }
    }
    int group = n_heads / n_kv_heads;
    for (int tq = 0; tq < T; ++tq) {
        for (int hq = 0; hq < n_heads; ++hq) {
            int hk = hq / group;
            const float *qrow = q_got + (hq * T + tq) * d_head;
            const float *kbase = cache_k + (size_t)hk * rt->cache_capacity * d_head;
            const float *vbase = cache_v + (size_t)hk * rt->cache_capacity * d_head;
            float scores[T];
            float max_score = -INFINITY;
            for (int n = 0; n < T; ++n) {
                int tk = (int)indices[n];
                float dot = 0.0f;
                for (int d = 0; d < d_head; ++d) {
                    dot += qrow[d] * kbase[tk * d_head + d];
                }
                scores[n] = dot / sqrtf((float)d_head);
                if (scores[n] > max_score) max_score = scores[n];
            }
            float denom = 0.0f;
            for (int n = 0; n < T; ++n) {
                denom += expf(scores[n] - max_score);
            }
            for (int d = 0; d < d_head; ++d) {
                float ref = 0.0f;
                for (int n = 0; n < T; ++n) {
                    int tk = (int)indices[n];
                    ref += expf(scores[n] - max_score) * vbase[tk * d_head + d];
                }
                ref /= denom;
                size_t idx = (size_t)(tq * n_heads + hq) * d_head + d;
                float diff = fabsf(attn[idx] - ref);
                if (diff > attn_max) attn_max = diff;
                attn_mean += diff;
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < D; ++o) {
            float ref = 0.0f;
            for (int d = 0; d < D; ++d) {
                ref += attn[t * D + d] * out_w[o * D + d];
            }
            attn_proj_ref[t * D + o] = ref;
            float diff = fabsf(attn_proj[t * D + o] - ref);
            if (diff > attn_proj_max) attn_proj_max = diff;
            attn_proj_mean += diff;
        }
    }
    const float *gate = layer_mod + 2 * D;
    for (int i = 0; i < T * D; ++i) {
        int d = i % D;
        float ref = tokens_ref[i] + attn_proj_ref[i] * gate[d];
        tokens_after_attn_ref[i] = ref;
        float diff = fabsf(tokens_after_attn[i] - ref);
        if (diff > tokens_after_attn_max) tokens_after_attn_max = diff;
        tokens_after_attn_mean += diff;
    }
    const float *ctrl_emb_norm = (const float *)rt->ctrl_emb_norm_mapped;
    for (int o = 0; o < D; ++o) {
        float ref = 0.0f;
        for (int d = 0; d < D; ++d) {
            ref += ctrl_emb_norm[d] * ctrl_fc1_c_w[o * D + d];
        }
        ctrl_cond_ref[o] = ref;
        float diff = fabsf(ctrl_cond[o] - ref);
        if (diff > ctrl_cond_max) ctrl_cond_max = diff;
    }
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = tokens_after_attn_ref[t * D + d];
            sum += v * v;
        }
        float inv = 1.0f / sqrtf(sum / (float)D + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
            float ref = tokens_after_attn_ref[t * D + d] * inv;
            ctrl_norm_ref[t * D + d] = ref;
            float diff = fabsf(ctrl_norm[t * D + d] - ref);
            if (diff > ctrl_norm_max) ctrl_norm_max = diff;
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < D; ++o) {
            float ref = 0.0f;
            for (int d = 0; d < D; ++d) {
                ref += ctrl_norm_ref[t * D + d] * ctrl_fc1_x_w[o * D + d];
            }
            ref += ctrl_cond_ref[o];
            ref = ref / (1.0f + expf(-ref));
            ctrl_hidden_ref[t * D + o] = ref;
            float diff = fabsf(ctrl_hidden[t * D + o] - ref);
            if (diff > ctrl_hidden_max) ctrl_hidden_max = diff;
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < D; ++o) {
            float ref = 0.0f;
            for (int d = 0; d < D; ++d) {
                ref += ctrl_hidden_ref[t * D + d] * ctrl_fc2_w[o * D + d];
            }
            ctrl_out_ref[t * D + o] = ref;
            float diff = fabsf(ctrl_out[t * D + o] - ref);
            if (diff > ctrl_out_max) ctrl_out_max = diff;
        }
    }
    for (int i = 0; i < T * D; ++i) {
        float ref = tokens_after_attn_ref[i] + ctrl_out_ref[i];
        tokens_after_ctrl_ref[i] = ref;
        float diff = fabsf(tokens_after_ctrl[i] - ref);
        if (diff > tokens_after_ctrl_max) tokens_after_ctrl_max = diff;
        tokens_after_ctrl_mean += diff;
    }
    const float *mlp_scale = layer_mod + 3 * D;
    const float *mlp_bias = layer_mod + 4 * D;
    const float *mlp_gate = layer_mod + 5 * D;
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = tokens_after_ctrl_ref[t * D + d];
            sum += v * v;
        }
        float inv = 1.0f / sqrtf(sum / (float)D + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
            float ref = tokens_after_ctrl_ref[t * D + d] * inv * (1.0f + mlp_scale[d]) + mlp_bias[d];
            mlp_in_ref[t * D + d] = ref;
            float diff = fabsf(mlp_in[t * D + d] - ref);
            if (diff > mlp_in_max) mlp_in_max = diff;
            mlp_in_mean += diff;
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < hidden; ++o) {
            float ref = 0.0f;
            for (int d = 0; d < D; ++d) {
                ref += mlp_in_ref[t * D + d] * dit_mlp_fc1_w[o * D + d];
            }
            ref = ref / (1.0f + expf(-ref));
            mlp_hidden_ref[t * hidden + o] = ref;
            float diff = fabsf(mlp_hidden[t * hidden + o] - ref);
            if (diff > mlp_hidden_max) mlp_hidden_max = diff;
            mlp_hidden_mean += diff;
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < D; ++o) {
            float ref = 0.0f;
            for (int h = 0; h < hidden; ++h) {
                ref += mlp_hidden_ref[t * hidden + h] * dit_mlp_fc2_w[o * hidden + h];
            }
            mlp_out_ref[t * D + o] = ref;
            float diff = fabsf(mlp_out[t * D + o] - ref);
            if (diff > mlp_out_max) mlp_out_max = diff;
            mlp_out_mean += diff;
        }
    }
    for (int i = 0; i < T * D; ++i) {
        int d = i % D;
        float ref = tokens_after_ctrl_ref[i] + mlp_out_ref[i] * mlp_gate[d];
        tokens_after_mlp_ref[i] = ref;
        float diff = fabsf(tokens_after_mlp[i] - ref);
        if (diff > tokens_after_mlp_max) tokens_after_mlp_max = diff;
        tokens_after_mlp_mean += diff;
    }
    const float *out_mod = (const float *)rt->out_mod_table_mapped;
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = tokens_after_mlp_ref[t * D + d];
            sum += v * v;
        }
        float inv = 1.0f / sqrtf(sum / (float)D + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
            float v = tokens_after_mlp_ref[t * D + d] * inv * (1.0f + out_mod[d]) + out_mod[D + d];
            float ref = v / (1.0f + expf(-v));
            final_tokens_ref[t * D + d] = ref;
            float diff = fabsf(final_tokens[t * D + d] - ref);
            if (diff > final_tokens_max) final_tokens_max = diff;
            final_tokens_mean += diff;
        }
    }
    for (int t = 0; t < T; ++t) {
        int y = t / width;
        int x = t - y * width;
        for (int c = 0; c < C; ++c) {
            float ref = unpatch_bias[c];
            for (int d = 0; d < D; ++d) {
                ref += final_tokens_ref[t * D + d] * unpatch[d * C + c];
            }
            size_t idx = (size_t)(c * height + y) * width + x;
            latent_out_ref[idx] = ref;
            float diff = fabsf(latent_out[idx] - ref);
            if (diff > latent_out_max) latent_out_max = diff;
            latent_out_mean += diff;
        }
    }
    for (size_t i = 0; i < latent_elems; ++i) {
        float ref = latent[i] + (cfg.scheduler_sigmas[1] - cfg.scheduler_sigmas[0]) * latent_out_ref[i];
        latent_final_ref[i] = ref;
        float diff = fabsf(latent_final[i] - ref);
        if (diff > latent_final_max) latent_final_max = diff;
        latent_final_mean += diff;
    }
    norm_mean /= (float)(T * D);
    qkv_mean /= (float)(T * qkv_dim);
    rope_mean /= (float)rope_count;
    attn_mean /= (float)(T * D);
    attn_proj_mean /= (float)(T * D);
    tokens_after_attn_mean /= (float)(T * D);
    tokens_after_ctrl_mean /= (float)(T * D);
    mlp_in_mean /= (float)(T * D);
    mlp_hidden_mean /= (float)mlp_hidden_token_elems;
    mlp_out_mean /= (float)(T * D);
    tokens_after_mlp_mean /= (float)(T * D);
    final_tokens_mean /= (float)(T * D);
    latent_out_mean /= (float)latent_elems;
    latent_final_mean /= (float)latent_elems;
    fprintf(stderr,
            "vulkan runtime_layer0_qkv_f32 probe: norm_max=%g qkv_max=%g rope_max=%g cache_max=%g attn_max=%g attn_proj_max=%g tokens_after_attn_max=%g ctrl_cond_max=%g ctrl_norm_max=%g ctrl_hidden_max=%g ctrl_out_max=%g tokens_after_ctrl_max=%g mlp_in_max=%g mlp_hidden_max=%g mlp_out_max=%g tokens_after_mlp_max=%g final_tokens_max=%g latent_out_max=%g latent_final_max=%g cache_mismatches=%u norm_mean=%g qkv_mean=%g rope_mean=%g attn_mean=%g attn_proj_mean=%g tokens_after_attn_mean=%g tokens_after_ctrl_mean=%g mlp_in_mean=%g mlp_hidden_mean=%g mlp_out_mean=%g tokens_after_mlp_mean=%g final_tokens_mean=%g latent_out_mean=%g latent_final_mean=%g\n",
            norm_max, qkv_max, rope_max, cache_max, attn_max, attn_proj_max, tokens_after_attn_max,
            ctrl_cond_max, ctrl_norm_max, ctrl_hidden_max, ctrl_out_max, tokens_after_ctrl_max,
            mlp_in_max, mlp_hidden_max, mlp_out_max, tokens_after_mlp_max,
            final_tokens_max, latent_out_max, latent_final_max,
            cache_mismatches, norm_mean, qkv_mean, rope_mean, attn_mean, attn_proj_mean,
            tokens_after_attn_mean, tokens_after_ctrl_mean,
            mlp_in_mean, mlp_hidden_mean, mlp_out_mean, tokens_after_mlp_mean,
            final_tokens_mean, latent_out_mean, latent_final_mean);
    if (norm_max > 4.0e-5f || qkv_max > 7.0e-5f || rope_max > 8.0e-5f ||
            cache_max != 0.0f || cache_mismatches != 0 || attn_max > 8.0e-5f ||
            attn_proj_max > 9.0e-5f || tokens_after_attn_max > 9.0e-5f ||
            ctrl_cond_max > 8.0e-5f || ctrl_norm_max > 8.0e-5f ||
            ctrl_hidden_max > 8.0e-5f || ctrl_out_max > 9.0e-5f ||
            tokens_after_ctrl_max > 9.0e-5f || mlp_in_max > 9.0e-5f ||
            mlp_hidden_max > 1.0e-4f || mlp_out_max > 1.2e-4f ||
            tokens_after_mlp_max > 1.2e-4f || final_tokens_max > 9.0e-5f ||
            latent_out_max > 1.2e-4f || latent_final_max > 1.2e-4f) goto cleanup;
    rc = 0;

cleanup:
    world_vulkan_runtime_destroy(rt);
    free(latent);
    free(control);
    free(patch);
    free(ctrl_fc1);
    free(ctrl_fc2);
    free(denoise_fc1);
    free(denoise_fc2);
    free(out_norm);
    free(unpatch);
    free(unpatch_bias);
    free(cond_bias);
    free(cond_proj);
    free(q_w);
    free(k_w);
    free(v_w);
    free(out_w);
    free(ctrl_fc1_c_w);
    free(ctrl_fc1_x_w);
    free(ctrl_fc2_w);
    free(dit_mlp_fc1_w);
    free(dit_mlp_fc2_w);
    free(tokens_ref);
    free(norm_ref);
    free(qkv_ref);
    free(attn_proj_ref);
    free(tokens_after_attn_ref);
    free(ctrl_cond_ref);
    free(ctrl_norm_ref);
    free(ctrl_hidden_ref);
    free(ctrl_out_ref);
    free(tokens_after_ctrl_ref);
    free(mlp_in_ref);
    free(mlp_hidden_ref);
    free(mlp_out_ref);
    free(tokens_after_mlp_ref);
    free(final_tokens_ref);
    free(latent_out_ref);
    free(latent_final_ref);
    free(norm_got);
    free(qkv_got);
    free(q_got);
    free(k_got);
    free(v_got);
    free(cache_k);
    free(cache_v);
    free(written);
    free(indices);
    free(cache_index_count);
    free(attn);
    free(attn_proj);
    free(tokens_after_attn);
    free(ctrl_norm);
    free(ctrl_hidden);
    free(ctrl_out);
    free(tokens_after_ctrl);
    free(mlp_in);
    free(mlp_hidden);
    free(mlp_out);
    free(tokens_after_mlp);
    free(final_tokens);
    free(latent_out);
    return rc;
}

int world_vulkan_masked_attention_f32_probe(void) {
    enum { B = 2, Hq = 6, Hkv = 2, Tq = 7, Tk = 11, D = 64 };
    const float scale = 1.0f / sqrtf((float)D);
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer q_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkBuffer written_buffer = VK_NULL_HANDLE;
    VkBuffer out_buffer = VK_NULL_HANDLE;
    VkDeviceMemory q_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    VkDeviceMemory written_memory = VK_NULL_HANDLE;
    VkDeviceMemory out_memory = VK_NULL_HANDLE;
    void *q_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;
    void *written_mapped = NULL;
    void *out_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t q_bytes = (size_t)B * Hq * Tq * D * sizeof(float);
    size_t kv_bytes = (size_t)B * Hkv * Tk * D * sizeof(float);
    size_t written_bytes = (size_t)Tk * sizeof(uint32_t);
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &q_buffer, &q_memory, &q_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v_buffer, &v_memory, &v_mapped)) goto cleanup;
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &written_buffer, &written_memory, &written_mapped)) goto cleanup;
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) goto cleanup;

    float *q = (float *)q_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    uint32_t *written = (uint32_t *)written_mapped;
    float *out = (float *)out_mapped;
    for (int i = 0; i < B * Hq * Tq * D; ++i) q[i] = probe_value(i + 701, 0.0234375f);
    for (int i = 0; i < B * Hkv * Tk * D; ++i) {
        k[i] = probe_value(i + 809, 0.021484375f);
        v[i] = probe_value(i + 907, 0.033203125f);
    }
    const uint32_t written_ref[Tk] = {1u, 1u, 0u, 1u, 0u, 1u, 1u, 0u, 1u, 0u, 1u};
    memcpy(written, written_ref, written_bytes);
    memset(out, 0, q_bytes);

    if (create_storage_pipeline(rt, "masked_attention_f32.comp", 5, sizeof(WorldVulkanMaskedAttentionPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[5] = {q_buffer, k_buffer, v_buffer, written_buffer, out_buffer};
    VkDeviceSize sizes[5] = {q_bytes, kv_bytes, kv_bytes, written_bytes, q_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 5, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanMaskedAttentionPush push;
    push.B = B;
    push.Hq = Hq;
    push.Hkv = Hkv;
    push.Tq = Tq;
    push.Tk = Tk;
    push.D = D;
    push.scale = scale;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * Hq * Tq, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int group = Hq / Hkv;
    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < Hq; ++hq) {
            int hk = hq / group;
            for (int tq = 0; tq < Tq; ++tq) {
                const float *qrow = q + ((b * Hq + hq) * Tq + tq) * D;
                const float *kbase = k + (b * Hkv + hk) * Tk * D;
                const float *vbase = v + (b * Hkv + hk) * Tk * D;
                float scores[Tk];
                float max_score = -INFINITY;
                for (int tk = 0; tk < Tk; ++tk) {
                    if (!written[tk]) {
                        scores[tk] = -INFINITY;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += qrow[d] * kbase[tk * D + d];
                    }
                    scores[tk] = dot * scale;
                    if (scores[tk] > max_score) max_score = scores[tk];
                }
                float denom = 0.0f;
                for (int tk = 0; tk < Tk; ++tk) {
                    if (written[tk]) denom += expf(scores[tk] - max_score);
                }
                for (int d = 0; d < D; ++d) {
                    float ref = 0.0f;
                    if (denom > 0.0f) {
                        for (int tk = 0; tk < Tk; ++tk) {
                            if (written[tk]) {
                                ref += expf(scores[tk] - max_score) * vbase[tk * D + d];
                            }
                        }
                        ref /= denom;
                    }
                    size_t idx = (size_t)(((b * Hq + hq) * Tq + tq) * D + d);
                    float diff = fabsf(out[idx] - ref);
                    if (diff > max_abs) max_abs = diff;
                    mean_abs += diff;
                }
            }
        }
    }
    mean_abs /= (float)(B * Hq * Tq * D);
    fprintf(stderr, "vulkan masked_attention_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, q_buffer, q_memory, q_mapped);
        destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
        destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
        destroy_host_buffer(rt, written_buffer, written_memory, written_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_gated_residual_add_f32_probe(void) {
    enum { T = 7, D = 257, n = T * D };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer residual_buffer = VK_NULL_HANDLE;
    VkBuffer update_buffer = VK_NULL_HANDLE;
    VkBuffer gate_buffer = VK_NULL_HANDLE;
    VkBuffer out_buffer = VK_NULL_HANDLE;
    VkDeviceMemory residual_memory = VK_NULL_HANDLE;
    VkDeviceMemory update_memory = VK_NULL_HANDLE;
    VkDeviceMemory gate_memory = VK_NULL_HANDLE;
    VkDeviceMemory out_memory = VK_NULL_HANDLE;
    void *residual_mapped = NULL;
    void *update_mapped = NULL;
    void *gate_mapped = NULL;
    void *out_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)n * sizeof(float);
    size_t gate_bytes = (size_t)D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &residual_buffer, &residual_memory, &residual_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &update_buffer, &update_memory, &update_mapped)) goto cleanup;
    if (create_host_buffer(rt, gate_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &gate_buffer, &gate_memory, &gate_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) goto cleanup;

    float *residual = (float *)residual_mapped;
    float *update = (float *)update_mapped;
    float *gate = (float *)gate_mapped;
    float *out = (float *)out_mapped;
    for (int i = 0; i < n; ++i) {
        residual[i] = probe_value(i + 1601, 0.03125f);
        update[i] = probe_value(i + 1709, 0.02734375f);
    }
    for (int d = 0; d < D; ++d) gate[d] = probe_value(d + 1801, 0.01953125f);
    memset(out, 0, x_bytes);

    if (create_storage_pipeline(rt, "gated_residual_add_f32.comp", 4, sizeof(WorldVulkanGatedResidualPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[4] = {residual_buffer, update_buffer, gate_buffer, out_buffer};
    VkDeviceSize sizes[4] = {x_bytes, x_bytes, gate_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 4, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanGatedResidualPush push;
    push.T = T;
    push.D = D;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (uint32_t)((n + 255) / 256), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = residual[i] + update[i] * gate[i % D];
        float diff = fabsf(out[i] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan gated_residual_add_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-6f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, residual_buffer, residual_memory, residual_mapped);
        destroy_host_buffer(rt, update_buffer, update_memory, update_mapped);
        destroy_host_buffer(rt, gate_buffer, gate_memory, gate_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

static int run_kv_cache_upsert_probe_case(
        WorldVulkanRuntime *rt,
        VkDescriptorSetLayout mask_set_layout,
        VkPipelineLayout mask_pipeline_layout,
        VkPipeline mask_pipeline,
        VkDescriptorSetLayout upsert_set_layout,
        VkPipelineLayout upsert_pipeline_layout,
        VkPipeline upsert_pipeline,
        const char *name,
        uint32_t B,
        uint32_t H,
        uint32_t T,
        uint32_t D,
        uint32_t L,
        uint32_t frame_idx,
        uint32_t pinned_dilation,
        uint32_t frozen) {
    int rc = 1;
    uint32_t capacity = L + T;
    uint32_t bucket = (frame_idx + pinned_dilation - 1u) / pinned_dilation;
    uint32_t num_buckets = (L / T) / pinned_dilation;
    uint32_t base = (bucket % num_buckets) * T;
    uint32_t write_step = (frame_idx % pinned_dilation) == 0u ? 1u : 0u;
    size_t cache_values = (size_t)B * H * capacity * D;
    size_t kv_values = (size_t)B * H * T * D;
    size_t cache_bytes = cache_values * sizeof(float);
    size_t kv_bytes = kv_values * sizeof(float);
    size_t written_bytes = (size_t)capacity * sizeof(uint32_t);

    VkDescriptorPool mask_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet mask_descriptor_set = VK_NULL_HANDLE;
    VkDescriptorPool upsert_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet upsert_descriptor_set = VK_NULL_HANDLE;
    VkBuffer cache_k_buffer = VK_NULL_HANDLE;
    VkBuffer cache_v_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkBuffer written_buffer = VK_NULL_HANDLE;
    VkBuffer mask_buffer = VK_NULL_HANDLE;
    VkDeviceMemory cache_k_memory = VK_NULL_HANDLE;
    VkDeviceMemory cache_v_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    VkDeviceMemory written_memory = VK_NULL_HANDLE;
    VkDeviceMemory mask_memory = VK_NULL_HANDLE;
    void *cache_k_mapped = NULL;
    void *cache_v_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;
    void *written_mapped = NULL;
    void *mask_mapped = NULL;
    float *ref_cache_k = NULL;
    float *ref_cache_v = NULL;
    uint32_t *ref_written = NULL;
    uint32_t *ref_mask = NULL;

    if (create_host_buffer(rt, cache_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &cache_k_buffer, &cache_k_memory, &cache_k_mapped)) goto cleanup;
    if (create_host_buffer(rt, cache_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &cache_v_buffer, &cache_v_memory, &cache_v_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &v_buffer, &v_memory, &v_mapped)) goto cleanup;
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &written_buffer, &written_memory, &written_mapped)) goto cleanup;
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &mask_buffer, &mask_memory, &mask_mapped)) goto cleanup;

    float *cache_k = (float *)cache_k_mapped;
    float *cache_v = (float *)cache_v_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    uint32_t *written = (uint32_t *)written_mapped;
    uint32_t *mask = (uint32_t *)mask_mapped;

    for (size_t i = 0; i < cache_values; ++i) {
        cache_k[i] = frozen ? probe_value((int)i + 1009, 0.017578125f) : 0.0f;
        cache_v[i] = frozen ? probe_value((int)i + 1201, 0.01953125f) : 0.0f;
    }
    for (size_t i = 0; i < kv_values; ++i) {
        k[i] = probe_value((int)i + 1409, 0.025390625f);
        v[i] = probe_value((int)i + 1601, 0.029296875f);
    }
    memset(written, 0, written_bytes);
    uint32_t initially_written = frozen ? T : 2u * T;
    for (uint32_t i = 0; i < initially_written && i < capacity; ++i) written[i] = 1u;
    for (uint32_t i = L; i < capacity; ++i) written[i] = 1u;
    memset(mask, 0, written_bytes);

    ref_cache_k = (float *)malloc(cache_bytes);
    ref_cache_v = (float *)malloc(cache_bytes);
    ref_written = (uint32_t *)malloc(written_bytes);
    ref_mask = (uint32_t *)malloc(written_bytes);
    if (!ref_cache_k || !ref_cache_v || !ref_written || !ref_mask) goto cleanup;
    memcpy(ref_cache_k, cache_k, cache_bytes);
    memcpy(ref_cache_v, cache_v, cache_bytes);
    memcpy(ref_written, written, written_bytes);
    memcpy(ref_mask, written, written_bytes);

    if (write_step) {
        for (uint32_t t = 0; t < T; ++t) ref_mask[base + t] = 0u;
    }
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t t = 0; t < T; ++t) {
                uint32_t tail_idx = L + t;
                uint32_t ring_idx = base + t;
                uint32_t dst_idx = (!frozen && write_step) ? ring_idx : tail_idx;
                for (uint32_t d = 0; d < D; ++d) {
                    size_t src = (size_t)(((b * H + h) * T + t) * D + d);
                    size_t tail = (size_t)(((b * H + h) * capacity + tail_idx) * D + d);
                    size_t dst = (size_t)(((b * H + h) * capacity + dst_idx) * D + d);
                    ref_cache_k[tail] = k[src];
                    ref_cache_v[tail] = v[src];
                    if (!frozen) {
                        ref_cache_k[dst] = k[src];
                        ref_cache_v[dst] = v[src];
                    }
                }
            }
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t tail_idx = L + t;
        uint32_t dst_idx = (!frozen && write_step) ? base + t : tail_idx;
        ref_written[tail_idx] = 1u;
        if (!frozen) ref_written[dst_idx] = 1u;
    }

    VkBuffer mask_buffers[2] = {written_buffer, mask_buffer};
    VkDeviceSize mask_sizes[2] = {written_bytes, written_bytes};
    if (create_storage_descriptor_set(rt, mask_set_layout, 2, mask_buffers, mask_sizes, NULL,
                &mask_descriptor_pool, &mask_descriptor_set)) goto cleanup;
    WorldVulkanKvCacheMaskPush mask_push;
    mask_push.capacity = capacity;
    mask_push.T = T;
    mask_push.base = base;
    mask_push.write_step = write_step;
    if (submit_compute(rt, mask_pipeline, mask_pipeline_layout, mask_descriptor_set,
                &mask_push, sizeof(mask_push), (capacity + 255u) / 256u, 1, 1)) goto cleanup;

    VkBuffer upsert_buffers[5] = {cache_k_buffer, cache_v_buffer, k_buffer, v_buffer, written_buffer};
    VkDeviceSize upsert_sizes[5] = {cache_bytes, cache_bytes, kv_bytes, kv_bytes, written_bytes};
    if (create_storage_descriptor_set(rt, upsert_set_layout, 5, upsert_buffers, upsert_sizes, NULL,
                &upsert_descriptor_pool, &upsert_descriptor_set)) goto cleanup;
    WorldVulkanKvCacheUpsertPush upsert_push;
    upsert_push.B = B;
    upsert_push.H = H;
    upsert_push.T = T;
    upsert_push.D = D;
    upsert_push.L = L;
    upsert_push.base = base;
    upsert_push.write_step = write_step;
    upsert_push.frozen = frozen;
    if (submit_compute(rt, upsert_pipeline, upsert_pipeline_layout, upsert_descriptor_set,
                &upsert_push, sizeof(upsert_push), (uint32_t)((kv_values + 255u) / 256u), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    uint32_t mismatches = 0;
    for (size_t i = 0; i < cache_values; ++i) {
        float dk = fabsf(cache_k[i] - ref_cache_k[i]);
        float dv = fabsf(cache_v[i] - ref_cache_v[i]);
        if (dk > max_abs) max_abs = dk;
        if (dv > max_abs) max_abs = dv;
        if (dk != 0.0f || dv != 0.0f) ++mismatches;
    }
    for (uint32_t i = 0; i < capacity; ++i) {
        if (written[i] != ref_written[i]) ++mismatches;
        if (mask[i] != ref_mask[i]) ++mismatches;
    }
    fprintf(stderr, "vulkan kv_cache_upsert_f32 probe (%s): max_abs=%g mismatches=%u\n",
            name, max_abs, mismatches);
    if (mismatches != 0 || max_abs != 0.0f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (mask_descriptor_pool) vkDestroyDescriptorPool(rt->device, mask_descriptor_pool, NULL);
    if (upsert_descriptor_pool) vkDestroyDescriptorPool(rt->device, upsert_descriptor_pool, NULL);
    destroy_host_buffer(rt, cache_k_buffer, cache_k_memory, cache_k_mapped);
    destroy_host_buffer(rt, cache_v_buffer, cache_v_memory, cache_v_mapped);
    destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
    destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
    destroy_host_buffer(rt, written_buffer, written_memory, written_mapped);
    destroy_host_buffer(rt, mask_buffer, mask_memory, mask_mapped);
    free(ref_cache_k);
    free(ref_cache_v);
    free(ref_written);
    free(ref_mask);
    return rc;
}

int world_vulkan_kv_cache_upsert_f32_probe(void) {
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule mask_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout mask_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout mask_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline mask_pipeline = VK_NULL_HANDLE;
    VkShaderModule upsert_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout upsert_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout upsert_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline upsert_pipeline = VK_NULL_HANDLE;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (create_storage_pipeline(rt, "kv_cache_mask.comp", 2, sizeof(WorldVulkanKvCacheMaskPush),
                &mask_shader, &mask_set_layout, &mask_pipeline_layout, &mask_pipeline)) goto cleanup;
    if (create_storage_pipeline(rt, "kv_cache_upsert_copy_f32.comp", 5, sizeof(WorldVulkanKvCacheUpsertPush),
                &upsert_shader, &upsert_set_layout, &upsert_pipeline_layout, &upsert_pipeline)) goto cleanup;

    if (run_kv_cache_upsert_probe_case(rt, mask_set_layout, mask_pipeline_layout, mask_pipeline,
                upsert_set_layout, upsert_pipeline_layout, upsert_pipeline,
                "frozen_write_step", 1, 2, 4, 8, 16, 4, 1, 1)) goto cleanup;
    if (run_kv_cache_upsert_probe_case(rt, mask_set_layout, mask_pipeline_layout, mask_pipeline,
                upsert_set_layout, upsert_pipeline_layout, upsert_pipeline,
                "unfrozen_pinned_dilation", 1, 3, 4, 16, 32, 5, 2, 0)) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (mask_pipeline) vkDestroyPipeline(rt->device, mask_pipeline, NULL);
    if (mask_pipeline_layout) vkDestroyPipelineLayout(rt->device, mask_pipeline_layout, NULL);
    if (mask_set_layout) vkDestroyDescriptorSetLayout(rt->device, mask_set_layout, NULL);
    if (mask_shader) vkDestroyShaderModule(rt->device, mask_shader, NULL);
    if (upsert_pipeline) vkDestroyPipeline(rt->device, upsert_pipeline, NULL);
    if (upsert_pipeline_layout) vkDestroyPipelineLayout(rt->device, upsert_pipeline_layout, NULL);
    if (upsert_set_layout) vkDestroyDescriptorSetLayout(rt->device, upsert_set_layout, NULL);
    if (upsert_shader) vkDestroyShaderModule(rt->device, upsert_shader, NULL);
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_cache_frame_indices_probe(void) {
    enum { T = 4, slots = 9, capacity = T * slots };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer written_buffer = VK_NULL_HANDLE;
    VkBuffer indices_buffer = VK_NULL_HANDLE;
    VkBuffer count_buffer = VK_NULL_HANDLE;
    VkDeviceMemory written_memory = VK_NULL_HANDLE;
    VkDeviceMemory indices_memory = VK_NULL_HANDLE;
    VkDeviceMemory count_memory = VK_NULL_HANDLE;
    void *written_mapped = NULL;
    void *indices_mapped = NULL;
    void *count_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t written_bytes = (size_t)capacity * sizeof(uint32_t);
    size_t indices_bytes = (size_t)capacity * sizeof(uint32_t);
    size_t count_bytes = sizeof(uint32_t);
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &written_buffer, &written_memory, &written_mapped)) goto cleanup;
    if (create_host_buffer(rt, indices_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &indices_buffer, &indices_memory, &indices_mapped)) goto cleanup;
    if (create_host_buffer(rt, count_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &count_buffer, &count_memory, &count_mapped)) goto cleanup;

    uint32_t *written = (uint32_t *)written_mapped;
    uint32_t *indices = (uint32_t *)indices_mapped;
    uint32_t *count = (uint32_t *)count_mapped;
    memset(written, 0, written_bytes);
    const uint32_t written_slots[4] = {0u, 2u, 5u, 8u};
    for (uint32_t s = 0; s < 4u; ++s) {
        uint32_t slot = written_slots[s];
        for (uint32_t t = 0; t < T; ++t) written[slot * T + t] = 1u;
    }

    if (create_storage_pipeline(rt, "cache_frame_indices.comp", 3, sizeof(WorldVulkanCacheFrameIndicesPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {written_buffer, indices_buffer, count_buffer};
    VkDeviceSize sizes[3] = {written_bytes, indices_bytes, count_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;

    for (uint32_t case_id = 0; case_id < 2u; ++case_id) {
        uint32_t base = case_id == 0u ? 2u * T : 0u;
        uint32_t write_step = case_id == 0u ? 1u : 0u;
        for (uint32_t i = 0; i < capacity; ++i) indices[i] = 0xffffffffu;
        count[0] = 0u;

        WorldVulkanCacheFrameIndicesPush push;
        push.capacity = capacity;
        push.T = T;
        push.base = base;
        push.write_step = write_step;
        if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                    slots, 1, 1)) goto cleanup;

        uint32_t ref_indices[capacity];
        uint32_t ref_count = 0u;
        for (uint32_t slot = 0; slot < slots; ++slot) {
            uint32_t slot_base = slot * T;
            uint32_t slot_written = written[slot_base] && !(write_step && slot_base == base);
            if (slot_written) {
                for (uint32_t t = 0; t < T; ++t) ref_indices[ref_count++] = slot_base + t;
            }
        }

        uint32_t mismatches = count[0] == ref_count ? 0u : 1u;
        for (uint32_t i = 0; i < ref_count; ++i) {
            if (indices[i] != ref_indices[i]) ++mismatches;
        }
        fprintf(stderr, "vulkan cache_frame_indices probe (%s): count=%u mismatches=%u\n",
                write_step ? "write_step" : "read_step", count[0], mismatches);
        if (mismatches != 0u) goto cleanup;
    }
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, written_buffer, written_memory, written_mapped);
        destroy_host_buffer(rt, indices_buffer, indices_memory, indices_mapped);
        destroy_host_buffer(rt, count_buffer, count_memory, count_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_patchify_f32_probe(void) {
    enum { B = 2, C = 5, H = 8, W = 10, D = 7, ph = 2, pw = 2 };
    enum { Hp = H / ph, Wp = W / pw, T = Hp * Wp };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer weight_buffer = VK_NULL_HANDLE;
    VkBuffer tokens_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory weight_memory = VK_NULL_HANDLE;
    VkDeviceMemory tokens_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *weight_mapped = NULL;
    void *tokens_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)B * C * H * W * sizeof(float);
    size_t weight_bytes = (size_t)D * C * ph * pw * sizeof(float);
    size_t tokens_bytes = (size_t)B * T * D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &weight_buffer, &weight_memory, &weight_mapped)) goto cleanup;
    if (create_host_buffer(rt, tokens_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tokens_buffer, &tokens_memory, &tokens_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    float *weight = (float *)weight_mapped;
    float *tokens = (float *)tokens_mapped;
    for (int i = 0; i < B * C * H * W; ++i) x[i] = probe_value(i + 1801, 0.02734375f);
    for (int i = 0; i < D * C * ph * pw; ++i) weight[i] = probe_value(i + 1907, 0.033203125f);
    memset(tokens, 0, tokens_bytes);

    if (create_storage_pipeline(rt, "patchify_f32.comp", 3, sizeof(WorldVulkanPatchifyPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, weight_buffer, tokens_buffer};
    VkDeviceSize sizes[3] = {x_bytes, weight_bytes, tokens_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanPatchifyPush push;
    push.B = B;
    push.C = C;
    push.H = H;
    push.W = W;
    push.D = D;
    push.ph = ph;
    push.pw = pw;
    push.Hp = Hp;
    push.Wp = Wp;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * T * D, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int token = 0; token < T; ++token) {
            int oy = token / Wp;
            int ox = token - oy * Wp;
            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                for (int c = 0; c < C; ++c) {
                    for (int dy = 0; dy < ph; ++dy) {
                        for (int dx = 0; dx < pw; ++dx) {
                            int iy = oy * ph + dy;
                            int ix = ox * pw + dx;
                            float xv = x[((b * C + c) * H + iy) * W + ix];
                            float wv = weight[(((d * C + c) * ph + dy) * pw + dx)];
                            ref += xv * wv;
                        }
                    }
                }
                size_t idx = (size_t)((b * T + token) * D + d);
                float diff = fabsf(tokens[idx] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(B * T * D);
    fprintf(stderr, "vulkan patchify_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 3.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, weight_buffer, weight_memory, weight_mapped);
        destroy_host_buffer(rt, tokens_buffer, tokens_memory, tokens_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_unpatchify_f32_probe(void) {
    enum { B = 2, C = 4, H = 6, W = 8, ph = 2, pw = 2, D = 9 };
    enum { Hp = H / ph, Wp = W / pw, T = Hp * Wp, out_dim = C * ph * pw };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer tokens_buffer = VK_NULL_HANDLE;
    VkBuffer weight_buffer = VK_NULL_HANDLE;
    VkBuffer bias_buffer = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tokens_memory = VK_NULL_HANDLE;
    VkDeviceMemory weight_memory = VK_NULL_HANDLE;
    VkDeviceMemory bias_memory = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    void *tokens_mapped = NULL;
    void *weight_mapped = NULL;
    void *bias_mapped = NULL;
    void *x_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t tokens_bytes = (size_t)B * T * D * sizeof(float);
    size_t weight_bytes = (size_t)out_dim * D * sizeof(float);
    size_t bias_bytes = (size_t)out_dim * sizeof(float);
    size_t x_bytes = (size_t)B * C * H * W * sizeof(float);
    if (create_host_buffer(rt, tokens_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tokens_buffer, &tokens_memory, &tokens_mapped)) goto cleanup;
    if (create_host_buffer(rt, weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &weight_buffer, &weight_memory, &weight_mapped)) goto cleanup;
    if (create_host_buffer(rt, bias_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bias_buffer, &bias_memory, &bias_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;

    float *tokens = (float *)tokens_mapped;
    float *weight = (float *)weight_mapped;
    float *bias = (float *)bias_mapped;
    float *x = (float *)x_mapped;
    for (int i = 0; i < B * T * D; ++i) tokens[i] = probe_value(i + 2003, 0.029296875f);
    for (int i = 0; i < out_dim * D; ++i) weight[i] = probe_value(i + 2111, 0.025390625f);
    for (int i = 0; i < out_dim; ++i) bias[i] = probe_value(i + 2203, 0.017578125f);
    memset(x, 0, x_bytes);

    if (create_storage_pipeline(rt, "unpatchify_f32.comp", 4, sizeof(WorldVulkanUnpatchifyPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[4] = {tokens_buffer, weight_buffer, bias_buffer, x_buffer};
    VkDeviceSize sizes[4] = {tokens_bytes, weight_bytes, bias_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 4, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanUnpatchifyPush push;
    push.B = B;
    push.T = T;
    push.D = D;
    push.C = C;
    push.H = H;
    push.W = W;
    push.ph = ph;
    push.pw = pw;
    push.Hp = Hp;
    push.Wp = Wp;
    push.out_dim = out_dim;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * T * out_dim, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int token = 0; token < T; ++token) {
            int oy = token / Wp;
            int ox = token - oy * Wp;
            for (int o = 0; o < out_dim; ++o) {
                float ref = bias[o];
                for (int d = 0; d < D; ++d) {
                    ref += tokens[(b * T + token) * D + d] * weight[o * D + d];
                }
                int p = o;
                int dx = p % pw;
                p /= pw;
                int dy = p % ph;
                p /= ph;
                int c = p;
                int iy = oy * ph + dy;
                int ix = ox * pw + dx;
                size_t idx = (size_t)(((b * C + c) * H + iy) * W + ix);
                float diff = fabsf(x[idx] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(B * C * H * W);
    fprintf(stderr, "vulkan unpatchify_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 3.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, tokens_buffer, tokens_memory, tokens_mapped);
        destroy_host_buffer(rt, weight_buffer, weight_memory, weight_mapped);
        destroy_host_buffer(rt, bias_buffer, bias_memory, bias_mapped);
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_indexed_attention_f32_probe(void) {
    enum { B = 1, Hq = 8, Hkv = 2, Tq = 9, Tk = 17, D = 64, Nkv = 5 };
    const float scale = 1.0f / sqrtf((float)D);
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule d64_shader = VK_NULL_HANDLE;
    VkPipelineLayout d64_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline d64_pipeline = VK_NULL_HANDLE;
    VkShaderModule flash_shader = VK_NULL_HANDLE;
    VkPipelineLayout flash_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline flash_pipeline = VK_NULL_HANDLE;
    VkShaderModule flash96_shader = VK_NULL_HANDLE;
    VkPipelineLayout flash96_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline flash96_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer q_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkBuffer indices_buffer = VK_NULL_HANDLE;
    VkBuffer out_buffer = VK_NULL_HANDLE;
    VkBuffer count_buffer = VK_NULL_HANDLE;
    VkDeviceMemory q_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    VkDeviceMemory indices_memory = VK_NULL_HANDLE;
    VkDeviceMemory out_memory = VK_NULL_HANDLE;
    VkDeviceMemory count_memory = VK_NULL_HANDLE;
    void *q_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;
    void *indices_mapped = NULL;
    void *out_mapped = NULL;
    void *count_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t q_bytes = (size_t)B * Hq * Tq * D * sizeof(float);
    size_t kv_bytes = (size_t)B * Hkv * Tk * D * sizeof(float);
    size_t indices_bytes = (size_t)Nkv * sizeof(uint32_t);
    size_t count_bytes = sizeof(uint32_t);
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &q_buffer, &q_memory, &q_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v_buffer, &v_memory, &v_mapped)) goto cleanup;
    if (create_host_buffer(rt, indices_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &indices_buffer, &indices_memory, &indices_mapped)) goto cleanup;
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) goto cleanup;
    if (create_host_buffer(rt, count_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &count_buffer, &count_memory, &count_mapped)) goto cleanup;

    float *q = (float *)q_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    uint32_t *indices = (uint32_t *)indices_mapped;
    float *out = (float *)out_mapped;
    uint32_t *count = (uint32_t *)count_mapped;
    for (int i = 0; i < B * Hq * Tq * D; ++i) q[i] = probe_value(i + 2309, 0.021484375f);
    for (int i = 0; i < B * Hkv * Tk * D; ++i) {
        k[i] = probe_value(i + 2411, 0.01953125f);
        v[i] = probe_value(i + 2503, 0.02734375f);
    }
    const uint32_t indices_ref[Nkv] = {0u, 3u, 4u, 9u, 16u};
    memcpy(indices, indices_ref, indices_bytes);
    count[0] = Nkv;
    memset(out, 0, q_bytes);

    if (create_storage_pipeline(rt, "indexed_attention_f32.comp", 6, sizeof(WorldVulkanIndexedAttentionPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[6] = {q_buffer, k_buffer, v_buffer, indices_buffer, out_buffer, count_buffer};
    VkDeviceSize sizes[6] = {q_bytes, kv_bytes, kv_bytes, indices_bytes, q_bytes, count_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 6, buffers, sizes, NULL, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanIndexedAttentionPush push;
    push.B = B;
    push.Hq = Hq;
    push.Hkv = Hkv;
    push.Tq = Tq;
    push.Nkv = Nkv;
    push.Tk = Tk;
    push.D = D;
    push.scale = scale;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * Hq * Tq, 1, 1)) goto cleanup;

    int group = Hq / Hkv;
#define CHECK_INDEXED_ATTENTION(label) do { \
        float max_abs = 0.0f; \
        float mean_abs = 0.0f; \
        for (int b = 0; b < B; ++b) { \
            for (int hq = 0; hq < Hq; ++hq) { \
                int hk = hq / group; \
                for (int tq = 0; tq < Tq; ++tq) { \
                    const float *qrow = q + ((b * Hq + hq) * Tq + tq) * D; \
                    const float *kbase = k + (b * Hkv + hk) * Tk * D; \
                    const float *vbase = v + (b * Hkv + hk) * Tk * D; \
                    float scores[Nkv]; \
                    float max_score = -INFINITY; \
                    for (int n = 0; n < Nkv; ++n) { \
                        int tk = (int)indices[n]; \
                        float dot = 0.0f; \
                        for (int d = 0; d < D; ++d) dot += qrow[d] * kbase[tk * D + d]; \
                        scores[n] = dot * scale; \
                        if (scores[n] > max_score) max_score = scores[n]; \
                    } \
                    float denom = 0.0f; \
                    for (int n = 0; n < Nkv; ++n) denom += expf(scores[n] - max_score); \
                    for (int d = 0; d < D; ++d) { \
                        float ref = 0.0f; \
                        for (int n = 0; n < Nkv; ++n) { \
                            int tk = (int)indices[n]; \
                            ref += expf(scores[n] - max_score) * vbase[tk * D + d]; \
                        } \
                        ref /= denom; \
                        size_t idx = (size_t)(((b * Tq + tq) * Hq + hq) * D + d); \
                        float diff = fabsf(out[idx] - ref); \
                        if (diff > max_abs) max_abs = diff; \
                        mean_abs += diff; \
                    } \
                } \
            } \
        } \
        mean_abs /= (float)(B * Hq * Tq * D); \
        fprintf(stderr, "vulkan %s probe: max_abs=%g mean_abs=%g\n", (label), max_abs, mean_abs); \
        if (max_abs > 4.0e-5f) goto cleanup; \
    } while (0)

    CHECK_INDEXED_ATTENTION("indexed_attention_f32");

    if (create_storage_pipeline_with_set_layout(rt, "indexed_attention_d64_warp_f32.comp",
                set_layout, sizeof(WorldVulkanIndexedAttentionPush),
                &d64_shader, &d64_pipeline_layout, &d64_pipeline)) goto cleanup;
    if (submit_compute(rt, d64_pipeline, d64_pipeline_layout, descriptor_set, &push, sizeof(push),
                B * Hq * Tq, 1, 1)) goto cleanup;
    CHECK_INDEXED_ATTENTION("indexed_attention_d64_warp_f32");

    if (create_storage_pipeline_with_set_layout(rt, "indexed_attention_d64_flash_f32.comp",
                set_layout, sizeof(WorldVulkanIndexedAttentionPush),
                &flash_shader, &flash_pipeline_layout, &flash_pipeline)) goto cleanup;
    int q_per_h = 16 / group;
    if (q_per_h < 1) q_per_h = 1;
    int q_blocks = (Tq + q_per_h - 1) / q_per_h;
    if (submit_compute(rt, flash_pipeline, flash_pipeline_layout, descriptor_set, &push, sizeof(push),
                Hkv * q_blocks, 1, 1)) goto cleanup;
    CHECK_INDEXED_ATTENTION("indexed_attention_d64_flash_f32");

    if (rt->max_compute_shared_memory_size >= 96u * 64u * 2u * (uint32_t)sizeof(float)) {
        if (create_storage_pipeline_with_set_layout(rt, "indexed_attention_d64_flash96_f32.comp",
                    set_layout, sizeof(WorldVulkanIndexedAttentionPush),
                    &flash96_shader, &flash96_pipeline_layout, &flash96_pipeline)) goto cleanup;
        if (submit_compute(rt, flash96_pipeline, flash96_pipeline_layout, descriptor_set, &push, sizeof(push),
                    Hkv * q_blocks, 1, 1)) goto cleanup;
        CHECK_INDEXED_ATTENTION("indexed_attention_d64_flash96_f32");
    } else {
        fprintf(stderr, "vulkan indexed_attention_d64_flash96_f32 probe: skipped; maxComputeSharedMemorySize=%u\n",
                rt->max_compute_shared_memory_size);
    }
#undef CHECK_INDEXED_ATTENTION
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (flash96_pipeline) vkDestroyPipeline(rt->device, flash96_pipeline, NULL);
    if (flash96_pipeline_layout) vkDestroyPipelineLayout(rt->device, flash96_pipeline_layout, NULL);
    if (flash96_shader) vkDestroyShaderModule(rt->device, flash96_shader, NULL);
    if (flash_pipeline) vkDestroyPipeline(rt->device, flash_pipeline, NULL);
    if (flash_pipeline_layout) vkDestroyPipelineLayout(rt->device, flash_pipeline_layout, NULL);
    if (flash_shader) vkDestroyShaderModule(rt->device, flash_shader, NULL);
    if (d64_pipeline) vkDestroyPipeline(rt->device, d64_pipeline, NULL);
    if (d64_pipeline_layout) vkDestroyPipelineLayout(rt->device, d64_pipeline_layout, NULL);
    if (d64_shader) vkDestroyShaderModule(rt->device, d64_shader, NULL);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, q_buffer, q_memory, q_mapped);
        destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
        destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
        destroy_host_buffer(rt, indices_buffer, indices_memory, indices_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
        destroy_host_buffer(rt, count_buffer, count_memory, count_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

void world_vulkan_runtime_destroy(WorldVulkanRuntime *rt) {
    if (!rt) return;
    if (rt->device) vkDeviceWaitIdle(rt->device);
    if (rt->profile_query_pool) vkDestroyQueryPool(rt->device, rt->profile_query_pool, NULL);
    free(rt->profile_query_values);
    free(rt->profile_events);
    if (rt->patchify_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->patchify_descriptor_pool, NULL);
    if (rt->unpatch_orig_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->unpatch_orig_descriptor_pool, NULL);
    if (rt->latent_rgba_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->latent_rgba_descriptor_pool, NULL);
    for (int i = 0; i < rt->vae_op_count; ++i) {
        if (rt->vae_ops[i].descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->vae_ops[i].descriptor_pool, NULL);
    }
    if (rt->ctrl_fc1_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->ctrl_fc1_descriptor_pool, NULL);
    if (rt->ctrl_silu_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->ctrl_silu_descriptor_pool, NULL);
    if (rt->ctrl_fc2_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->ctrl_fc2_descriptor_pool, NULL);
    if (rt->ctrl_rms_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->ctrl_rms_descriptor_pool, NULL);
    if (rt->denoise_fc1_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->denoise_fc1_descriptor_pool, NULL);
    if (rt->denoise_silu_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->denoise_silu_descriptor_pool, NULL);
    if (rt->denoise_fc2_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->denoise_fc2_descriptor_pool, NULL);
    if (rt->denoise_cond_silu_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->denoise_cond_silu_descriptor_pool, NULL);
    for (int i = 0; i < WORLD_VULKAN_MAX_PASSES; ++i) {
        if (rt->out_norm_descriptor_pool[i]) vkDestroyDescriptorPool(rt->device, rt->out_norm_descriptor_pool[i], NULL);
    }
    if (rt->layer_bias_silu_descriptor_pools) {
        for (int i = 0; i < rt->layers_to_run; ++i) {
            if (rt->layer_bias_silu_descriptor_pools[i]) {
                vkDestroyDescriptorPool(rt->device, rt->layer_bias_silu_descriptor_pools[i], NULL);
            }
        }
    }
    if (rt->layer_mod_descriptor_pools) {
        for (int i = 0; i < rt->total_passes * rt->layers_to_run; ++i) {
            if (rt->layer_mod_descriptor_pools[i]) {
                vkDestroyDescriptorPool(rt->device, rt->layer_mod_descriptor_pools[i], NULL);
            }
        }
    }
    if (rt->attn_ada_descriptor_pools) {
        for (int i = 0; i < rt->total_passes * rt->layers_to_run; ++i) {
            if (rt->attn_ada_descriptor_pools[i]) {
                vkDestroyDescriptorPool(rt->device, rt->attn_ada_descriptor_pools[i], NULL);
            }
        }
    }
    if (rt->qkv_proj_descriptor_pools) {
        for (int i = 0; i < rt->layers_to_run; ++i) {
            if (rt->qkv_proj_descriptor_pools[i]) {
                vkDestroyDescriptorPool(rt->device, rt->qkv_proj_descriptor_pools[i], NULL);
            }
        }
    }
    if (rt->qkv_proj_wf16_descriptor_pools) {
        for (int i = 0; i < rt->layers_to_run; ++i) {
            if (rt->qkv_proj_wf16_descriptor_pools[i]) {
                vkDestroyDescriptorPool(rt->device, rt->qkv_proj_wf16_descriptor_pools[i], NULL);
            }
        }
    }
    if (rt->qkv_rms_rope_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->qkv_rms_rope_descriptor_pool, NULL);
    if (rt->qkv_rms_rope_first_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->qkv_rms_rope_first_descriptor_pool, NULL);
    for (int i = 0; i < rt->layers_to_run; ++i) {
        if (rt->kv_upsert_descriptor_pools && rt->kv_upsert_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->kv_upsert_descriptor_pools[i], NULL);
        if (rt->cache_indices_descriptor_pools && rt->cache_indices_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->cache_indices_descriptor_pools[i], NULL);
        if (rt->indexed_attention_descriptor_pools && rt->indexed_attention_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->indexed_attention_descriptor_pools[i], NULL);
        if (rt->attn_out_proj_descriptor_pools && rt->attn_out_proj_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->attn_out_proj_descriptor_pools[i], NULL);
        if (rt->attn_out_proj_wf16_descriptor_pools && rt->attn_out_proj_wf16_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->attn_out_proj_wf16_descriptor_pools[i], NULL);
        if (rt->ctrl_cond_descriptor_pools && rt->ctrl_cond_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->ctrl_cond_descriptor_pools[i], NULL);
        if (rt->ctrl_fc1_x_descriptor_pools && rt->ctrl_fc1_x_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->ctrl_fc1_x_descriptor_pools[i], NULL);
        if (rt->ctrl_add_silu_descriptor_pools && rt->ctrl_add_silu_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->ctrl_add_silu_descriptor_pools[i], NULL);
        if (rt->ctrl_fc2_descriptor_pools_layer && rt->ctrl_fc2_descriptor_pools_layer[i]) vkDestroyDescriptorPool(rt->device, rt->ctrl_fc2_descriptor_pools_layer[i], NULL);
        if (rt->mlp_fc1_descriptor_pools && rt->mlp_fc1_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_fc1_descriptor_pools[i], NULL);
        if (rt->mlp_fc1_wf16_descriptor_pools && rt->mlp_fc1_wf16_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_fc1_wf16_descriptor_pools[i], NULL);
        if (rt->mlp_fc2_descriptor_pools && rt->mlp_fc2_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_fc2_descriptor_pools[i], NULL);
        if (rt->mlp_fc2_wf16_descriptor_pools && rt->mlp_fc2_wf16_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_fc2_wf16_descriptor_pools[i], NULL);
        if (rt->mlp_fc2_f16x_descriptor_pools && rt->mlp_fc2_f16x_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_fc2_f16x_descriptor_pools[i], NULL);
    }
    for (int i = 0; i < rt->total_passes * rt->layers_to_run; ++i) {
        if (rt->attn_residual_descriptor_pools && rt->attn_residual_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->attn_residual_descriptor_pools[i], NULL);
        if (rt->mlp_ada_descriptor_pools && rt->mlp_ada_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_ada_descriptor_pools[i], NULL);
        if (rt->mlp_residual_descriptor_pools && rt->mlp_residual_descriptor_pools[i]) vkDestroyDescriptorPool(rt->device, rt->mlp_residual_descriptor_pools[i], NULL);
    }
    if (rt->ctrl_norm_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->ctrl_norm_descriptor_pool, NULL);
    if (rt->ctrl_add_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->ctrl_add_descriptor_pool, NULL);
    if (rt->mlp_silu_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->mlp_silu_descriptor_pool, NULL);
    if (rt->mlp_silu_f16_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->mlp_silu_f16_descriptor_pool, NULL);
    for (int i = 0; i < WORLD_VULKAN_MAX_PASSES; ++i) {
        if (rt->out_norm_silu_descriptor_pool[i]) vkDestroyDescriptorPool(rt->device, rt->out_norm_silu_descriptor_pool[i], NULL);
    }
    if (rt->latent_update_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->latent_update_descriptor_pool, NULL);
    if (rt->value_residual_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->value_residual_descriptor_pool, NULL);
    if (rt->runtime_linear_pipeline) vkDestroyPipeline(rt->device, rt->runtime_linear_pipeline, NULL);
    if (rt->runtime_linear_coopmat_pipeline) vkDestroyPipeline(rt->device, rt->runtime_linear_coopmat_pipeline, NULL);
    if (rt->runtime_linear_wf16_coopmat_pipeline) vkDestroyPipeline(rt->device, rt->runtime_linear_wf16_coopmat_pipeline, NULL);
    if (rt->runtime_linear_wf16_coopmat_n16_pipeline) vkDestroyPipeline(rt->device, rt->runtime_linear_wf16_coopmat_n16_pipeline, NULL);
    if (rt->runtime_linear_f16x_wf16_coopmat_pipeline) vkDestroyPipeline(rt->device, rt->runtime_linear_f16x_wf16_coopmat_pipeline, NULL);
    if (rt->runtime_linear_tile64_pipeline) vkDestroyPipeline(rt->device, rt->runtime_linear_tile64_pipeline, NULL);
    if (rt->runtime_silu_pipeline) vkDestroyPipeline(rt->device, rt->runtime_silu_pipeline, NULL);
    if (rt->runtime_silu_f16_pipeline) vkDestroyPipeline(rt->device, rt->runtime_silu_f16_pipeline, NULL);
    if (rt->runtime_add_bias_silu_pipeline) vkDestroyPipeline(rt->device, rt->runtime_add_bias_silu_pipeline, NULL);
    if (rt->runtime_rms_pipeline) vkDestroyPipeline(rt->device, rt->runtime_rms_pipeline, NULL);
    if (rt->runtime_ada_rms_pipeline) vkDestroyPipeline(rt->device, rt->runtime_ada_rms_pipeline, NULL);
    if (rt->runtime_qkv_rms_rope_pipeline) vkDestroyPipeline(rt->device, rt->runtime_qkv_rms_rope_pipeline, NULL);
    if (rt->runtime_kv_upsert_pipeline) vkDestroyPipeline(rt->device, rt->runtime_kv_upsert_pipeline, NULL);
    if (rt->runtime_cache_indices_pipeline) vkDestroyPipeline(rt->device, rt->runtime_cache_indices_pipeline, NULL);
    if (rt->runtime_indexed_attention_pipeline) vkDestroyPipeline(rt->device, rt->runtime_indexed_attention_pipeline, NULL);
    if (rt->runtime_indexed_attention_d64_pipeline) vkDestroyPipeline(rt->device, rt->runtime_indexed_attention_d64_pipeline, NULL);
    if (rt->runtime_indexed_attention_d64_flash_pipeline) vkDestroyPipeline(rt->device, rt->runtime_indexed_attention_d64_flash_pipeline, NULL);
    if (rt->runtime_indexed_attention_d64_flash96_pipeline) vkDestroyPipeline(rt->device, rt->runtime_indexed_attention_d64_flash96_pipeline, NULL);
    if (rt->runtime_gated_residual_pipeline) vkDestroyPipeline(rt->device, rt->runtime_gated_residual_pipeline, NULL);
    if (rt->runtime_add_channel_silu_pipeline) vkDestroyPipeline(rt->device, rt->runtime_add_channel_silu_pipeline, NULL);
    if (rt->runtime_add_pipeline) vkDestroyPipeline(rt->device, rt->runtime_add_pipeline, NULL);
    if (rt->runtime_out_norm_silu_pipeline) vkDestroyPipeline(rt->device, rt->runtime_out_norm_silu_pipeline, NULL);
    if (rt->runtime_latent_update_pipeline) vkDestroyPipeline(rt->device, rt->runtime_latent_update_pipeline, NULL);
    if (rt->runtime_lerp_pipeline) vkDestroyPipeline(rt->device, rt->runtime_lerp_pipeline, NULL);
    if (rt->patchify_pipeline) vkDestroyPipeline(rt->device, rt->patchify_pipeline, NULL);
    if (rt->unpatch_orig_pipeline) vkDestroyPipeline(rt->device, rt->unpatch_orig_pipeline, NULL);
    if (rt->latent_rgba_pipeline) vkDestroyPipeline(rt->device, rt->latent_rgba_pipeline, NULL);
    if (rt->taehv_repeat_clamp_pipeline) vkDestroyPipeline(rt->device, rt->taehv_repeat_clamp_pipeline, NULL);
    if (rt->taehv_conv_pipeline) vkDestroyPipeline(rt->device, rt->taehv_conv_pipeline, NULL);
    if (rt->taehv_conv_c4_pipeline) vkDestroyPipeline(rt->device, rt->taehv_conv_c4_pipeline, NULL);
    if (rt->taehv_conv_1x1_c4_pipeline) vkDestroyPipeline(rt->device, rt->taehv_conv_1x1_c4_pipeline, NULL);
    if (rt->taehv_conv_out_rgba_pipeline) vkDestroyPipeline(rt->device, rt->taehv_conv_out_rgba_pipeline, NULL);
    if (rt->taehv_relu_pipeline) vkDestroyPipeline(rt->device, rt->taehv_relu_pipeline, NULL);
    if (rt->taehv_concat_past_pipeline) vkDestroyPipeline(rt->device, rt->taehv_concat_past_pipeline, NULL);
    if (rt->taehv_add_relu_pipeline) vkDestroyPipeline(rt->device, rt->taehv_add_relu_pipeline, NULL);
    if (rt->taehv_upsample2_pipeline) vkDestroyPipeline(rt->device, rt->taehv_upsample2_pipeline, NULL);
    if (rt->taehv_tgrow_pipeline) vkDestroyPipeline(rt->device, rt->taehv_tgrow_pipeline, NULL);
    if (rt->taehv_pixel_shuffle_pipeline) vkDestroyPipeline(rt->device, rt->taehv_pixel_shuffle_pipeline, NULL);
    if (rt->runtime_linear_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_linear_pipeline_layout, NULL);
    if (rt->runtime_linear_coopmat_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_linear_coopmat_pipeline_layout, NULL);
    if (rt->runtime_linear_wf16_coopmat_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_linear_wf16_coopmat_pipeline_layout, NULL);
    if (rt->runtime_linear_wf16_coopmat_n16_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_linear_wf16_coopmat_n16_pipeline_layout, NULL);
    if (rt->runtime_linear_f16x_wf16_coopmat_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_linear_f16x_wf16_coopmat_pipeline_layout, NULL);
    if (rt->runtime_linear_tile64_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_linear_tile64_pipeline_layout, NULL);
    if (rt->runtime_silu_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_silu_pipeline_layout, NULL);
    if (rt->runtime_silu_f16_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_silu_f16_pipeline_layout, NULL);
    if (rt->runtime_add_bias_silu_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_add_bias_silu_pipeline_layout, NULL);
    if (rt->runtime_rms_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_rms_pipeline_layout, NULL);
    if (rt->runtime_ada_rms_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_ada_rms_pipeline_layout, NULL);
    if (rt->runtime_qkv_rms_rope_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_qkv_rms_rope_pipeline_layout, NULL);
    if (rt->runtime_kv_upsert_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_kv_upsert_pipeline_layout, NULL);
    if (rt->runtime_cache_indices_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_cache_indices_pipeline_layout, NULL);
    if (rt->runtime_indexed_attention_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_indexed_attention_pipeline_layout, NULL);
    if (rt->runtime_indexed_attention_d64_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_indexed_attention_d64_pipeline_layout, NULL);
    if (rt->runtime_indexed_attention_d64_flash_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_indexed_attention_d64_flash_pipeline_layout, NULL);
    if (rt->runtime_indexed_attention_d64_flash96_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_indexed_attention_d64_flash96_pipeline_layout, NULL);
    if (rt->runtime_gated_residual_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_gated_residual_pipeline_layout, NULL);
    if (rt->runtime_add_channel_silu_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_add_channel_silu_pipeline_layout, NULL);
    if (rt->runtime_add_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_add_pipeline_layout, NULL);
    if (rt->runtime_out_norm_silu_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_out_norm_silu_pipeline_layout, NULL);
    if (rt->runtime_latent_update_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_latent_update_pipeline_layout, NULL);
    if (rt->runtime_lerp_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->runtime_lerp_pipeline_layout, NULL);
    if (rt->patchify_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->patchify_pipeline_layout, NULL);
    if (rt->unpatch_orig_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->unpatch_orig_pipeline_layout, NULL);
    if (rt->latent_rgba_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->latent_rgba_pipeline_layout, NULL);
    if (rt->taehv_repeat_clamp_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_repeat_clamp_pipeline_layout, NULL);
    if (rt->taehv_conv_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_conv_pipeline_layout, NULL);
    if (rt->taehv_conv_c4_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_conv_c4_pipeline_layout, NULL);
    if (rt->taehv_conv_1x1_c4_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_conv_1x1_c4_pipeline_layout, NULL);
    if (rt->taehv_conv_out_rgba_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_conv_out_rgba_pipeline_layout, NULL);
    if (rt->taehv_relu_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_relu_pipeline_layout, NULL);
    if (rt->taehv_concat_past_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_concat_past_pipeline_layout, NULL);
    if (rt->taehv_add_relu_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_add_relu_pipeline_layout, NULL);
    if (rt->taehv_upsample2_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_upsample2_pipeline_layout, NULL);
    if (rt->taehv_tgrow_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_tgrow_pipeline_layout, NULL);
    if (rt->taehv_pixel_shuffle_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->taehv_pixel_shuffle_pipeline_layout, NULL);
    if (rt->runtime_linear_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_linear_set_layout, NULL);
    if (rt->runtime_silu_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_silu_set_layout, NULL);
    if (rt->runtime_silu_f16_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_silu_f16_set_layout, NULL);
    if (rt->runtime_add_bias_silu_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_add_bias_silu_set_layout, NULL);
    if (rt->runtime_rms_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_rms_set_layout, NULL);
    if (rt->runtime_ada_rms_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_ada_rms_set_layout, NULL);
    if (rt->runtime_qkv_rms_rope_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_qkv_rms_rope_set_layout, NULL);
    if (rt->runtime_kv_upsert_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_kv_upsert_set_layout, NULL);
    if (rt->runtime_cache_indices_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_cache_indices_set_layout, NULL);
    if (rt->runtime_indexed_attention_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_indexed_attention_set_layout, NULL);
    if (rt->runtime_gated_residual_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_gated_residual_set_layout, NULL);
    if (rt->runtime_add_channel_silu_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_add_channel_silu_set_layout, NULL);
    if (rt->runtime_add_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_add_set_layout, NULL);
    if (rt->runtime_out_norm_silu_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_out_norm_silu_set_layout, NULL);
    if (rt->runtime_latent_update_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_latent_update_set_layout, NULL);
    if (rt->runtime_lerp_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->runtime_lerp_set_layout, NULL);
    if (rt->patchify_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->patchify_set_layout, NULL);
    if (rt->unpatch_orig_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->unpatch_orig_set_layout, NULL);
    if (rt->latent_rgba_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->latent_rgba_set_layout, NULL);
    if (rt->taehv_repeat_clamp_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_repeat_clamp_set_layout, NULL);
    if (rt->taehv_conv_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_conv_set_layout, NULL);
    if (rt->taehv_conv_c4_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_conv_c4_set_layout, NULL);
    if (rt->taehv_conv_1x1_c4_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_conv_1x1_c4_set_layout, NULL);
    if (rt->taehv_conv_out_rgba_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_conv_out_rgba_set_layout, NULL);
    if (rt->taehv_relu_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_relu_set_layout, NULL);
    if (rt->taehv_concat_past_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_concat_past_set_layout, NULL);
    if (rt->taehv_add_relu_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_add_relu_set_layout, NULL);
    if (rt->taehv_upsample2_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_upsample2_set_layout, NULL);
    if (rt->taehv_tgrow_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_tgrow_set_layout, NULL);
    if (rt->taehv_pixel_shuffle_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->taehv_pixel_shuffle_set_layout, NULL);
    if (rt->runtime_linear_shader) vkDestroyShaderModule(rt->device, rt->runtime_linear_shader, NULL);
    if (rt->runtime_linear_coopmat_shader) vkDestroyShaderModule(rt->device, rt->runtime_linear_coopmat_shader, NULL);
    if (rt->runtime_linear_wf16_coopmat_shader) vkDestroyShaderModule(rt->device, rt->runtime_linear_wf16_coopmat_shader, NULL);
    if (rt->runtime_linear_wf16_coopmat_n16_shader) vkDestroyShaderModule(rt->device, rt->runtime_linear_wf16_coopmat_n16_shader, NULL);
    if (rt->runtime_linear_f16x_wf16_coopmat_shader) vkDestroyShaderModule(rt->device, rt->runtime_linear_f16x_wf16_coopmat_shader, NULL);
    if (rt->runtime_linear_tile64_shader) vkDestroyShaderModule(rt->device, rt->runtime_linear_tile64_shader, NULL);
    if (rt->runtime_silu_shader) vkDestroyShaderModule(rt->device, rt->runtime_silu_shader, NULL);
    if (rt->runtime_silu_f16_shader) vkDestroyShaderModule(rt->device, rt->runtime_silu_f16_shader, NULL);
    if (rt->runtime_add_bias_silu_shader) vkDestroyShaderModule(rt->device, rt->runtime_add_bias_silu_shader, NULL);
    if (rt->runtime_rms_shader) vkDestroyShaderModule(rt->device, rt->runtime_rms_shader, NULL);
    if (rt->runtime_ada_rms_shader) vkDestroyShaderModule(rt->device, rt->runtime_ada_rms_shader, NULL);
    if (rt->runtime_qkv_rms_rope_shader) vkDestroyShaderModule(rt->device, rt->runtime_qkv_rms_rope_shader, NULL);
    if (rt->runtime_kv_upsert_shader) vkDestroyShaderModule(rt->device, rt->runtime_kv_upsert_shader, NULL);
    if (rt->runtime_cache_indices_shader) vkDestroyShaderModule(rt->device, rt->runtime_cache_indices_shader, NULL);
    if (rt->runtime_indexed_attention_shader) vkDestroyShaderModule(rt->device, rt->runtime_indexed_attention_shader, NULL);
    if (rt->runtime_indexed_attention_d64_shader) vkDestroyShaderModule(rt->device, rt->runtime_indexed_attention_d64_shader, NULL);
    if (rt->runtime_indexed_attention_d64_flash_shader) vkDestroyShaderModule(rt->device, rt->runtime_indexed_attention_d64_flash_shader, NULL);
    if (rt->runtime_indexed_attention_d64_flash96_shader) vkDestroyShaderModule(rt->device, rt->runtime_indexed_attention_d64_flash96_shader, NULL);
    if (rt->runtime_gated_residual_shader) vkDestroyShaderModule(rt->device, rt->runtime_gated_residual_shader, NULL);
    if (rt->runtime_add_channel_silu_shader) vkDestroyShaderModule(rt->device, rt->runtime_add_channel_silu_shader, NULL);
    if (rt->runtime_add_shader) vkDestroyShaderModule(rt->device, rt->runtime_add_shader, NULL);
    if (rt->runtime_out_norm_silu_shader) vkDestroyShaderModule(rt->device, rt->runtime_out_norm_silu_shader, NULL);
    if (rt->runtime_latent_update_shader) vkDestroyShaderModule(rt->device, rt->runtime_latent_update_shader, NULL);
    if (rt->runtime_lerp_shader) vkDestroyShaderModule(rt->device, rt->runtime_lerp_shader, NULL);
    if (rt->patchify_shader) vkDestroyShaderModule(rt->device, rt->patchify_shader, NULL);
    if (rt->unpatch_orig_shader) vkDestroyShaderModule(rt->device, rt->unpatch_orig_shader, NULL);
    if (rt->latent_rgba_shader) vkDestroyShaderModule(rt->device, rt->latent_rgba_shader, NULL);
    if (rt->taehv_repeat_clamp_shader) vkDestroyShaderModule(rt->device, rt->taehv_repeat_clamp_shader, NULL);
    if (rt->taehv_conv_shader) vkDestroyShaderModule(rt->device, rt->taehv_conv_shader, NULL);
    if (rt->taehv_conv_c4_shader) vkDestroyShaderModule(rt->device, rt->taehv_conv_c4_shader, NULL);
    if (rt->taehv_conv_1x1_c4_shader) vkDestroyShaderModule(rt->device, rt->taehv_conv_1x1_c4_shader, NULL);
    if (rt->taehv_conv_out_rgba_shader) vkDestroyShaderModule(rt->device, rt->taehv_conv_out_rgba_shader, NULL);
    if (rt->taehv_relu_shader) vkDestroyShaderModule(rt->device, rt->taehv_relu_shader, NULL);
    if (rt->taehv_concat_past_shader) vkDestroyShaderModule(rt->device, rt->taehv_concat_past_shader, NULL);
    if (rt->taehv_add_relu_shader) vkDestroyShaderModule(rt->device, rt->taehv_add_relu_shader, NULL);
    if (rt->taehv_upsample2_shader) vkDestroyShaderModule(rt->device, rt->taehv_upsample2_shader, NULL);
    if (rt->taehv_tgrow_shader) vkDestroyShaderModule(rt->device, rt->taehv_tgrow_shader, NULL);
    if (rt->taehv_pixel_shuffle_shader) vkDestroyShaderModule(rt->device, rt->taehv_pixel_shuffle_shader, NULL);
    destroy_host_buffer(rt, rt->vae_buf0_buffer, rt->vae_buf0_memory, rt->vae_buf0_mapped);
    destroy_host_buffer(rt, rt->vae_buf1_buffer, rt->vae_buf1_memory, rt->vae_buf1_mapped);
    destroy_host_buffer(rt, rt->vae_buf2_buffer, rt->vae_buf2_memory, rt->vae_buf2_mapped);
    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
        destroy_host_buffer(rt, rt->vae_weight_buffer[i], rt->vae_weight_memory[i], rt->vae_weight_mapped[i]);
        destroy_host_buffer(rt, rt->vae_bias_buffer[i], rt->vae_bias_memory[i], rt->vae_bias_mapped[i]);
    }
    destroy_host_buffer(rt, rt->latent_buffer, rt->latent_memory, rt->latent_mapped);
    destroy_host_buffer(rt, rt->control_buffer, rt->control_memory, rt->control_mapped);
    destroy_host_buffer(rt, rt->ctrl_fc1_weight_buffer, rt->ctrl_fc1_weight_memory, rt->ctrl_fc1_weight_mapped);
    destroy_host_buffer(rt, rt->ctrl_fc2_weight_buffer, rt->ctrl_fc2_weight_memory, rt->ctrl_fc2_weight_mapped);
    destroy_host_buffer(rt, rt->ctrl_hidden_buffer, rt->ctrl_hidden_memory, rt->ctrl_hidden_mapped);
    destroy_host_buffer(rt, rt->ctrl_emb_buffer, rt->ctrl_emb_memory, rt->ctrl_emb_mapped);
    destroy_host_buffer(rt, rt->ctrl_emb_norm_buffer, rt->ctrl_emb_norm_memory, rt->ctrl_emb_norm_mapped);
    destroy_host_buffer(rt, rt->dummy_bias_buffer, rt->dummy_bias_memory, rt->dummy_bias_mapped);
    destroy_host_buffer(rt, rt->rms_weight_buffer, rt->rms_weight_memory, rt->rms_weight_mapped);
    destroy_host_buffer(rt, rt->noise_buffer, rt->noise_memory, rt->noise_mapped);
    destroy_host_buffer(rt, rt->denoise_fc1_weight_buffer, rt->denoise_fc1_weight_memory, rt->denoise_fc1_weight_mapped);
    destroy_host_buffer(rt, rt->denoise_fc2_weight_buffer, rt->denoise_fc2_weight_memory, rt->denoise_fc2_weight_mapped);
    destroy_host_buffer(rt, rt->noise_hidden_buffer, rt->noise_hidden_memory, rt->noise_hidden_mapped);
    destroy_host_buffer(rt, rt->cond_buffer, rt->cond_memory, rt->cond_mapped);
    destroy_host_buffer(rt, rt->cond_act_buffer, rt->cond_act_memory, rt->cond_act_mapped);
    destroy_host_buffer(rt, rt->out_norm_weight_buffer, rt->out_norm_weight_memory, rt->out_norm_weight_mapped);
    destroy_host_buffer(rt, rt->out_mod_table_buffer, rt->out_mod_table_memory, rt->out_mod_table_mapped);
    destroy_host_buffer(rt, rt->layer_cond_bias_buffer, rt->layer_cond_bias_memory, rt->layer_cond_bias_mapped);
    destroy_host_buffer(rt, rt->layer_cond_proj_weight_buffer, rt->layer_cond_proj_weight_memory, rt->layer_cond_proj_weight_mapped);
    destroy_host_buffer(rt, rt->layer_mod_table_buffer, rt->layer_mod_table_memory, rt->layer_mod_table_mapped);
    destroy_host_buffer(rt, rt->qkv_proj_weight_buffer, rt->qkv_proj_weight_memory, rt->qkv_proj_weight_mapped);
    destroy_host_buffer(rt, rt->qkv_proj_weight_f16_buffer, rt->qkv_proj_weight_f16_memory, NULL);
    destroy_host_buffer(rt, rt->patch_weight_buffer, rt->patch_weight_memory, rt->patch_weight_mapped);
    destroy_host_buffer(rt, rt->tokens_buffer, rt->tokens_memory, rt->tokens_mapped);
    destroy_host_buffer(rt, rt->norm_buffer, rt->norm_memory, rt->norm_mapped);
    destroy_host_buffer(rt, rt->qkv_raw_buffer, rt->qkv_raw_memory, rt->qkv_raw_mapped);
    destroy_host_buffer(rt, rt->q_buffer, rt->q_memory, rt->q_mapped);
    destroy_host_buffer(rt, rt->k_buffer, rt->k_memory, rt->k_mapped);
    destroy_host_buffer(rt, rt->v_buffer, rt->v_memory, rt->v_mapped);
    destroy_host_buffer(rt, rt->v_first_buffer, rt->v_first_memory, rt->v_first_mapped);
    destroy_host_buffer(rt, rt->x_pos_buffer, rt->x_pos_memory, rt->x_pos_mapped);
    destroy_host_buffer(rt, rt->y_pos_buffer, rt->y_pos_memory, rt->y_pos_mapped);
    destroy_host_buffer(rt, rt->t_pos_buffer, rt->t_pos_memory, rt->t_pos_mapped);
    destroy_host_buffer(rt, rt->xy_buffer, rt->xy_memory, rt->xy_mapped);
    destroy_host_buffer(rt, rt->inv_t_buffer, rt->inv_t_memory, rt->inv_t_mapped);
    destroy_host_buffer(rt, rt->cache_k_buffer, rt->cache_k_memory, rt->cache_k_mapped);
    destroy_host_buffer(rt, rt->cache_v_buffer, rt->cache_v_memory, rt->cache_v_mapped);
    destroy_host_buffer(rt, rt->cache_written_buffer, rt->cache_written_memory, rt->cache_written_mapped);
    destroy_host_buffer(rt, rt->cache_indices_buffer, rt->cache_indices_memory, rt->cache_indices_mapped);
    destroy_host_buffer(rt, rt->cache_index_count_buffer, rt->cache_index_count_memory, rt->cache_index_count_mapped);
    destroy_host_buffer(rt, rt->attn_buffer, rt->attn_memory, rt->attn_mapped);
    destroy_host_buffer(rt, rt->attn_out_proj_weight_buffer, rt->attn_out_proj_weight_memory, rt->attn_out_proj_weight_mapped);
    destroy_host_buffer(rt, rt->attn_out_proj_weight_f16_buffer, rt->attn_out_proj_weight_f16_memory, NULL);
    destroy_host_buffer(rt, rt->attn_proj_buffer, rt->attn_proj_memory, rt->attn_proj_mapped);
    destroy_host_buffer(rt, rt->tokens_after_attn_buffer, rt->tokens_after_attn_memory, rt->tokens_after_attn_mapped);
    destroy_host_buffer(rt, rt->ctrl_fc1_c_weight_buffer, rt->ctrl_fc1_c_weight_memory, rt->ctrl_fc1_c_weight_mapped);
    destroy_host_buffer(rt, rt->ctrl_fc1_x_weight_buffer, rt->ctrl_fc1_x_weight_memory, rt->ctrl_fc1_x_weight_mapped);
    destroy_host_buffer(rt, rt->ctrl_fc2_weight_buffer_layer, rt->ctrl_fc2_weight_memory_layer, rt->ctrl_fc2_weight_mapped_layer);
    destroy_host_buffer(rt, rt->ctrl_cond_buffer, rt->ctrl_cond_memory, rt->ctrl_cond_mapped);
    destroy_host_buffer(rt, rt->ctrl_norm_buffer, rt->ctrl_norm_memory, rt->ctrl_norm_mapped);
    destroy_host_buffer(rt, rt->ctrl_hidden_layer_buffer, rt->ctrl_hidden_layer_memory, rt->ctrl_hidden_layer_mapped);
    destroy_host_buffer(rt, rt->ctrl_out_buffer, rt->ctrl_out_memory, rt->ctrl_out_mapped);
    destroy_host_buffer(rt, rt->tokens_after_ctrl_buffer, rt->tokens_after_ctrl_memory, rt->tokens_after_ctrl_mapped);
    destroy_host_buffer(rt, rt->dit_mlp_fc1_weight_buffer, rt->dit_mlp_fc1_weight_memory, rt->dit_mlp_fc1_weight_mapped);
    destroy_host_buffer(rt, rt->dit_mlp_fc1_weight_f16_buffer, rt->dit_mlp_fc1_weight_f16_memory, NULL);
    destroy_host_buffer(rt, rt->dit_mlp_fc2_weight_buffer, rt->dit_mlp_fc2_weight_memory, rt->dit_mlp_fc2_weight_mapped);
    destroy_host_buffer(rt, rt->dit_mlp_fc2_weight_f16_buffer, rt->dit_mlp_fc2_weight_f16_memory, NULL);
    destroy_host_buffer(rt, rt->mlp_in_buffer, rt->mlp_in_memory, rt->mlp_in_mapped);
    destroy_host_buffer(rt, rt->mlp_hidden_buffer, rt->mlp_hidden_memory, rt->mlp_hidden_mapped);
    destroy_host_buffer(rt, rt->mlp_hidden_f16_buffer, rt->mlp_hidden_f16_memory, NULL);
    destroy_host_buffer(rt, rt->mlp_out_buffer, rt->mlp_out_memory, rt->mlp_out_mapped);
    destroy_host_buffer(rt, rt->tokens_after_mlp_buffer, rt->tokens_after_mlp_memory, rt->tokens_after_mlp_mapped);
    destroy_host_buffer(rt, rt->final_tokens_buffer, rt->final_tokens_memory, rt->final_tokens_mapped);
    destroy_host_buffer(rt, rt->unpatch_weight_buffer, rt->unpatch_weight_memory, rt->unpatch_weight_mapped);
    destroy_host_buffer(rt, rt->unpatch_bias_buffer, rt->unpatch_bias_memory, rt->unpatch_bias_mapped);
    destroy_host_buffer(rt, rt->latent_out_buffer, rt->latent_out_memory, rt->latent_out_mapped);
    if (rt->output_mapped) vkUnmapMemory(rt->device, rt->output_readback_memory);
    if (rt->fence) vkDestroyFence(rt->device, rt->fence, NULL);
    if (rt->command_pool) vkDestroyCommandPool(rt->device, rt->command_pool, NULL);
    if (rt->descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->descriptor_pool, NULL);
    if (rt->fill_pipeline) vkDestroyPipeline(rt->device, rt->fill_pipeline, NULL);
    if (rt->pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->pipeline_layout, NULL);
    if (rt->descriptor_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->descriptor_set_layout, NULL);
    if (rt->fill_shader) vkDestroyShaderModule(rt->device, rt->fill_shader, NULL);
    if (rt->output_readback_buffer) vkDestroyBuffer(rt->device, rt->output_readback_buffer, NULL);
    if (rt->output_readback_memory) vkFreeMemory(rt->device, rt->output_readback_memory, NULL);
    if (rt->output_buffer) vkDestroyBuffer(rt->device, rt->output_buffer, NULL);
    if (rt->output_memory) vkFreeMemory(rt->device, rt->output_memory, NULL);
    if (rt->device) vkDestroyDevice(rt->device, NULL);
    if (rt->instance) vkDestroyInstance(rt->instance, NULL);
    free(rt->rgb_host);
    free(rt->layer_bias_silu_descriptor_pools);
    free(rt->layer_bias_silu_descriptor_sets);
    free(rt->layer_mod_descriptor_pools);
    free(rt->layer_mod_descriptor_sets);
    free(rt->attn_ada_descriptor_pools);
    free(rt->attn_ada_descriptor_sets);
    free(rt->qkv_proj_descriptor_pools);
    free(rt->qkv_proj_descriptor_sets);
    free(rt->qkv_proj_wf16_descriptor_pools);
    free(rt->qkv_proj_wf16_descriptor_sets);
    free(rt->kv_upsert_descriptor_pools);
    free(rt->kv_upsert_descriptor_sets);
    free(rt->cache_indices_descriptor_pools);
    free(rt->cache_indices_descriptor_sets);
    free(rt->indexed_attention_descriptor_pools);
    free(rt->indexed_attention_descriptor_sets);
    free(rt->attn_out_proj_descriptor_pools);
    free(rt->attn_out_proj_descriptor_sets);
    free(rt->attn_out_proj_wf16_descriptor_pools);
    free(rt->attn_out_proj_wf16_descriptor_sets);
    free(rt->attn_residual_descriptor_pools);
    free(rt->attn_residual_descriptor_sets);
    free(rt->ctrl_cond_descriptor_pools);
    free(rt->ctrl_cond_descriptor_sets);
    free(rt->ctrl_fc1_x_descriptor_pools);
    free(rt->ctrl_fc1_x_descriptor_sets);
    free(rt->ctrl_add_silu_descriptor_pools);
    free(rt->ctrl_add_silu_descriptor_sets);
    free(rt->ctrl_fc2_descriptor_pools_layer);
    free(rt->ctrl_fc2_descriptor_sets_layer);
    free(rt->mlp_ada_descriptor_pools);
    free(rt->mlp_ada_descriptor_sets);
    free(rt->mlp_fc1_descriptor_pools);
    free(rt->mlp_fc1_descriptor_sets);
    free(rt->mlp_fc1_wf16_descriptor_pools);
    free(rt->mlp_fc1_wf16_descriptor_sets);
    free(rt->mlp_fc2_descriptor_pools);
    free(rt->mlp_fc2_descriptor_sets);
    free(rt->mlp_fc2_wf16_descriptor_pools);
    free(rt->mlp_fc2_wf16_descriptor_sets);
    free(rt->mlp_fc2_f16x_descriptor_pools);
    free(rt->mlp_fc2_f16x_descriptor_sets);
    free(rt->mlp_residual_descriptor_pools);
    free(rt->mlp_residual_descriptor_sets);
    free(rt->layer_has_ctrl);
    free(rt->layer_v_lamb);
    free(rt->cache_ring_lengths);
    free(rt->cache_capacities);
    free(rt->cache_pinned_dilations);
    free(rt->cache_is_global);
    free(rt->cache_kv_offsets);
    free(rt->cache_meta_offsets);
    free(rt);
}
