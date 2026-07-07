#ifndef WORLD_CONFIG_H
#define WORLD_CONFIG_H

typedef struct {
    int channels;
    int d_model;
    int n_heads;
    int n_kv_heads;
    int n_layers;
    int height;
    int width;
    int patch_h;
    int patch_w;
    int n_buttons;
    int mlp_ratio;
    int local_window;
    int global_window;
    int global_pinned_dilation;
    int global_attn_period;
    int global_attn_offset;
    int base_fps;
    int inference_fps;
    int temporal_compression;
    int value_residual;
    int prompt_conditioning;
    float scheduler_sigmas[32];
    int scheduler_sigmas_count;
} WorldConfig;

void world_config_defaults(WorldConfig *cfg);
int world_config_load(WorldConfig *cfg, const char *path);
void world_config_print(const WorldConfig *cfg);

#endif
