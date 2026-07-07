#include "world_config.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}

void world_config_defaults(WorldConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->channels = 32;
    cfg->d_model = 2048;
    cfg->n_heads = 32;
    cfg->n_kv_heads = 16;
    cfg->n_layers = 24;
    cfg->height = 16;
    cfg->width = 32;
    cfg->patch_h = 2;
    cfg->patch_w = 2;
    cfg->n_buttons = 256;
    cfg->mlp_ratio = 4;
    cfg->local_window = 16;
    cfg->global_window = 128;
    cfg->global_pinned_dilation = 8;
    cfg->global_attn_period = 4;
    cfg->global_attn_offset = -1;
    cfg->base_fps = 15;
    cfg->inference_fps = 60;
    cfg->temporal_compression = 4;
    cfg->value_residual = 1;
    cfg->prompt_conditioning = 0;
    cfg->scheduler_sigmas[0] = 1.0f;
    cfg->scheduler_sigmas[1] = 0.9f;
    cfg->scheduler_sigmas[2] = 0.75f;
    cfg->scheduler_sigmas[3] = 0.3f;
    cfg->scheduler_sigmas[4] = 0.0f;
    cfg->scheduler_sigmas_count = 5;
}

static int parse_scalar_int(const char *line, const char *key, int *out) {
    size_t n = strlen(key);
    if (strncmp(line, key, n) != 0 || line[n] != ':') return 0;
    *out = atoi(line + n + 1);
    return 1;
}

static int parse_scalar_bool_null(const char *line, const char *key, int *out) {
    size_t n = strlen(key);
    if (strncmp(line, key, n) != 0 || line[n] != ':') return 0;
    const char *v = line + n + 1;
    while (*v && isspace((unsigned char)*v)) v++;
    *out = (strncmp(v, "null", 4) != 0 && strncmp(v, "false", 5) != 0) ? 1 : 0;
    return 1;
}

int world_config_load(WorldConfig *cfg, const char *path) {
    world_config_defaults(cfg);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open config: %s\n", path);
        return 1;
    }

    char line_buf[512];
    int in_patch = 0;
    int patch_index = 0;
    int in_sigmas = 0;
    cfg->scheduler_sigmas_count = 0;

    while (fgets(line_buf, sizeof(line_buf), f)) {
        char *line = trim(line_buf);
        if (!line[0] || line[0] == '#') continue;

        if (strcmp(line, "patch:") == 0) {
            in_patch = 1;
            patch_index = 0;
            in_sigmas = 0;
            continue;
        }
        if (strcmp(line, "scheduler_sigmas:") == 0) {
            in_sigmas = 1;
            in_patch = 0;
            cfg->scheduler_sigmas_count = 0;
            continue;
        }
        if (line[0] != '-') {
            in_patch = 0;
            in_sigmas = 0;
        }
        if (in_patch && line[0] == '-') {
            int v = atoi(line + 1);
            if (patch_index == 0) cfg->patch_h = v;
            if (patch_index == 1) cfg->patch_w = v;
            patch_index++;
            continue;
        }
        if (in_sigmas && line[0] == '-') {
            if (cfg->scheduler_sigmas_count < (int)(sizeof(cfg->scheduler_sigmas) / sizeof(cfg->scheduler_sigmas[0]))) {
                cfg->scheduler_sigmas[cfg->scheduler_sigmas_count++] = (float)atof(line + 1);
            }
            continue;
        }

        parse_scalar_int(line, "channels", &cfg->channels);
        parse_scalar_int(line, "n_layers", &cfg->n_layers);
        parse_scalar_int(line, "n_heads", &cfg->n_heads);
        parse_scalar_int(line, "n_kv_heads", &cfg->n_kv_heads);
        parse_scalar_int(line, "d_model", &cfg->d_model);
        parse_scalar_int(line, "mlp_ratio", &cfg->mlp_ratio);
        parse_scalar_int(line, "n_buttons", &cfg->n_buttons);
        parse_scalar_int(line, "tokens_per_frame", &(int){0});
        parse_scalar_int(line, "height", &cfg->height);
        parse_scalar_int(line, "width", &cfg->width);
        parse_scalar_int(line, "local_window", &cfg->local_window);
        parse_scalar_int(line, "global_window", &cfg->global_window);
        parse_scalar_int(line, "global_pinned_dilation", &cfg->global_pinned_dilation);
        parse_scalar_int(line, "global_attn_period", &cfg->global_attn_period);
        parse_scalar_int(line, "global_attn_offset", &cfg->global_attn_offset);
        parse_scalar_int(line, "base_fps", &cfg->base_fps);
        parse_scalar_int(line, "inference_fps", &cfg->inference_fps);
        parse_scalar_int(line, "temporal_compression", &cfg->temporal_compression);
        parse_scalar_bool_null(line, "value_residual", &cfg->value_residual);
        parse_scalar_bool_null(line, "prompt_conditioning", &cfg->prompt_conditioning);
    }

    fclose(f);
    if (cfg->scheduler_sigmas_count == 0) {
        cfg->scheduler_sigmas[0] = 1.0f;
        cfg->scheduler_sigmas[1] = 0.0f;
        cfg->scheduler_sigmas_count = 2;
    }
    return 0;
}

void world_config_print(const WorldConfig *cfg) {
    fprintf(stderr,
            "config: C=%d D=%d layers=%d heads=%d kv_heads=%d latent_grid=%dx%d patch=%dx%d\n",
            cfg->channels,
            cfg->d_model,
            cfg->n_layers,
            cfg->n_heads,
            cfg->n_kv_heads,
            cfg->height,
            cfg->width,
            cfg->patch_h,
            cfg->patch_w);
    fprintf(stderr, "scheduler:");
    for (int i = 0; i < cfg->scheduler_sigmas_count; ++i) {
        fprintf(stderr, " %.6g", cfg->scheduler_sigmas[i]);
    }
    fprintf(stderr, "\n");
}
