#include "world_config.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    const char *key;
    size_t offset;
} ConfigField;

static const ConfigField k_int_fields[] = {
    {"channels", offsetof(WorldConfig, channels)},
    {"n_layers", offsetof(WorldConfig, n_layers)},
    {"n_heads", offsetof(WorldConfig, n_heads)},
    {"n_kv_heads", offsetof(WorldConfig, n_kv_heads)},
    {"d_model", offsetof(WorldConfig, d_model)},
    {"mlp_ratio", offsetof(WorldConfig, mlp_ratio)},
    {"n_buttons", offsetof(WorldConfig, n_buttons)},
    {"height", offsetof(WorldConfig, height)},
    {"width", offsetof(WorldConfig, width)},
    {"local_window", offsetof(WorldConfig, local_window)},
    {"global_window", offsetof(WorldConfig, global_window)},
    {"global_pinned_dilation", offsetof(WorldConfig, global_pinned_dilation)},
    {"global_attn_period", offsetof(WorldConfig, global_attn_period)},
    {"global_attn_offset", offsetof(WorldConfig, global_attn_offset)},
    {"base_fps", offsetof(WorldConfig, base_fps)},
    {"inference_fps", offsetof(WorldConfig, inference_fps)},
    {"temporal_compression", offsetof(WorldConfig, temporal_compression)},
};

static const ConfigField k_bool_fields[] = {
    {"value_residual", offsetof(WorldConfig, value_residual)},
    {"prompt_conditioning", offsetof(WorldConfig, prompt_conditioning)},
};

static char *trim(char *s) {
    char *e;
    while (*s && isspace((unsigned char)*s)) s++;
    e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}

static const char *skip_space(const char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

static int value_tail_ok(const char *s) {
    s = skip_space(s);
    return *s == 0 || *s == '#';
}

static int match_key(const char *line, const char *key, const char **value) {
    size_t n = strlen(key);
    if (strncmp(line, key, n) != 0 || line[n] != ':') return 0;
    *value = skip_space(line + n + 1);
    return 1;
}

static int parse_int_number(const char *value, int *out, const char **end_out) {
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (end == value || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX) return 1;
    *out = (int)parsed;
    *end_out = end;
    return 0;
}

static int parse_float_number(const char *value, float *out, const char **end_out) {
    char *end = NULL;
    float parsed;
    errno = 0;
    parsed = strtof(value, &end);
    if (end == value || errno == ERANGE || !isfinite(parsed)) return 1;
    *out = parsed;
    *end_out = end;
    return 0;
}

/* Returns zero for no key match, one for success, and -1 for a bad value. */
static int parse_scalar_int(const char *line, const char *key, int *out) {
    const char *value;
    const char *end;
    int parsed;
    if (!match_key(line, key, &value)) return 0;
    if (parse_int_number(value, &parsed, &end) || !value_tail_ok(end)) return -1;
    *out = parsed;
    return 1;
}

/* YAML null is the disabled value for the two optional boolean features. */
static int parse_scalar_bool_null(const char *line, const char *key, int *out) {
    const char *value;
    if (!match_key(line, key, &value)) return 0;
    if (strncmp(value, "true", 4) == 0 && value_tail_ok(value + 4)) {
        *out = 1;
        return 1;
    }
    if (strncmp(value, "false", 5) == 0 && value_tail_ok(value + 5)) {
        *out = 0;
        return 1;
    }
    if (strncmp(value, "null", 4) == 0 && value_tail_ok(value + 4)) {
        *out = 0;
        return 1;
    }
    return -1;
}

static int parse_inline_int_list(const char *value, int *values, int capacity, int *count_out) {
    const char *p = skip_space(value);
    int count = 0;
    if (*p++ != '[') return 1;
    p = skip_space(p);
    if (*p == ']') {
        p++;
    } else {
        for (;;) {
            const char *end;
            int parsed;
            if (count >= capacity || parse_int_number(p, &parsed, &end)) return 1;
            values[count++] = parsed;
            p = skip_space(end);
            if (*p == ']') {
                p++;
                break;
            }
            if (*p++ != ',') return 1;
            p = skip_space(p);
            if (*p == ']') {
                p++;
                break;
            }
        }
    }
    if (!value_tail_ok(p)) return 1;
    *count_out = count;
    return 0;
}

static int parse_inline_float_list(const char *value, float *values, int capacity, int *count_out) {
    const char *p = skip_space(value);
    int count = 0;
    if (*p++ != '[') return 1;
    p = skip_space(p);
    if (*p == ']') {
        p++;
    } else {
        for (;;) {
            const char *end;
            float parsed;
            if (count >= capacity || parse_float_number(p, &parsed, &end)) return 1;
            values[count++] = parsed;
            p = skip_space(end);
            if (*p == ']') {
                p++;
                break;
            }
            if (*p++ != ',') return 1;
            p = skip_space(p);
            if (*p == ']') {
                p++;
                break;
            }
        }
    }
    if (!value_tail_ok(p)) return 1;
    *count_out = count;
    return 0;
}

static int config_error(const char *fmt, ...) {
    va_list args;
    fprintf(stderr, "invalid config: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    return 1;
}

static void config_load_error(const char *path, unsigned long line, const char *fmt, ...) {
    va_list args;
    fprintf(stderr, "invalid config %s", path ? path : "(null)");
    if (line) fprintf(stderr, ":%lu", line);
    fprintf(stderr, ": ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
}

static int checked_int_product(const char *name, int a, int b, int *out) {
    if (a <= 0 || b <= 0 || a > INT_MAX / b) {
        return config_error("%s overflows int", name);
    }
    *out = a * b;
    return 0;
}

static int checked_size_product(const char *name, const size_t *factors, size_t count) {
    size_t product = 1;
    size_t i;
    for (i = 0; i < count; ++i) {
        if (factors[i] != 0 && product > SIZE_MAX / factors[i]) {
            return config_error("%s overflows size_t", name);
        }
        product *= factors[i];
    }
    return 0;
}

void world_config_defaults(WorldConfig *cfg) {
    if (!cfg) return;
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

int world_config_validate(const WorldConfig *cfg) {
    int H;
    int W;
    int output_h;
    int output_w;
    int T;
    int d_head;
    int kv_dim;
    int hidden;
    int out_dim;
    int local_capacity;
    int global_capacity;
    int ignored;
    int i;

    if (!cfg) return config_error("config pointer is null");
    if (cfg->channels <= 0) return config_error("channels must be positive");
    if (cfg->d_model <= 0) return config_error("d_model must be positive");
    if (cfg->n_heads <= 0) return config_error("n_heads must be positive");
    if (cfg->n_kv_heads <= 0) return config_error("n_kv_heads must be positive");
    if (cfg->n_layers <= 0) return config_error("n_layers must be positive");
    if (cfg->height <= 0 || cfg->width <= 0) return config_error("height and width must be positive");
    if (cfg->patch_h <= 0 || cfg->patch_w <= 0) return config_error("patch dimensions must be positive");
    if (cfg->n_buttons < 0 || cfg->n_buttons > INT_MAX - 3) return config_error("n_buttons must be in 0..INT_MAX-3");
    if (cfg->mlp_ratio <= 0) return config_error("mlp_ratio must be positive");
    if (cfg->local_window <= 0 || cfg->global_window <= 0) return config_error("cache windows must be positive");
    if (cfg->global_pinned_dilation <= 0) return config_error("global_pinned_dilation must be positive");
    if (cfg->global_attn_period <= 0) return config_error("global_attn_period must be positive");
    if (cfg->base_fps <= 0 || cfg->inference_fps <= 0 || cfg->temporal_compression <= 0) {
        return config_error("frame-rate values must be positive");
    }
    if ((cfg->value_residual != 0 && cfg->value_residual != 1) ||
            (cfg->prompt_conditioning != 0 && cfg->prompt_conditioning != 1)) {
        return config_error("boolean fields must be zero or one");
    }
    if (cfg->d_model % cfg->n_heads != 0) return config_error("d_model must be divisible by n_heads");
    d_head = cfg->d_model / cfg->n_heads;
    if (d_head % 8 != 0) return config_error("attention head dimension must be divisible by 8");
    if (cfg->n_heads % cfg->n_kv_heads != 0) return config_error("n_heads must be divisible by n_kv_heads");
    if (cfg->global_window % cfg->global_pinned_dilation != 0) {
        return config_error("global_window must be divisible by global_pinned_dilation");
    }

    if (cfg->scheduler_sigmas_count < 2 ||
            cfg->scheduler_sigmas_count > (int)ARRAY_COUNT(cfg->scheduler_sigmas)) {
        return config_error("scheduler_sigmas must contain 2..%zu values", ARRAY_COUNT(cfg->scheduler_sigmas));
    }
    for (i = 0; i < cfg->scheduler_sigmas_count; ++i) {
        float sigma = cfg->scheduler_sigmas[i];
        if (!isfinite(sigma) || sigma < 0.0f) {
            return config_error("scheduler_sigmas[%d] must be finite and non-negative", i);
        }
        if (i > 0 && sigma > cfg->scheduler_sigmas[i - 1]) {
            return config_error("scheduler_sigmas must be non-increasing");
        }
    }

    if (checked_int_product("height * patch_h", cfg->height, cfg->patch_h, &H) ||
            checked_int_product("width * patch_w", cfg->width, cfg->patch_w, &W) ||
            checked_int_product("output height", H, 16, &output_h) ||
            checked_int_product("output width", W, 16, &output_w) ||
            checked_int_product("tokens per frame", cfg->height, cfg->width, &T) ||
            checked_int_product("MLP hidden dimension", cfg->d_model, cfg->mlp_ratio, &hidden) ||
            checked_int_product("KV dimension", cfg->n_kv_heads, d_head, &kv_dim) ||
            checked_int_product("patch output dimension", cfg->channels, cfg->patch_h, &out_dim) ||
            checked_int_product("patch output dimension", out_dim, cfg->patch_w, &out_dim) ||
            checked_int_product("attention launch size", cfg->n_heads, T, &ignored) ||
            checked_int_product("KV launch size", cfg->n_kv_heads, T, &ignored) ||
            checked_int_product("double model dimension", cfg->d_model, 2, &ignored) ||
            checked_int_product("double KV dimension", kv_dim, 2, &ignored)) {
        return 1;
    }
    if (kv_dim > (INT_MAX - cfg->d_model) / 2) return config_error("QKV dimension overflows int");
    if (cfg->local_window > INT_MAX / T - 1) return config_error("local cache capacity overflows int");
    if (cfg->global_window > INT_MAX / T - 1) return config_error("global cache capacity overflows int");
    local_capacity = (cfg->local_window + 1) * T;
    global_capacity = (cfg->global_window + 1) * T;

    {
        const size_t factors[] = {(size_t)cfg->channels, (size_t)H, (size_t)W, sizeof(float)};
        if (checked_size_product("latent allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->d_model, (size_t)cfg->channels,
            (size_t)cfg->patch_h, (size_t)cfg->patch_w, sizeof(float)
        };
        if (checked_size_product("patch weight allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {(size_t)T, (size_t)hidden, sizeof(float)};
        if (checked_size_product("MLP activation allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {(size_t)cfg->d_model, (size_t)hidden, sizeof(float)};
        if (checked_size_product("MLP weight allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->n_heads, (size_t)T, (size_t)global_capacity, sizeof(float)
        };
        if (checked_size_product("attention score allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)output_h, (size_t)output_w, (size_t)4, (size_t)3
        };
        if (checked_size_product("RGB output allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->n_layers, (size_t)6, (size_t)cfg->d_model,
            (size_t)cfg->d_model, sizeof(float)
        };
        if (checked_size_product("layer projection weight address range", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->n_layers, (size_t)cfg->d_model, (size_t)hidden, sizeof(float)
        };
        if (checked_size_product("layer MLP weight address range", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->scheduler_sigmas_count, (size_t)cfg->n_layers,
            (size_t)6, (size_t)cfg->d_model, sizeof(float)
        };
        if (checked_size_product("layer modulation table", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->n_layers, (size_t)cfg->n_kv_heads,
            (size_t)local_capacity, (size_t)d_head, sizeof(float)
        };
        if (checked_size_product("local KV cache allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    {
        const size_t factors[] = {
            (size_t)cfg->n_layers, (size_t)cfg->n_kv_heads,
            (size_t)global_capacity, (size_t)d_head, sizeof(float)
        };
        if (checked_size_product("global KV cache allocation", factors, ARRAY_COUNT(factors))) return 1;
    }
    (void)out_dim;
    return 0;
}

int world_config_load(WorldConfig *cfg, const char *path) {
    FILE *f = NULL;
    char line_buf[512];
    unsigned long line_number = 0;
    unsigned long patch_line = 0;
    unsigned long sigmas_line = 0;
    int in_patch = 0;
    int patch_index = 0;
    int patch_seen = 0;
    int in_sigmas = 0;
    int sigmas_seen = 0;
    int tokens_per_frame = 0;
    int tokens_per_frame_seen = 0;
    uint64_t scalar_seen = 0;

#define LOAD_FAIL(...) do { \
        config_load_error(path, line_number, __VA_ARGS__); \
        goto fail; \
    } while (0)

    if (!cfg || !path || !path[0]) {
        config_load_error(path, 0, "config pointer and path are required");
        return 1;
    }
    world_config_defaults(cfg);

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open config: %s\n", path);
        return 1;
    }

    /* Preserve the historical loaded-config fallback when no schedule is specified. */
    cfg->scheduler_sigmas_count = 0;

    while (fgets(line_buf, sizeof(line_buf), f)) {
        char *line;
        size_t raw_length;
        size_t indentation = 0;
        int matched = 0;
        size_t i;

        line_number++;
        raw_length = strlen(line_buf);
        if (raw_length > 0 && line_buf[raw_length - 1] != '\n' && !feof(f)) {
            int ch;
            while ((ch = fgetc(f)) != '\n' && ch != EOF) {}
            LOAD_FAIL("line is longer than %zu bytes", sizeof(line_buf) - 1);
        }
        while (line_buf[indentation] == ' ' || line_buf[indentation] == '\t') indentation++;
        line = trim(line_buf);
        if (!line[0] || line[0] == '#') continue;

        if ((in_patch || in_sigmas) && line[0] != '-') {
            if (in_patch && patch_index != 2) {
                line_number = patch_line;
                LOAD_FAIL("patch must contain exactly two integers");
            }
            if (in_sigmas && cfg->scheduler_sigmas_count < 2) {
                line_number = sigmas_line;
                LOAD_FAIL("scheduler_sigmas must contain at least two values");
            }
            in_patch = 0;
            in_sigmas = 0;
        }

        if (in_patch && line[0] == '-') {
            const char *end;
            int value;
            if (patch_index >= 2 || parse_int_number(skip_space(line + 1), &value, &end) || !value_tail_ok(end)) {
                LOAD_FAIL("patch entries must be exactly two valid integers");
            }
            if (patch_index == 0) cfg->patch_h = value;
            else cfg->patch_w = value;
            patch_index++;
            continue;
        }
        if (in_sigmas && line[0] == '-') {
            const char *end;
            float value;
            if (cfg->scheduler_sigmas_count >= (int)ARRAY_COUNT(cfg->scheduler_sigmas) ||
                    parse_float_number(skip_space(line + 1), &value, &end) || !value_tail_ok(end)) {
                LOAD_FAIL("scheduler_sigmas entry is invalid or exceeds %zu values", ARRAY_COUNT(cfg->scheduler_sigmas));
            }
            cfg->scheduler_sigmas[cfg->scheduler_sigmas_count++] = value;
            continue;
        }

        /* Only top-level mappings are model configuration fields. */
        if (indentation != 0) continue;

        {
            const char *value;
            if (match_key(line, "patch", &value)) {
                int pair[2];
                int count = 0;
                if (patch_seen) LOAD_FAIL("duplicate patch field");
                patch_seen = 1;
                patch_line = line_number;
                if (value_tail_ok(value)) {
                    in_patch = 1;
                    patch_index = 0;
                } else {
                    if (parse_inline_int_list(value, pair, 2, &count) || count != 2) {
                        LOAD_FAIL("patch must be an inline pair or a two-item block list");
                    }
                    cfg->patch_h = pair[0];
                    cfg->patch_w = pair[1];
                    patch_index = 2;
                }
                continue;
            }
            if (match_key(line, "scheduler_sigmas", &value)) {
                int count = 0;
                if (sigmas_seen) LOAD_FAIL("duplicate scheduler_sigmas field");
                sigmas_seen = 1;
                sigmas_line = line_number;
                cfg->scheduler_sigmas_count = 0;
                if (value_tail_ok(value)) {
                    in_sigmas = 1;
                } else {
                    if (parse_inline_float_list(
                                value,
                                cfg->scheduler_sigmas,
                                (int)ARRAY_COUNT(cfg->scheduler_sigmas),
                                &count) || count < 2) {
                        LOAD_FAIL("scheduler_sigmas must be an inline list of 2..%zu finite numbers",
                                  ARRAY_COUNT(cfg->scheduler_sigmas));
                    }
                    cfg->scheduler_sigmas_count = count;
                }
                continue;
            }
        }

        for (i = 0; i < ARRAY_COUNT(k_int_fields); ++i) {
            int *field = (int *)((unsigned char *)cfg + k_int_fields[i].offset);
            int result = parse_scalar_int(line, k_int_fields[i].key, field);
            if (result < 0) LOAD_FAIL("%s must be a complete base-10 integer", k_int_fields[i].key);
            if (result > 0) {
                uint64_t bit = UINT64_C(1) << i;
                if (scalar_seen & bit) LOAD_FAIL("duplicate %s field", k_int_fields[i].key);
                scalar_seen |= bit;
                matched = 1;
                break;
            }
        }
        if (matched) continue;

        for (i = 0; i < ARRAY_COUNT(k_bool_fields); ++i) {
            int *field = (int *)((unsigned char *)cfg + k_bool_fields[i].offset);
            int result = parse_scalar_bool_null(line, k_bool_fields[i].key, field);
            if (result < 0) LOAD_FAIL("%s must be exactly true, false, or null", k_bool_fields[i].key);
            if (result > 0) {
                size_t bit_index = ARRAY_COUNT(k_int_fields) + i;
                uint64_t bit = UINT64_C(1) << bit_index;
                if (scalar_seen & bit) LOAD_FAIL("duplicate %s field", k_bool_fields[i].key);
                scalar_seen |= bit;
                matched = 1;
                break;
            }
        }
        if (matched) continue;

        {
            int result = parse_scalar_int(line, "tokens_per_frame", &tokens_per_frame);
            if (result < 0) LOAD_FAIL("tokens_per_frame must be a complete base-10 integer");
            if (result > 0) {
                if (tokens_per_frame_seen) LOAD_FAIL("duplicate tokens_per_frame field");
                tokens_per_frame_seen = 1;
            }
        }
        /* Unknown top-level YAML keys are intentionally ignored for compatibility. */
    }

    if (ferror(f)) LOAD_FAIL("failed while reading file");
    if (in_patch && patch_index != 2) {
        line_number = patch_line;
        LOAD_FAIL("patch must contain exactly two integers");
    }
    if (in_sigmas && cfg->scheduler_sigmas_count < 2) {
        line_number = sigmas_line;
        LOAD_FAIL("scheduler_sigmas must contain at least two values");
    }
    if (fclose(f) != 0) {
        f = NULL;
        config_load_error(path, 0, "failed to close file after reading");
        return 1;
    }
    f = NULL;

    if (!sigmas_seen) {
        cfg->scheduler_sigmas[0] = 1.0f;
        cfg->scheduler_sigmas[1] = 0.0f;
        cfg->scheduler_sigmas_count = 2;
    }
    if (world_config_validate(cfg)) return 1;
    if (tokens_per_frame_seen && tokens_per_frame != cfg->height * cfg->width) {
        config_load_error(path, 0, "tokens_per_frame=%d does not equal height*width=%d",
                          tokens_per_frame, cfg->height * cfg->width);
        return 1;
    }
    return 0;

fail:
    if (f) fclose(f);
    return 1;
#undef LOAD_FAIL
}

void world_config_print(const WorldConfig *cfg) {
    int i;
    if (!cfg) return;
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
    for (i = 0; i < cfg->scheduler_sigmas_count; ++i) {
        fprintf(stderr, " %.6g", cfg->scheduler_sigmas[i]);
    }
    fprintf(stderr, "\n");
}
