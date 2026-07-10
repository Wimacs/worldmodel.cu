#define WORLD_CLI_NO_MAIN
#include "world_cli.c"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#ifndef NOUSER
#define NOUSER
#endif
#endif

#include "raylib.h"
#include "world_backend.h"

#include <math.h>
#include <errno.h>
#ifndef _WIN32
#include <time.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#if WORLD_BACKEND_BYTES_PER_PIXEL == 4
#define WORLD_RAYLIB_PIXEL_FORMAT PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
#else
#define WORLD_RAYLIB_PIXEL_FORMAT PIXELFORMAT_UNCOMPRESSED_R8G8B8
#endif

#ifdef _WIN32
typedef CRITICAL_SECTION WorldMutex;
typedef HANDLE WorldThread;
typedef DWORD (WINAPI *WorldThreadFn)(void *);
#define WORLD_THREAD_RETURN DWORD WINAPI
#define WORLD_THREAD_OK 0

static int world_mutex_init(WorldMutex *m) {
    InitializeCriticalSection(m);
    return 0;
}

static void world_mutex_destroy(WorldMutex *m) {
    DeleteCriticalSection(m);
}

static void world_mutex_lock(WorldMutex *m) {
    EnterCriticalSection(m);
}

static void world_mutex_unlock(WorldMutex *m) {
    LeaveCriticalSection(m);
}

static int world_thread_create(WorldThread *thread, WorldThreadFn fn, void *arg) {
    *thread = CreateThread(NULL, 0, fn, arg, 0, NULL);
    return *thread == NULL;
}

static void world_thread_join(WorldThread thread) {
    if (!thread) return;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}
#else
typedef pthread_mutex_t WorldMutex;
typedef pthread_t WorldThread;
typedef void *(*WorldThreadFn)(void *);
#define WORLD_THREAD_RETURN void *
#define WORLD_THREAD_OK NULL

static int world_mutex_init(WorldMutex *m) {
    return pthread_mutex_init(m, NULL);
}

static void world_mutex_destroy(WorldMutex *m) {
    pthread_mutex_destroy(m);
}

static void world_mutex_lock(WorldMutex *m) {
    pthread_mutex_lock(m);
}

static void world_mutex_unlock(WorldMutex *m) {
    pthread_mutex_unlock(m);
}

static int world_thread_create(WorldThread *thread, WorldThreadFn fn, void *arg) {
    return pthread_create(thread, NULL, fn, arg);
}

static void world_thread_join(WorldThread thread) {
    pthread_join(thread, NULL);
}
#endif

static void world_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

typedef struct {
    WorldModelProbeWeights probe;
    WorldLayerWeights *layers;
    float *patchify_weight;
    float *denoise_fc1_weight;
    float *denoise_fc2_weight;
    float *ctrl_emb_fc1_weight;
    float *ctrl_emb_fc2_weight;
    float *out_norm_fc_weight;
    float *unpatchify_weight;
    float *unpatchify_bias;
} LoadedWorldModel;

typedef struct {
    WorldMutex mutex;
    WorldRuntime *rt;
    float *control;
    int ctrl_dim;
    int n_buttons;
    unsigned char *rgb;
    size_t rgb_bytes;
    int width;
    int height;
    int frames;
    int ready;
    int stop;
    int failed;
    int produced_chunks;
    float generation_seconds;
    float control_seconds;
    float target_control_seconds;
} LiveShared;

enum {
    WORLD_VK_LMB = 0x01,
    WORLD_VK_RMB = 0x02,
    WORLD_VK_SHIFT = 0x10,
    WORLD_VK_SPACE = 0x20,
    WORLD_VK_A = 0x41,
    WORLD_VK_D = 0x44,
    WORLD_VK_S = 0x53,
    WORLD_VK_W = 0x57,
    WORLD_VK_LSHIFT = 0xA0,
};

static void ray_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--model-dir DIR] [--weights FILE] [--vae-weights FILE] [--seed-latent FILE] [--steps N] [--layers N] [--cache-window N] [--fast-realtime] [--frame-idx N] [--seed N] [--noise normal|uniform] [--mouse-scale X] [--window-width N] [--window-height N] [--warmup N] [--headless-smoke] [--headless-out PATH] [--headless-mouse X Y] [--headless-button N]\n"
            "\n"
            "Raylib realtime frontend. Loads weights, optionally prewarms a temporary runtime,\n"
            "then renders decoded frames from a fresh resident runtime without writing images.\n",
            argv0);
}

static void free_loaded_model(LoadedWorldModel *m) {
    if (!m) return;
    free(m->patchify_weight);
    free(m->denoise_fc1_weight);
    free(m->denoise_fc2_weight);
    free(m->ctrl_emb_fc1_weight);
    free(m->ctrl_emb_fc2_weight);
    free(m->out_norm_fc_weight);
    free(m->unpatchify_weight);
    free(m->unpatchify_bias);
    free_layer_weights(m->layers, m->probe.n_layers);
    memset(m, 0, sizeof(*m));
}

static int ray_load_layer_as_f32(
        const SafeTensors *st,
        int layer,
        const char *suffix,
        const int64_t *shape,
        int ndim,
        float **out) {
    char name[256];
    int n = snprintf(name, sizeof(name), "transformer.blocks.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return 1;
    return load_required_as_f32(st, name, shape, ndim, out);
}

static int ray_load_optional_layer_as_f32(
        const SafeTensors *st,
        int layer,
        const char *suffix,
        const int64_t *shape,
        int ndim,
        float **out) {
    char name[256];
    int n = snprintf(name, sizeof(name), "transformer.blocks.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return 1;
    const SafeTensorEntry *e = safetensors_find(st, name);
    if (!e) {
        *out = NULL;
        return 0;
    }
    return load_required_as_f32(st, name, shape, ndim, out);
}

static int load_live_model_weights(
        const SafeTensors *st,
        const WorldConfig *cfg,
        int layers_to_run,
        LoadedWorldModel *m) {
    memset(m, 0, sizeof(*m));
    int hidden = cfg->d_model * cfg->mlp_ratio;
    int d_head = cfg->d_model / cfg->n_heads;
    int kv_dim = cfg->n_kv_heads * d_head;
    int ctrl_dim = cfg->n_buttons + 3;
    int64_t patch_shape[4] = {cfg->d_model, cfg->channels, cfg->patch_h, cfg->patch_w};
    int64_t denoise_fc1_shape[2] = {hidden, 512};
    int64_t denoise_fc2_shape[2] = {cfg->d_model, hidden};
    int64_t ctrl_emb_fc1_shape[2] = {hidden, ctrl_dim};
    int64_t ctrl_emb_fc2_shape[2] = {cfg->d_model, hidden};
    int64_t hidden_d_shape[2] = {hidden, cfg->d_model};
    int64_t d_shape[1] = {cfg->d_model};
    int64_t dxd_shape[2] = {cfg->d_model, cfg->d_model};
    int64_t out_norm_shape[2] = {cfg->d_model * 2, cfg->d_model};
    int64_t kv_proj_shape[2] = {kv_dim, cfg->d_model};
    int64_t unpatch_bias_shape[1] = {cfg->channels};

    if (load_required_as_f32(st, "patchify.weight", patch_shape, 4, &m->patchify_weight)) return 1;
    if (load_required_as_f32(st, "denoise_step_emb.mlp.fc1.weight", denoise_fc1_shape, 2, &m->denoise_fc1_weight)) return 1;
    if (load_required_as_f32(st, "denoise_step_emb.mlp.fc2.weight", denoise_fc2_shape, 2, &m->denoise_fc2_weight)) return 1;
    if (load_required_as_f32(st, "ctrl_emb.mlp.fc1.weight", ctrl_emb_fc1_shape, 2, &m->ctrl_emb_fc1_weight)) return 1;
    if (load_required_as_f32(st, "ctrl_emb.mlp.fc2.weight", ctrl_emb_fc2_shape, 2, &m->ctrl_emb_fc2_weight)) return 1;
    if (load_required_as_f32(st, "out_norm.fc.weight", out_norm_shape, 2, &m->out_norm_fc_weight)) return 1;
    if (load_required_as_f32(st, "unpatchify.weight", patch_shape, 4, &m->unpatchify_weight)) return 1;
    if (load_required_as_f32(st, "unpatchify.bias", unpatch_bias_shape, 1, &m->unpatchify_bias)) return 1;

    m->layers = (WorldLayerWeights *)calloc((size_t)layers_to_run, sizeof(*m->layers));
    if (!m->layers) return 1;
    for (int layer = 0; layer < layers_to_run; ++layer) {
        WorldLayerWeights *lw = &m->layers[layer];
        if (ray_load_layer_as_f32(st, layer, "mlp_cond_head.bias_in", d_shape, 1, (float **)&lw->cond_bias)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn_cond_head.cond_proj.0.weight", dxd_shape, 2, (float **)&lw->attn_cond_s_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn_cond_head.cond_proj.1.weight", dxd_shape, 2, (float **)&lw->attn_cond_b_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn_cond_head.cond_proj.2.weight", dxd_shape, 2, (float **)&lw->attn_cond_g_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn.q_proj.weight", dxd_shape, 2, (float **)&lw->q_proj_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn.k_proj.weight", kv_proj_shape, 2, (float **)&lw->k_proj_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn.v_proj.weight", kv_proj_shape, 2, (float **)&lw->v_proj_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn.out_proj.weight", dxd_shape, 2, (float **)&lw->out_proj_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "attn.v_lamb", NULL, 0, (float **)&lw->v_lamb)) return 1;
        if (ray_load_layer_as_f32(st, layer, "mlp_cond_head.cond_proj.0.weight", dxd_shape, 2, (float **)&lw->mlp_cond_s_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "mlp_cond_head.cond_proj.1.weight", dxd_shape, 2, (float **)&lw->mlp_cond_b_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "mlp_cond_head.cond_proj.2.weight", dxd_shape, 2, (float **)&lw->mlp_cond_g_weight)) return 1;
        if (ray_load_optional_layer_as_f32(st, layer, "ctrl_mlpfusion.fc1_x.weight", dxd_shape, 2, (float **)&lw->ctrl_fc1_x_weight)) return 1;
        lw->has_ctrl = lw->ctrl_fc1_x_weight != NULL;
        if (lw->has_ctrl && ray_load_layer_as_f32(st, layer, "ctrl_mlpfusion.fc1_c.weight", dxd_shape, 2, (float **)&lw->ctrl_fc1_c_weight)) return 1;
        if (lw->has_ctrl && ray_load_layer_as_f32(st, layer, "ctrl_mlpfusion.fc2.weight", dxd_shape, 2, (float **)&lw->ctrl_fc2_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "dit_mlp.fc1.weight", hidden_d_shape, 2, (float **)&lw->dit_mlp_fc1_weight)) return 1;
        if (ray_load_layer_as_f32(st, layer, "dit_mlp.fc2.weight", denoise_fc2_shape, 2, (float **)&lw->dit_mlp_fc2_weight)) return 1;
    }

    m->probe.patchify_weight = m->patchify_weight;
    m->probe.denoise_fc1_weight = m->denoise_fc1_weight;
    m->probe.denoise_fc2_weight = m->denoise_fc2_weight;
    m->probe.ctrl_emb_fc1_weight = m->ctrl_emb_fc1_weight;
    m->probe.ctrl_emb_fc2_weight = m->ctrl_emb_fc2_weight;
    m->probe.layers = m->layers;
    m->probe.n_layers = layers_to_run;
    m->probe.out_norm_fc_weight = m->out_norm_fc_weight;
    m->probe.unpatchify_weight = m->unpatchify_weight;
    m->probe.unpatchify_bias = m->unpatchify_bias;
    return 0;
}

static void set_button_vk(float *control, int n_buttons, int vk, int down) {
    if (vk >= 0 && vk < n_buttons) control[2 + vk] = down ? 1.0f : 0.0f;
}

static float clamp_scroll_sign(float x) {
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

static float clamp_mouse_axis(float x) {
    if (x > 1.0f) return 1.0f;
    if (x < -1.0f) return -1.0f;
    return x;
}

static float clamp_mouse_accum(float x) {
    if (x > 8.0f) return 8.0f;
    if (x < -8.0f) return -8.0f;
    return x;
}

static void fill_raylib_control(float *control, int ctrl_dim, int n_buttons, float mouse_scale) {
    memset(control, 0, (size_t)ctrl_dim * sizeof(float));
    Vector2 delta = GetMouseDelta();
    if (ctrl_dim >= n_buttons + 3) {
        control[0] = clamp_mouse_axis(delta.x * mouse_scale * 0.01f);
        control[1] = clamp_mouse_axis(delta.y * mouse_scale * 0.01f);
        set_button_vk(control, n_buttons, WORLD_VK_W, IsKeyDown(KEY_W));
        set_button_vk(control, n_buttons, WORLD_VK_S, IsKeyDown(KEY_S));
        set_button_vk(control, n_buttons, WORLD_VK_A, IsKeyDown(KEY_A));
        set_button_vk(control, n_buttons, WORLD_VK_D, IsKeyDown(KEY_D));
        set_button_vk(control, n_buttons, WORLD_VK_SPACE, IsKeyDown(KEY_SPACE));
        set_button_vk(control, n_buttons, WORLD_VK_SHIFT, IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT));
        set_button_vk(control, n_buttons, WORLD_VK_LSHIFT, IsKeyDown(KEY_LEFT_SHIFT));
        set_button_vk(control, n_buttons, WORLD_VK_LMB, IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        set_button_vk(control, n_buttons, WORLD_VK_RMB, IsMouseButtonDown(MOUSE_BUTTON_RIGHT));
        control[2 + n_buttons] = clamp_scroll_sign(GetMouseWheelMove());
    }
}

static void merge_frame_control(LiveShared *s, const float *frame_control, float frame_seconds) {
    if (s->ctrl_dim < s->n_buttons + 3) return;
    s->control[0] = clamp_mouse_accum(s->control[0] + frame_control[0]);
    s->control[1] = clamp_mouse_accum(s->control[1] + frame_control[1]);
    if (frame_seconds > 0.0f && frame_seconds < 0.25f) {
        s->control_seconds += frame_seconds;
    }
    for (int i = 0; i < s->n_buttons; ++i) {
        s->control[2 + i] = frame_control[2 + i];
    }
    s->control[2 + s->n_buttons] =
        clamp_scroll_sign(s->control[2 + s->n_buttons] + frame_control[2 + s->n_buttons]);
}

static int control_has_activity(const float *control, int ctrl_dim, int n_buttons) {
    if (!control || ctrl_dim < n_buttons + 3) return 1;
    if (fabsf(control[0]) > 1.5e-1f || fabsf(control[1]) > 1.5e-1f) return 1;
    for (int i = 0; i < n_buttons; ++i) {
        if (fabsf(control[2 + i]) > 0.5f) return 1;
    }
    if (fabsf(control[2 + n_buttons]) > 0.5f) return 1;
    return 0;
}

static WORLD_THREAD_RETURN generation_worker(void *arg) {
    LiveShared *s = (LiveShared *)arg;
    float *control = (float *)malloc((size_t)s->ctrl_dim * sizeof(float));
    if (!control) {
        world_mutex_lock(&s->mutex);
        s->failed = 1;
        world_mutex_unlock(&s->mutex);
        return WORLD_THREAD_OK;
    }

    for (;;) {
        world_mutex_lock(&s->mutex);
        int stop = s->stop;
        memcpy(control, s->control, (size_t)s->ctrl_dim * sizeof(float));
        if (s->ctrl_dim >= s->n_buttons + 3) {
            if (s->target_control_seconds > 0.0f && s->control_seconds > s->target_control_seconds) {
                float k = s->target_control_seconds / s->control_seconds;
                control[0] = clamp_mouse_axis(control[0] * k);
                control[1] = clamp_mouse_axis(control[1] * k);
            }
            s->control[0] = 0.0f;
            s->control[1] = 0.0f;
            s->control[2 + s->n_buttons] = 0.0f;
            s->control_seconds = 0.0f;
        }
        world_mutex_unlock(&s->mutex);
        if (stop) break;
        if (!control_has_activity(control, s->ctrl_dim, s->n_buttons)) {
            world_sleep_ms(5);
            continue;
        }

        const unsigned char *pixels = NULL;
        int width = 0;
        int height = 0;
        int frames = 0;
        float seconds = 0.0f;
        if (world_runtime_step_pixels(s->rt, control, &pixels, &width, &height, &frames, &seconds)) {
            world_mutex_lock(&s->mutex);
            s->failed = 1;
            s->stop = 1;
            world_mutex_unlock(&s->mutex);
            break;
        }

        size_t bytes = (size_t)width * height * WORLD_BACKEND_BYTES_PER_PIXEL * frames;
        world_mutex_lock(&s->mutex);
        if (bytes == s->rgb_bytes) {
            memcpy(s->rgb, pixels, bytes);
            s->width = width;
            s->height = height;
            s->frames = frames;
            s->generation_seconds = seconds;
            s->produced_chunks += 1;
            s->ready = 1;
        } else {
            s->failed = 1;
            s->stop = 1;
        }
        world_mutex_unlock(&s->mutex);
    }

    free(control);
    return WORLD_THREAD_OK;
}

static Rectangle fit_rect(int dst_w, int dst_h, int src_w, int src_h) {
    float sx = (float)dst_w / (float)src_w;
    float sy = (float)dst_h / (float)src_h;
    float scale = sx < sy ? sx : sy;
    float w = (float)src_w * scale;
    float h = (float)src_h * scale;
    Rectangle r = {(float)(dst_w - w) * 0.5f, (float)(dst_h - h) * 0.5f, w, h};
    return r;
}

static int ray_write_ppm(
        const char *path,
        const unsigned char *pixels,
        int width,
        int height,
        int bytes_per_pixel) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "failed to open debug image %s: %s\n", path, strerror(errno));
        return 1;
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    int ok = 1;
    if (bytes_per_pixel == 3) {
        size_t n = (size_t)width * height * 3;
        ok = fwrite(pixels, 1, n, f) == n;
    } else if (bytes_per_pixel == 4) {
        unsigned char *row = (unsigned char *)malloc((size_t)width * 3);
        if (!row) {
            fclose(f);
            fprintf(stderr, "failed to allocate debug image row\n");
            return 1;
        }
        for (int y = 0; y < height && ok; ++y) {
            const unsigned char *src = pixels + (size_t)y * width * 4;
            for (int x = 0; x < width; ++x) {
                row[x * 3 + 0] = src[x * 4 + 0];
                row[x * 3 + 1] = src[x * 4 + 1];
                row[x * 3 + 2] = src[x * 4 + 2];
            }
            size_t n = (size_t)width * 3;
            ok = fwrite(row, 1, n, f) == n;
        }
        free(row);
    } else {
        ok = 0;
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "failed to write debug image %s\n", path);
        return 1;
    }
    return 0;
}

static int ray_make_frame_path(char *out, size_t out_size, const char *path, int frame_idx) {
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    const char *dot = strrchr(name, '.');
    int stem_len = dot ? (int)(dot - path) : (int)strlen(path);
    const char *ext = dot ? dot : "";
    int n = snprintf(out, out_size, "%.*s.%d%s", stem_len, path, frame_idx, ext);
    return n < 0 || (size_t)n >= out_size;
}

static int ray_write_ppm_frames(
        const char *path,
        const unsigned char *pixels,
        int frame_count,
        int width,
        int height,
        int bytes_per_pixel) {
    size_t frame_bytes = (size_t)width * height * bytes_per_pixel;
    if (ray_write_ppm(path, pixels, width, height, bytes_per_pixel)) return 1;
    for (int i = 0; i < frame_count; ++i) {
        char frame_path[PATH_BUF];
        if (ray_make_frame_path(frame_path, sizeof(frame_path), path, i)) return 1;
        if (ray_write_ppm(frame_path, pixels + (size_t)i * frame_bytes, width, height, bytes_per_pixel)) return 1;
    }
    return 0;
}

static float playback_interval_seconds(float generation_seconds, int frames) {
    if (frames <= 0 || generation_seconds <= 0.0f) return 1.0f / 60.0f;
    return generation_seconds / (float)frames;
}

int main(int argc, char **argv) {
    const char *model_dir = "../Waypoint-1.5-1B";
    const char *weights = NULL;
    const char *vae_weights = NULL;
    const char *seed_latent_path = NULL;
    const char *headless_out_path = NULL;
    int steps_to_run = -1;
    int layers_to_run = -1;
    int frame_idx = 0;
    int window_width = 1280;
    int window_height = 720;
    int warmup_chunks = 1;
    int headless_smoke = 0;
    int fast_realtime = 0;
    int cache_window_override = -1;
    int headless_has_mouse = 0;
    float headless_mouse_x = 0.0f;
    float headless_mouse_y = 0.0f;
    int headless_button = -1;
    float mouse_scale = 1.0f;
    unsigned int seed = 1234;
    int noise_mode = WORLD_NOISE_NORMAL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
        } else if (strcmp(argv[i], "--vae-weights") == 0 && i + 1 < argc) {
            vae_weights = argv[++i];
        } else if (strcmp(argv[i], "--seed-latent") == 0 && i + 1 < argc) {
            seed_latent_path = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            layers_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache-window") == 0 && i + 1 < argc) {
            cache_window_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fast-realtime") == 0) {
            fast_realtime = 1;
        } else if (strcmp(argv[i], "--frame-idx") == 0 && i + 1 < argc) {
            frame_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--noise") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "normal") == 0) noise_mode = WORLD_NOISE_NORMAL;
            else if (strcmp(mode, "uniform") == 0) noise_mode = WORLD_NOISE_UNIFORM;
            else {
                fprintf(stderr, "invalid --noise %s\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--mouse-scale") == 0 && i + 1 < argc) {
            mouse_scale = (float)atof(argv[++i]);
            if (mouse_scale <= 0.0f) {
                fprintf(stderr, "invalid --mouse-scale %.6f\n", mouse_scale);
                return 1;
            }
        } else if (strcmp(argv[i], "--window-width") == 0 && i + 1 < argc) {
            window_width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--window-height") == 0 && i + 1 < argc) {
            window_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_chunks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--headless-smoke") == 0) {
            headless_smoke = 1;
        } else if (strcmp(argv[i], "--headless-out") == 0 && i + 1 < argc) {
            headless_out_path = argv[++i];
        } else if (strcmp(argv[i], "--headless-mouse") == 0 && i + 2 < argc) {
            headless_mouse_x = strtof(argv[++i], NULL);
            headless_mouse_y = strtof(argv[++i], NULL);
            headless_has_mouse = 1;
        } else if (strcmp(argv[i], "--headless-button") == 0 && i + 1 < argc) {
            headless_button = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            ray_usage(argv[0]);
            return 0;
        } else {
            ray_usage(argv[0]);
            return 1;
        }
    }

    char config_path[PATH_BUF];
    char default_weights[PATH_BUF];
    char default_vae_weights[PATH_BUF];
    if (join_path(config_path, sizeof(config_path), model_dir, "config.yaml")) return 1;
    if (!weights) {
        if (join_path(default_weights, sizeof(default_weights), model_dir, "model.safetensors")) return 1;
        weights = default_weights;
    }
    if (!vae_weights) {
        if (join_path(default_vae_weights, sizeof(default_vae_weights), model_dir, "vae/diffusion_pytorch_model.safetensors")) return 1;
        vae_weights = default_vae_weights;
    }

    WorldConfig cfg;
    if (world_config_load(&cfg, config_path)) return 1;
    if (fast_realtime) {
        if (steps_to_run < 0) steps_to_run = 1;
        if (cache_window_override < 0) cache_window_override = 2;
    }
    if (cache_window_override > 0) {
        cfg.local_window = cache_window_override;
        cfg.global_window = cache_window_override;
        if (fast_realtime) {
            cfg.global_pinned_dilation = 1;
            fprintf(stderr, "fast realtime cache override: local_window=%d global_window=%d global_pinned_dilation=1\n",
                    cfg.local_window, cfg.global_window);
        } else {
            int min_global_window = cfg.global_pinned_dilation * 2;
            if (cfg.global_window < min_global_window) {
                fprintf(stderr,
                        "global cache override raised from %d to %d to keep at least two pinned global buckets (global_pinned_dilation=%d)\n",
                        cfg.global_window, min_global_window, cfg.global_pinned_dilation);
                cfg.global_window = min_global_window;
            }
            if (cfg.global_window % cfg.global_pinned_dilation != 0) {
                int adjusted = ((cfg.global_window + cfg.global_pinned_dilation - 1) / cfg.global_pinned_dilation) *
                    cfg.global_pinned_dilation;
                fprintf(stderr,
                        "global cache override rounded from %d to %d to preserve global_pinned_dilation=%d\n",
                        cfg.global_window, adjusted, cfg.global_pinned_dilation);
                cfg.global_window = adjusted;
            }
            fprintf(stderr, "cache override: local_window=%d global_window=%d global_pinned_dilation=%d\n",
                    cfg.local_window, cfg.global_window, cfg.global_pinned_dilation);
        }
        if (cache_window_override == 1) {
            fprintf(stderr,
                    "warning: --cache-window 1 masks the current ring slot and leaves no previous-frame history; use --cache-window 2 or higher for controllable rollout\n");
        }
    }
    if (layers_to_run < 0) layers_to_run = cfg.n_layers;
    if (steps_to_run < 0) steps_to_run = cfg.scheduler_sigmas_count - 1;
    if (layers_to_run <= 0 || layers_to_run > cfg.n_layers || steps_to_run <= 0 || steps_to_run >= cfg.scheduler_sigmas_count) {
        fprintf(stderr, "invalid --layers/--steps for config\n");
        return 1;
    }
    fprintf(stderr, "raylib mouse scale: %.4f\n", mouse_scale);
    world_config_print(&cfg);

    SafeTensors st;
    LoadedWorldModel model;
    WorldVaeDecoderWeights vae;
    WorldRuntime *rt = NULL;
    float *seed_latent = NULL;
    memset(&st, 0, sizeof(st));
    memset(&model, 0, sizeof(model));
    memset(&vae, 0, sizeof(vae));
    int rc = 1;

    size_t seed_latent_elems = (size_t)cfg.channels * (size_t)(cfg.height * cfg.patch_h) * (size_t)(cfg.width * cfg.patch_w);
    if (seed_latent_path) {
        if (read_f32_file_exact(seed_latent_path, seed_latent_elems, &seed_latent)) goto cleanup_before_window;
        fprintf(stderr, "loaded seed latent: %s elems=%zu\n", seed_latent_path, seed_latent_elems);
    }

    fprintf(stderr, "loading transformer safetensors: %s\n", weights);
    if (safetensors_open(&st, weights)) goto cleanup_before_window;
    if (load_live_model_weights(&st, &cfg, layers_to_run, &model)) goto cleanup_before_window;
    safetensors_close(&st);
    memset(&st, 0, sizeof(st));
    if (load_vae_decoder_weights(vae_weights, &vae)) goto cleanup_before_window;

    int ctrl_dim = cfg.n_buttons + 3;
    float *seed_control = (float *)calloc((size_t)ctrl_dim, sizeof(float));
    float *warm_control = (float *)calloc((size_t)ctrl_dim, sizeof(float));
    if (!seed_control || !warm_control) {
        free(seed_control);
        free(warm_control);
        goto cleanup_before_window;
    }
    if (headless_has_mouse) {
        warm_control[0] = clamp_mouse_axis(headless_mouse_x);
        warm_control[1] = clamp_mouse_axis(headless_mouse_y);
    }
    if (headless_button >= 0) {
        if (headless_button < cfg.n_buttons) {
            warm_control[2 + headless_button] = 1.0f;
        } else {
            fprintf(stderr, "warning: ignoring --headless-button %d outside n_buttons=%d\n",
                    headless_button, cfg.n_buttons);
        }
    }
    const unsigned char *warm_pixels = NULL;
    int rgb_w = 0;
    int rgb_h = 0;
    int rgb_frames = 0;
    float warm_seconds = 0.0f;
    if (warmup_chunks > 0) {
        WorldRuntime *prewarm_rt = NULL;
        const unsigned char *prewarm_pixels = NULL;
        int prewarm_w = 0;
        int prewarm_h = 0;
        int prewarm_frames = 0;
        float prewarm_seconds = 0.0f;
        fprintf(stderr,
                "prewarming %s runtime for %d chunk(s) on a temporary state; displayed runtime will be reset\n",
                WORLD_BACKEND_NAME, warmup_chunks);
        if (world_runtime_create(&prewarm_rt, &cfg, &model.probe, layers_to_run, steps_to_run, frame_idx, seed, noise_mode, &vae)) {
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        if (seed_latent) {
            fprintf(stderr, "prewarm seed latent cache pass\n");
            if (world_runtime_seed_latent_pixels(prewarm_rt, seed_latent, seed_control,
                        &prewarm_pixels, &prewarm_w, &prewarm_h, &prewarm_frames, &prewarm_seconds)) {
                world_runtime_destroy(prewarm_rt);
                free(seed_control);
                free(warm_control);
                goto cleanup_before_window;
            }
        }
        for (int i = 0; i < warmup_chunks; ++i) {
            fprintf(stderr, "prewarm chunk %d/%d\n", i + 1, warmup_chunks);
            if (world_runtime_step_pixels(prewarm_rt, warm_control,
                        &prewarm_pixels, &prewarm_w, &prewarm_h, &prewarm_frames, &prewarm_seconds)) {
                world_runtime_destroy(prewarm_rt);
                free(seed_control);
                free(warm_control);
                goto cleanup_before_window;
            }
        }
        world_runtime_destroy(prewarm_rt);
    }

    fprintf(stderr, "creating resident %s runtime\n", WORLD_BACKEND_NAME);
    if (world_runtime_create(&rt, &cfg, &model.probe, layers_to_run, steps_to_run, frame_idx, seed, noise_mode, &vae)) {
        free(seed_control);
        free(warm_control);
        goto cleanup_before_window;
    }
    free_loaded_model(&model);
    free_vae_decoder_weights(&vae);

    if (seed_latent) {
        fprintf(stderr, "seeding runtime KV cache from latent\n");
        if (world_runtime_seed_latent_pixels(rt, seed_latent, seed_control, &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds)) {
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
    } else {
        fprintf(stderr, "generating initial chunk before window\n");
        if (world_runtime_step_pixels(rt, warm_control, &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds)) {
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
    }
    free(seed_control);
    free(warm_control);
    if (!warm_pixels || rgb_w <= 0 || rgb_h <= 0 || rgb_frames <= 0) goto cleanup_before_window;

    size_t frame_bytes = (size_t)rgb_w * rgb_h * WORLD_BACKEND_BYTES_PER_PIXEL;
    size_t rgb_bytes = frame_bytes * (size_t)rgb_frames;
    unsigned char *display_rgb = (unsigned char *)malloc(rgb_bytes);
    if (!display_rgb) goto cleanup_before_window;
    memcpy(display_rgb, warm_pixels, rgb_bytes);
    if (headless_smoke) {
        if (headless_out_path && ray_write_ppm_frames(headless_out_path, display_rgb, rgb_frames, rgb_w, rgb_h, WORLD_BACKEND_BYTES_PER_PIXEL)) {
            free(display_rgb);
            free(seed_latent);
            world_runtime_destroy(rt);
            return 1;
        }
        fprintf(stderr,
                "headless smoke: resident runtime produced %d %s frame(s) %dx%d in %.3fs%s\n",
                rgb_frames, WORLD_BACKEND_OUTPUT_LABEL, rgb_w, rgb_h, warm_seconds,
                headless_out_path ? ", wrote debug frames" : ", no image files written");
        free(display_rgb);
        free(seed_latent);
        world_runtime_destroy(rt);
        return 0;
    }

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(window_width, window_height, "worldmodel.cu");
    SetExitKey(KEY_ESCAPE);
    DisableCursor();
    SetTargetFPS(60);

    int latest_frame = rgb_frames - 1;
    Image image = {display_rgb + (size_t)latest_frame * frame_bytes, rgb_w, rgb_h, 1, WORLD_RAYLIB_PIXEL_FORMAT};
    Texture2D texture = LoadTextureFromImage(image);
    UpdateTexture(texture, display_rgb + (size_t)latest_frame * frame_bytes);

    LiveShared shared;
    memset(&shared, 0, sizeof(shared));
    if (world_mutex_init(&shared.mutex) != 0) {
        UnloadTexture(texture);
        CloseWindow();
        free(display_rgb);
        goto cleanup_runtime_only;
    }
    shared.rt = rt;
    shared.ctrl_dim = ctrl_dim;
    shared.n_buttons = cfg.n_buttons;
    shared.rgb_bytes = rgb_bytes;
    shared.width = rgb_w;
    shared.height = rgb_h;
    shared.frames = rgb_frames;
    shared.target_control_seconds = cfg.inference_fps > 0 ? (float)rgb_frames / (float)cfg.inference_fps : 0.0f;
    shared.control = (float *)calloc((size_t)ctrl_dim, sizeof(float));
    shared.rgb = (unsigned char *)malloc(rgb_bytes);
    if (!shared.control || !shared.rgb) {
        world_mutex_destroy(&shared.mutex);
        UnloadTexture(texture);
        CloseWindow();
        free(display_rgb);
        goto cleanup_runtime_only;
    }

    WorldThread worker;
    if (world_thread_create(&worker, generation_worker, &shared) != 0) {
        world_mutex_destroy(&shared.mutex);
        UnloadTexture(texture);
        CloseWindow();
        free(shared.control);
        free(shared.rgb);
        free(display_rgb);
        goto cleanup_runtime_only;
    }

    float last_generation_seconds = warm_seconds;
    int chunk_id = 0;
    int playback_frame = rgb_frames - 1;
    float playback_timer = 0.0f;
    float playback_interval = playback_interval_seconds(warm_seconds, rgb_frames);
    float *frame_control = (float *)malloc((size_t)ctrl_dim * sizeof(float));
    if (!frame_control) {
        world_mutex_lock(&shared.mutex);
        shared.failed = 1;
        shared.stop = 1;
        world_mutex_unlock(&shared.mutex);
    }
    while (!WindowShouldClose()) {
        if (!frame_control) break;
        float frame_seconds = GetFrameTime();
        fill_raylib_control(frame_control, ctrl_dim, cfg.n_buttons, mouse_scale);
        world_mutex_lock(&shared.mutex);
        merge_frame_control(&shared, frame_control, frame_seconds);
        int failed = shared.failed;
        if (shared.ready) {
            memcpy(display_rgb, shared.rgb, rgb_bytes);
            last_generation_seconds = shared.generation_seconds;
            chunk_id = shared.produced_chunks;
            shared.ready = 0;
            playback_frame = 0;
            playback_timer = 0.0f;
            playback_interval = playback_interval_seconds(last_generation_seconds, rgb_frames);
            UpdateTexture(texture, display_rgb);
        }
        world_mutex_unlock(&shared.mutex);
        if (failed) break;

        if (playback_frame < rgb_frames - 1) {
            playback_timer += GetFrameTime();
            if (playback_timer >= playback_interval) {
                int advance = (int)(playback_timer / playback_interval);
                playback_timer -= (float)advance * playback_interval;
                playback_frame += advance;
                if (playback_frame >= rgb_frames - 1) {
                    playback_frame = rgb_frames - 1;
                    playback_timer = 0.0f;
                }
                UpdateTexture(texture, display_rgb + (size_t)playback_frame * frame_bytes);
            }
        }

        char title[256];
        snprintf(title, sizeof(title), "worldmodel.cu %s | chunk %d | frame %d/%d | gen %.2fs | %.2f %s fps",
                 WORLD_BACKEND_NAME,
                 chunk_id, playback_frame + 1, rgb_frames, last_generation_seconds,
                 rgb_frames / fmaxf(last_generation_seconds, 1.0e-6f),
                 WORLD_BACKEND_OUTPUT_LABEL);
        SetWindowTitle(title);

        BeginDrawing();
        ClearBackground(BLACK);
        Rectangle src = {0.0f, 0.0f, (float)rgb_w, (float)rgb_h};
        Rectangle dst = fit_rect(GetScreenWidth(), GetScreenHeight(), rgb_w, rgb_h);
        Vector2 origin = {0.0f, 0.0f};
        DrawTexturePro(texture, src, dst, origin, 0.0f, WHITE);
        EndDrawing();
    }

    world_mutex_lock(&shared.mutex);
    shared.stop = 1;
    int final_failed = shared.failed;
    world_mutex_unlock(&shared.mutex);
    world_thread_join(worker);
    world_mutex_destroy(&shared.mutex);
    free(frame_control);
    free(shared.control);
    free(shared.rgb);
    UnloadTexture(texture);
    CloseWindow();
    free(display_rgb);
    rc = final_failed ? 1 : 0;

cleanup_runtime_only:
    free(seed_latent);
    world_runtime_destroy(rt);
    return rc;

cleanup_before_window:
    safetensors_close(&st);
    free_loaded_model(&model);
    free_vae_decoder_weights(&vae);
    free(seed_latent);
    world_runtime_destroy(rt);
    return rc;
}
