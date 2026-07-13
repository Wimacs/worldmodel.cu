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

static void world_thread_sleep_millis(unsigned int millis) {
    Sleep(millis);
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

static void world_thread_sleep_millis(unsigned int millis) {
    struct timespec duration;
    duration.tv_sec = (time_t)(millis / 1000);
    duration.tv_nsec = (long)(millis % 1000) * 1000000L;
    nanosleep(&duration, NULL);
}
#endif

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
    int paused;
    int control_debug_enabled;
    float generation_seconds;
    float *pending_image;
    int pending_image_width;
    int pending_image_height;
    int image_pending;
    int image_loading;
    size_t latent_elems;
    int reset_frame_idx;
    unsigned int seed;
} LiveShared;

enum {
    WORLD_CTRL_LMB = 0x01,
    WORLD_CTRL_RMB = 0x02,
    WORLD_CTRL_MMB = 0x04,
    WORLD_CTRL_X1 = 0x05,
    WORLD_CTRL_X2 = 0x06,
};

typedef struct {
    int ray_key;
    int vk;
} WorldKeyBinding;

#define WORLD_KEY(ray_key_, vk_) {ray_key_, vk_}

/* Owl-Control records RAWKEYBOARD.VKey directly into this 256-wide button space. */
static const WorldKeyBinding WORLD_KEY_BINDINGS[] = {
    WORLD_KEY(KEY_ESCAPE,       -1),
    WORLD_KEY(KEY_F1,           0x70),
    WORLD_KEY(KEY_F2,           0x71),
    WORLD_KEY(KEY_F3,           0x72),
    WORLD_KEY(KEY_F4,           0x73),
    WORLD_KEY(KEY_F5,           0x74),
    WORLD_KEY(KEY_F6,           0x75),
    WORLD_KEY(KEY_F7,           0x76),
    WORLD_KEY(KEY_F8,           0x77),
    WORLD_KEY(KEY_F9,           0x78),
    WORLD_KEY(KEY_F10,          0x79),
    WORLD_KEY(KEY_F11,          0x7A),
    WORLD_KEY(KEY_F12,          0x7B),
    WORLD_KEY(KEY_PRINT_SCREEN, 0x2C),
    WORLD_KEY(KEY_SCROLL_LOCK,  0x91),
    WORLD_KEY(KEY_PAUSE,        0x13),
    WORLD_KEY(KEY_KP_EQUAL,     0xBB),

    WORLD_KEY(KEY_GRAVE,        0xC0),
    WORLD_KEY(KEY_ONE,          0x31),
    WORLD_KEY(KEY_TWO,          0x32),
    WORLD_KEY(KEY_THREE,        0x33),
    WORLD_KEY(KEY_FOUR,         0x34),
    WORLD_KEY(KEY_FIVE,         0x35),
    WORLD_KEY(KEY_SIX,          0x36),
    WORLD_KEY(KEY_SEVEN,        0x37),
    WORLD_KEY(KEY_EIGHT,        0x38),
    WORLD_KEY(KEY_NINE,         0x39),
    WORLD_KEY(KEY_ZERO,         0x30),
    WORLD_KEY(KEY_MINUS,        0xBD),
    WORLD_KEY(KEY_EQUAL,        0xBB),
    WORLD_KEY(KEY_BACKSPACE,    0x08),
    WORLD_KEY(KEY_INSERT,       0x2D),
    WORLD_KEY(KEY_HOME,         0x24),
    WORLD_KEY(KEY_PAGE_UP,      0x21),
    WORLD_KEY(KEY_NUM_LOCK,     0x90),
    WORLD_KEY(KEY_KP_DIVIDE,    0x6F),
    WORLD_KEY(KEY_KP_MULTIPLY,  0x6A),
    WORLD_KEY(KEY_KP_SUBTRACT,  0x6D),

    WORLD_KEY(KEY_TAB,          0x09),
    WORLD_KEY(KEY_Q,            0x51),
    WORLD_KEY(KEY_W,            0x57),
    WORLD_KEY(KEY_E,            0x45),
    WORLD_KEY(KEY_R,            0x52),
    WORLD_KEY(KEY_T,            0x54),
    WORLD_KEY(KEY_Y,            0x59),
    WORLD_KEY(KEY_U,            0x55),
    WORLD_KEY(KEY_I,            0x49),
    WORLD_KEY(KEY_O,            0x4F),
    WORLD_KEY(KEY_P,            0x50),
    WORLD_KEY(KEY_LEFT_BRACKET, 0xDB),
    WORLD_KEY(KEY_RIGHT_BRACKET,0xDD),
    WORLD_KEY(KEY_BACKSLASH,    0xDC),
    WORLD_KEY(KEY_DELETE,       0x2E),
    WORLD_KEY(KEY_END,          0x23),
    WORLD_KEY(KEY_PAGE_DOWN,    0x22),
    WORLD_KEY(KEY_KP_7,         0x67),
    WORLD_KEY(KEY_KP_8,         0x68),
    WORLD_KEY(KEY_KP_9,         0x69),
    WORLD_KEY(KEY_KP_ADD,       0x6B),

    WORLD_KEY(KEY_CAPS_LOCK,    0x14),
    WORLD_KEY(KEY_A,            0x41),
    WORLD_KEY(KEY_S,            0x53),
    WORLD_KEY(KEY_D,            0x44),
    WORLD_KEY(KEY_F,            0x46),
    WORLD_KEY(KEY_G,            0x47),
    WORLD_KEY(KEY_H,            0x48),
    WORLD_KEY(KEY_J,            0x4A),
    WORLD_KEY(KEY_K,            0x4B),
    WORLD_KEY(KEY_L,            0x4C),
    WORLD_KEY(KEY_SEMICOLON,    0xBA),
    WORLD_KEY(KEY_APOSTROPHE,   0xDE),
    WORLD_KEY(KEY_ENTER,        0x0D),
    WORLD_KEY(KEY_KP_4,         0x64),
    WORLD_KEY(KEY_KP_5,         0x65),
    WORLD_KEY(KEY_KP_6,         0x66),

    WORLD_KEY(KEY_LEFT_SHIFT,   0x10),
    WORLD_KEY(KEY_Z,            0x5A),
    WORLD_KEY(KEY_X,            0x58),
    WORLD_KEY(KEY_C,            0x43),
    WORLD_KEY(KEY_V,            0x56),
    WORLD_KEY(KEY_B,            0x42),
    WORLD_KEY(KEY_N,            0x4E),
    WORLD_KEY(KEY_M,            0x4D),
    WORLD_KEY(KEY_COMMA,        0xBC),
    WORLD_KEY(KEY_PERIOD,       0xBE),
    WORLD_KEY(KEY_SLASH,        0xBF),
    WORLD_KEY(KEY_RIGHT_SHIFT,  0x10),
    WORLD_KEY(KEY_UP,           0x26),
    WORLD_KEY(KEY_KP_1,         0x61),
    WORLD_KEY(KEY_KP_2,         0x62),
    WORLD_KEY(KEY_KP_3,         0x63),
    WORLD_KEY(KEY_KP_ENTER,     0x0D),

    WORLD_KEY(KEY_LEFT_CONTROL, 0x11),
    WORLD_KEY(KEY_LEFT_SUPER,   0x5B),
    WORLD_KEY(KEY_LEFT_ALT,     0x12),
    WORLD_KEY(KEY_SPACE,        0x20),
    WORLD_KEY(KEY_RIGHT_ALT,    0x12),
    WORLD_KEY(KEY_RIGHT_SUPER,  0x5C),
    WORLD_KEY(KEY_KB_MENU,      0x5D),
    WORLD_KEY(KEY_RIGHT_CONTROL,0x11),
    WORLD_KEY(KEY_LEFT,         0x25),
    WORLD_KEY(KEY_DOWN,         0x28),
    WORLD_KEY(KEY_RIGHT,        0x27),
    WORLD_KEY(KEY_KP_0,         0x60),
    WORLD_KEY(KEY_KP_DECIMAL,   0x6E),
};

#undef WORLD_KEY

#define WORLD_KEY_BINDING_COUNT \
    (sizeof(WORLD_KEY_BINDINGS) / sizeof(WORLD_KEY_BINDINGS[0]))

static void ray_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [MODEL_DIR|MODEL.safetensors] [IMAGE] [--model-dir DIR] [--weights FILE] [--vae-weights FILE] [--seed-image FILE] [--seed-latent FILE] [--steps N] [--layers N] [--cache-window N] [--fast-realtime] [--frame-idx N] [--seed N] [--noise normal|uniform] [--mouse-scale X] [--window-width N] [--window-height N] [--warmup N] [--headless-smoke] [--headless-generate N] [--headless-reset-check] [--headless-out PATH] [--headless-mouse X Y] [--headless-button N]\n"
            "\n"
            "Defaults: full model, full denoise schedule, cache window 8, mouse scale 30.0.\n",
            argv0);
}

static int ray_path_has_suffix(const char *path, const char *suffix) {
    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);
    return path_len >= suffix_len && strcmp(path + path_len - suffix_len, suffix) == 0;
}

static int ray_dirname(char *out, size_t out_size, const char *path) {
    size_t len = strlen(path);
    while (len > 1 && (path[len - 1] == '/' || path[len - 1] == '\\')) --len;
    size_t slash = len;
    while (slash > 0 && path[slash - 1] != '/' && path[slash - 1] != '\\') --slash;
    if (slash == 0) {
        if (out_size < 2) return 1;
        memcpy(out, ".", 2);
        return 0;
    }
    size_t dir_len = slash - 1;
    if (dir_len == 0 || (dir_len == 2 && path[1] == ':')) ++dir_len;
    if (dir_len + 1 > out_size) return 1;
    memcpy(out, path, dir_len);
    out[dir_len] = '\0';
    return 0;
}

static int ray_file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void ray_enable_default_cuda_fmha(void) {
#if !defined(WORLD_BACKEND_VULKAN) || !WORLD_BACKEND_VULKAN
    if (getenv("WORLD_ATTN_D64_FMHA")) return;
#ifdef _WIN32
    if (_putenv_s("WORLD_ATTN_D64_FMHA", "1") == 0) {
#else
    if (setenv("WORLD_ATTN_D64_FMHA", "1", 0) == 0) {
#endif
        fprintf(stderr, "raylib default: WORLD_ATTN_D64_FMHA=1\n");
    }
#endif
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

static void fill_raylib_control(float *control, int ctrl_dim, int n_buttons, float mouse_scale) {
    memset(control, 0, (size_t)ctrl_dim * sizeof(float));
    Vector2 delta = GetMouseDelta();
    if (ctrl_dim >= n_buttons + 3) {
        control[0] = delta.x * mouse_scale * 0.01f;
        control[1] = delta.y * mouse_scale * 0.01f;
        for (size_t i = 0; i < WORLD_KEY_BINDING_COUNT; ++i) {
            const WorldKeyBinding *binding = &WORLD_KEY_BINDINGS[i];
            if (binding->vk >= 0 && IsKeyDown(binding->ray_key)) {
                set_button_vk(control, n_buttons, binding->vk, 1);
            }
        }
        set_button_vk(control, n_buttons, WORLD_CTRL_LMB, IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        set_button_vk(control, n_buttons, WORLD_CTRL_RMB, IsMouseButtonDown(MOUSE_BUTTON_RIGHT));
        set_button_vk(control, n_buttons, WORLD_CTRL_MMB, IsMouseButtonDown(MOUSE_BUTTON_MIDDLE));
        set_button_vk(control, n_buttons, WORLD_CTRL_X1, IsMouseButtonDown(MOUSE_BUTTON_SIDE));
        set_button_vk(control, n_buttons, WORLD_CTRL_X2, IsMouseButtonDown(MOUSE_BUTTON_EXTRA));
        control[2 + n_buttons] = clamp_scroll_sign(GetMouseWheelMove());
    }
}

static void merge_frame_control(LiveShared *s, const float *frame_control) {
    if (s->ctrl_dim < s->n_buttons + 3) return;
    s->control[0] += frame_control[0];
    s->control[1] += frame_control[1];
    for (int i = 0; i < s->n_buttons; ++i) {
        s->control[2 + i] = frame_control[2 + i];
    }
    s->control[2 + s->n_buttons] =
        clamp_scroll_sign(s->control[2 + s->n_buttons] + frame_control[2 + s->n_buttons]);
}

static float *ray_resize_rgb_bilinear(
        const float *src,
        int src_w,
        int src_h,
        int dst_w,
        int dst_h) {
    if (!src || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return NULL;
    size_t elems = (size_t)dst_w * dst_h * 3;
    float *dst = (float *)malloc(elems * sizeof(float));
    if (!dst) return NULL;
    if (src_w == dst_w && src_h == dst_h) {
        memcpy(dst, src, elems * sizeof(float));
        return dst;
    }
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;
    for (int y = 0; y < dst_h; ++y) {
        float sy = ((float)y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        if (y0 < 0) {
            y0 = 0;
            fy = 0.0f;
        }
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;
        for (int x = 0; x < dst_w; ++x) {
            float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            if (x0 < 0) {
                x0 = 0;
                fx = 0.0f;
            }
            int x1 = x0 + 1;
            if (x1 >= src_w) x1 = src_w - 1;
            const float *p00 = src + ((size_t)y0 * src_w + x0) * 3;
            const float *p01 = src + ((size_t)y0 * src_w + x1) * 3;
            const float *p10 = src + ((size_t)y1 * src_w + x0) * 3;
            const float *p11 = src + ((size_t)y1 * src_w + x1) * 3;
            float *out = dst + ((size_t)y * dst_w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float top = p00[c] + (p01[c] - p00[c]) * fx;
                float bottom = p10[c] + (p11[c] - p10[c]) * fx;
                out[c] = top + (bottom - top) * fy;
            }
        }
    }
    return dst;
}

static int ray_load_seed_image(
        const char *path,
        int raw_w,
        int raw_h,
        float **rgb_out) {
    *rgb_out = NULL;
    Image image = LoadImage(path);
    if (!image.data || image.width <= 0 || image.height <= 0) {
        fprintf(stderr, "failed to load seed image: %s\n", path);
        return 1;
    }
    Color *colors = LoadImageColors(image);
    if (!colors) {
        UnloadImage(image);
        return 1;
    }
    int crop_x = 0;
    int crop_y = 0;
    int crop_w = image.width;
    int crop_h = image.height;
    if ((int64_t)image.width * 9 > (int64_t)image.height * 16) {
        crop_w = (image.height * 16) / 9;
        crop_x = (image.width - crop_w) / 2;
    } else if ((int64_t)image.width * 9 < (int64_t)image.height * 16) {
        crop_h = (image.width * 9) / 16;
        crop_y = (image.height - crop_h) / 2;
    }
    float *cropped = (float *)malloc((size_t)crop_w * crop_h * 3 * sizeof(float));
    if (!cropped) {
        UnloadImageColors(colors);
        UnloadImage(image);
        return 1;
    }
    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            Color p = colors[(size_t)(crop_y + y) * image.width + crop_x + x];
            float *dst = cropped + ((size_t)y * crop_w + x) * 3;
            dst[0] = (float)p.r / 255.0f;
            dst[1] = (float)p.g / 255.0f;
            dst[2] = (float)p.b / 255.0f;
        }
    }
    UnloadImageColors(colors);
    UnloadImage(image);

    int visible_w = raw_w * 5 / 4;
    int visible_h = raw_h * 45 / 32;
    float *visible = ray_resize_rgb_bilinear(cropped, crop_w, crop_h, visible_w, visible_h);
    free(cropped);
    if (!visible) return 1;
    float *raw = ray_resize_rgb_bilinear(visible, visible_w, visible_h, raw_w, raw_h);
    free(visible);
    if (!raw) return 1;
    const char *dump_path = getenv("WORLD_DUMP_VAE_INPUT");
    if (dump_path && dump_path[0]) {
        FILE *dump = fopen(dump_path, "wb");
        size_t elems = (size_t)raw_w * raw_h * 3;
        if (!dump || fwrite(raw, sizeof(float), elems, dump) != elems) {
            fprintf(stderr, "failed to write VAE input dump: %s\n", dump_path);
            if (dump) fclose(dump);
            free(raw);
            return 1;
        }
        fclose(dump);
    }
    *rgb_out = raw;
    fprintf(stderr,
            "seed image prepared: %s center-crop=%dx%d visible=%dx%d VAE_RGB=%dx%d\n",
            path, crop_w, crop_h, visible_w, visible_h, raw_w, raw_h);
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
        int paused = s->paused;
        float *pending_image = NULL;
        int pending_width = 0;
        int pending_height = 0;
        if (!stop && s->image_pending) {
            pending_image = s->pending_image;
            pending_width = s->pending_image_width;
            pending_height = s->pending_image_height;
            s->pending_image = NULL;
            s->image_pending = 0;
            s->image_loading = 1;
            s->ready = 0;
            memset(s->control, 0, (size_t)s->ctrl_dim * sizeof(float));
        }
        if (stop || (paused && !pending_image)) {
            world_mutex_unlock(&s->mutex);
            if (stop) break;
            world_thread_sleep_millis(4);
            continue;
        }
        if (pending_image) {
            world_mutex_unlock(&s->mutex);
            float *latent = (float *)malloc(s->latent_elems * sizeof(float));
            float encode_seconds = 0.0f;
            const unsigned char *pixels = NULL;
            int width = 0;
            int height = 0;
            int frames = 0;
            float seconds = 0.0f;
            memset(control, 0, (size_t)s->ctrl_dim * sizeof(float));
            int image_failed = !latent ||
                world_runtime_encode_image_rgb(s->rt, pending_image,
                    pending_width, pending_height, latent, &encode_seconds) ||
                world_runtime_reset(s->rt, s->reset_frame_idx, s->seed) ||
                world_runtime_seed_latent_pixels(s->rt, latent, control,
                    &pixels, &width, &height, &frames, &seconds);
            free(latent);
            free(pending_image);

            size_t bytes = (size_t)width * height * WORLD_BACKEND_BYTES_PER_PIXEL * frames;
            world_mutex_lock(&s->mutex);
            s->image_loading = 0;
            int superseded = s->image_pending;
            if (!image_failed && superseded) {
                fprintf(stderr, "discarded superseded seed image\n");
            } else if (!image_failed && pixels && bytes == s->rgb_bytes) {
                memcpy(s->rgb, pixels, bytes);
                s->width = width;
                s->height = height;
                s->frames = frames;
                s->generation_seconds = seconds;
                s->ready = 1;
                fprintf(stderr, "seed image active: encode=%.3fms cache+decode=%.3fms\n",
                        encode_seconds * 1000.0f, seconds * 1000.0f);
            } else {
                s->failed = 1;
                s->stop = 1;
            }
            world_mutex_unlock(&s->mutex);
            if (image_failed) break;
            if (superseded) continue;

            for (;;) {
                world_mutex_lock(&s->mutex);
                int consumed = !s->ready || s->stop;
                world_mutex_unlock(&s->mutex);
                if (consumed) break;
                world_thread_sleep_millis(1);
            }
            continue;
        }
        memcpy(control, s->control, (size_t)s->ctrl_dim * sizeof(float));
        if (s->ctrl_dim >= s->n_buttons + 3) {
            control[0] = clamp_mouse_axis(control[0]);
            control[1] = clamp_mouse_axis(control[1]);
            s->control[0] = 0.0f;
            s->control[1] = 0.0f;
            s->control[2 + s->n_buttons] = 0.0f;
        }
        world_mutex_unlock(&s->mutex);
        if (s->control_debug_enabled && s->ctrl_dim >= s->n_buttons + 3) {
            int active_buttons = 0;
            char active_ids[256];
            size_t active_len = 0;
            active_ids[0] = '\0';
            for (int i = 0; i < s->n_buttons; ++i) {
                if (control[2 + i] > 0.5f) {
                    if (active_len + 8 < sizeof(active_ids)) {
                        int n = snprintf(active_ids + active_len, sizeof(active_ids) - active_len,
                                         "%s%d", active_buttons ? "," : "", i);
                        if (n > 0) active_len += (size_t)n;
                    }
                    ++active_buttons;
                }
            }
            fprintf(stderr,
                    "raylib control: mouse=(%.4f, %.4f) buttons={%s} wheel=%.0f\n",
                    control[0], control[1], active_ids, control[2 + s->n_buttons]);
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
        if (s->image_pending) {
            world_mutex_unlock(&s->mutex);
            continue;
        }
        if (bytes == s->rgb_bytes) {
            memcpy(s->rgb, pixels, bytes);
            s->width = width;
            s->height = height;
            s->frames = frames;
            s->generation_seconds = seconds;
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

static const float WORLD_HUD_SCALE = 2.0f;

static void draw_hud_key(Rectangle key, const char *label, int pressed) {
    float s = WORLD_HUD_SCALE;
    Rectangle shadow = {key.x + s, key.y + s, key.width, key.height};
    DrawRectangleRec(shadow, Fade(BLACK, 0.58f));
    DrawRectangleRec(key, pressed ? GREEN : Fade(BLACK, 0.48f));
    DrawRectangleLinesEx(key, s, pressed ? RAYWHITE : Fade(RAYWHITE, 0.68f));
    int font_size = (int)(10.0f * s);
    int min_font_size = (int)(6.0f * s);
    int text_padding = (int)(4.0f * s);
    while (font_size > min_font_size && MeasureText(label, font_size) > (int)key.width - text_padding) --font_size;
    int text_width = MeasureText(label, font_size);
    DrawText(label,
             (int)(key.x + (key.width - (float)text_width) * 0.5f),
             (int)(key.y + (key.height - (float)font_size) * 0.5f),
             font_size, pressed ? BLACK : RAYWHITE);
}

static void draw_keyboard_hud(int screen_width, int screen_height) {
    float s = WORLD_HUD_SCALE;
    const float key_w = 32.0f * s;
    const float key_h = 28.0f * s;
    const float gap = 4.0f * s;
    float origin_x = (float)screen_width - 230.0f * s;
    float origin_y = (float)screen_height - 142.0f * s;

    draw_hud_key((Rectangle){origin_x + key_w + gap, origin_y, key_w, key_h},
                 "W", IsKeyDown(KEY_W));

    draw_hud_key((Rectangle){origin_x, origin_y + key_h + gap, key_w, key_h},
                 "A", IsKeyDown(KEY_A));
    draw_hud_key((Rectangle){origin_x + key_w + gap, origin_y + key_h + gap, key_w, key_h},
                 "S", IsKeyDown(KEY_S));
    draw_hud_key((Rectangle){origin_x + (key_w + gap) * 2.0f, origin_y + key_h + gap, key_w, key_h},
                 "D", IsKeyDown(KEY_D));

    draw_hud_key((Rectangle){origin_x, origin_y + (key_h + gap) * 2.0f, 52.0f * s, key_h},
                  "Shift", IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT));
    draw_hud_key((Rectangle){origin_x + 56.0f * s, origin_y + (key_h + gap) * 2.0f, 48.0f * s, key_h},
                  "Ctrl", IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL));
    draw_hud_key((Rectangle){origin_x, origin_y + (key_h + gap) * 3.0f, 104.0f * s, key_h},
                  "Space", IsKeyDown(KEY_SPACE));
}

static void draw_mouse_hud(int screen_width, int screen_height, float mouse_x, float mouse_y, float scroll) {
    float s = WORLD_HUD_SCALE;
    Color active = ORANGE;
    Color grid = Fade(RAYWHITE, 0.48f);
    float radius = 30.0f * s;
    Vector2 center = {(float)screen_width - 58.0f * s, (float)screen_height - 76.0f * s};
    DrawCircleV(center, radius, Fade(BLACK, 0.28f));
    DrawLineEx((Vector2){center.x - radius + 7.0f * s, center.y},
               (Vector2){center.x + radius - 7.0f * s, center.y}, s, grid);
    DrawLineEx((Vector2){center.x, center.y - radius + 7.0f * s},
               (Vector2){center.x, center.y + radius - 7.0f * s}, s, grid);
    Vector2 shadow_center = {center.x + 2.0f * s, center.y + 2.0f * s};
    DrawRing(shadow_center, radius - s, radius + s, 0.0f, 360.0f, 48,
              Fade(BLACK, 0.62f));
    DrawRing(center, radius - s, radius, 0.0f, 360.0f, 48, RAYWHITE);

    float magnitude = sqrtf(mouse_x * mouse_x + mouse_y * mouse_y);
    if (magnitude > 1.0e-5f) {
        float nx = mouse_x / magnitude;
        float ny = mouse_y / magnitude;
        float strength = fminf(1.0f, sqrtf(fminf(magnitude, 1.0f)) * 1.6f);
        float vector_length = (radius - 8.0f * s) * strength;
        Vector2 tip = {center.x + nx * vector_length, center.y + ny * vector_length};
        DrawLineEx(center, tip, 3.0f * s, active);
        Vector2 left = {tip.x - nx * 7.0f * s - ny * 4.0f * s,
                        tip.y - ny * 7.0f * s + nx * 4.0f * s};
        Vector2 right = {tip.x - nx * 7.0f * s + ny * 4.0f * s,
                         tip.y - ny * 7.0f * s - nx * 4.0f * s};
        DrawTriangle(tip, right, left, active);
    }
    DrawCircleV(center, 3.0f * s, magnitude > 1.0e-5f ? active : RAYWHITE);

    DrawRectangle((int)(center.x - 3.0f * s), (int)(center.y - 13.0f * s), (int)(6.0f * s), (int)(9.0f * s),
                   scroll > 0.0f ? active : Fade(RAYWHITE, 0.62f));
    DrawRectangle((int)(center.x - 3.0f * s), (int)(center.y + 4.0f * s), (int)(6.0f * s), (int)(9.0f * s),
                   scroll < 0.0f ? active : Fade(RAYWHITE, 0.62f));
    DrawRectangle((int)(center.x - radius - 4.0f * s), (int)(center.y - 9.0f * s), (int)(5.0f * s), (int)(18.0f * s),
                   IsMouseButtonDown(MOUSE_BUTTON_SIDE) ? active : Fade(RAYWHITE, 0.55f));
    DrawRectangle((int)(center.x + radius - s), (int)(center.y - 9.0f * s), (int)(5.0f * s), (int)(18.0f * s),
                   IsMouseButtonDown(MOUSE_BUTTON_EXTRA) ? active : Fade(RAYWHITE, 0.55f));

    draw_hud_key((Rectangle){center.x - 39.0f * s, center.y + radius + 6.0f * s, 24.0f * s, 20.0f * s},
                  "L", IsMouseButtonDown(MOUSE_BUTTON_LEFT));
    draw_hud_key((Rectangle){center.x - 12.0f * s, center.y + radius + 6.0f * s, 24.0f * s, 20.0f * s},
                  "M", IsMouseButtonDown(MOUSE_BUTTON_MIDDLE));
    draw_hud_key((Rectangle){center.x + 15.0f * s, center.y + radius + 6.0f * s, 24.0f * s, 20.0f * s},
                  "R", IsMouseButtonDown(MOUSE_BUTTON_RIGHT));
}

static void draw_live_hud(
        int screen_width,
        int screen_height,
        float mouse_x,
        float mouse_y,
        float scroll,
        int paused,
        int image_loading,
        int rgb_frames,
    float generation_seconds) {
    draw_keyboard_hud(screen_width, screen_height);
    draw_mouse_hud(screen_width, screen_height, mouse_x, mouse_y, scroll);

    float inference_fps = generation_seconds > 0.0f ? (float)rgb_frames / generation_seconds : 0.0f;
    int rounded_fps = (int)(inference_fps + 0.5f);
    Color fps_color = rounded_fps >= 60 ? GREEN : (rounded_fps >= 30 ? ORANGE : RED);
    char fps_text[32];
    snprintf(fps_text, sizeof(fps_text), "%d FPS", rounded_fps);
    float s = WORLD_HUD_SCALE;
    int fps_font_size = (int)(20.0f * s);
    int fps_width = MeasureText(fps_text, fps_font_size);
    int fps_x = screen_width - fps_width - (int)(18.0f * s);
    int fps_y = (int)(18.0f * s);
    DrawText(fps_text, fps_x + (int)(2.0f * s), fps_y + (int)(2.0f * s), fps_font_size, Fade(BLACK, 0.82f));
    DrawText(fps_text, fps_x, fps_y, fps_font_size, fps_color);

    if (paused) {
        const char *pause_text = "PAUSED";
        int pause_font_size = (int)(24.0f * s);
        int pause_text_width = MeasureText(pause_text, pause_font_size);
        int pause_label_x = (int)(31.0f * s);
        int pause_width = pause_label_x + pause_text_width;
        int pause_x = (screen_width - pause_width) / 2;
        int pause_y = screen_height / 2 - pause_font_size / 2;
        DrawRectangle(pause_x + (int)(2.0f * s), pause_y + (int)(3.0f * s), (int)(6.0f * s), (int)(24.0f * s), Fade(BLACK, 0.82f));
        DrawRectangle(pause_x + (int)(14.0f * s), pause_y + (int)(3.0f * s), (int)(6.0f * s), (int)(24.0f * s), Fade(BLACK, 0.82f));
        DrawRectangle(pause_x, pause_y + (int)s, (int)(6.0f * s), (int)(24.0f * s), ORANGE);
        DrawRectangle(pause_x + (int)(12.0f * s), pause_y + (int)s, (int)(6.0f * s), (int)(24.0f * s), ORANGE);
        DrawText(pause_text, pause_x + pause_label_x + (int)(2.0f * s), pause_y + (int)(2.0f * s),
                  pause_font_size, Fade(BLACK, 0.82f));
        DrawText(pause_text, pause_x + pause_label_x, pause_y,
                  pause_font_size, RAYWHITE);
    } else if (image_loading) {
        const char *loading_text = "LOADING IMAGE";
        int font_size = (int)(20.0f * s);
        int text_width = MeasureText(loading_text, font_size);
        int x = (screen_width - text_width) / 2;
        int y = screen_height / 2 - font_size / 2;
        DrawText(loading_text, x + (int)(2.0f * s), y + (int)(2.0f * s), font_size, Fade(BLACK, 0.82f));
        DrawText(loading_text, x, y, font_size, RAYWHITE);
    }
}

int main(int argc, char **argv) {
    const char *model_dir = "../Waypoint-1.5-1B";
    const char *weights = NULL;
    const char *vae_weights = NULL;
    const char *seed_latent_path = NULL;
    const char *seed_image_path = NULL;
    const char *headless_out_path = NULL;
    int steps_to_run = -1;
    int layers_to_run = -1;
    int frame_idx = 0;
    int window_width = 1600;
    int window_height = 800;
    int warmup_chunks = 1;
    int headless_smoke = 0;
    int headless_generate_chunks = 0;
    int headless_reset_check = 0;
    int fast_realtime = 0;
    int cache_window_override = 8;
    int cache_window_explicit = 0;
    int headless_has_mouse = 0;
    float headless_mouse_x = 0.0f;
    float headless_mouse_y = 0.0f;
    int headless_button = -1;
    float mouse_scale = 30.0f;
    unsigned int seed = 1234;
    int noise_mode = WORLD_NOISE_NORMAL;
    int positional_model_seen = 0;
    int positional_image_seen = 0;
    char positional_model_dir[PATH_BUF];

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
            positional_model_seen = 1;
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
            positional_model_seen = 1;
        } else if (strcmp(argv[i], "--vae-weights") == 0 && i + 1 < argc) {
            vae_weights = argv[++i];
        } else if (strcmp(argv[i], "--seed-latent") == 0 && i + 1 < argc) {
            seed_latent_path = argv[++i];
        } else if (strcmp(argv[i], "--seed-image") == 0 && i + 1 < argc) {
            seed_image_path = argv[++i];
            positional_image_seen = 1;
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            layers_to_run = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache-window") == 0 && i + 1 < argc) {
            cache_window_override = atoi(argv[++i]);
            cache_window_explicit = 1;
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
        } else if (strcmp(argv[i], "--headless-generate") == 0 && i + 1 < argc) {
            headless_generate_chunks = atoi(argv[++i]);
            if (headless_generate_chunks < 0) {
                fprintf(stderr, "invalid --headless-generate %d\n", headless_generate_chunks);
                return 1;
            }
        } else if (strcmp(argv[i], "--headless-reset-check") == 0) {
            headless_reset_check = 1;
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
        } else if (argv[i][0] != '-' && !positional_model_seen) {
            positional_model_seen = 1;
            if (ray_path_has_suffix(argv[i], ".safetensors")) {
                weights = argv[i];
                if (ray_dirname(positional_model_dir, sizeof(positional_model_dir), argv[i])) {
                    fprintf(stderr, "model weight path is too long: %s\n", argv[i]);
                    return 1;
                }
                model_dir = positional_model_dir;
            } else {
                model_dir = argv[i];
            }
        } else if (argv[i][0] != '-' && !positional_image_seen) {
            positional_image_seen = 1;
            seed_image_path = argv[i];
        } else {
            ray_usage(argv[0]);
            return 1;
        }
    }

    if (seed_latent_path && seed_image_path) {
        fprintf(stderr, "--seed-latent and --seed-image cannot be used together\n");
        return 1;
    }

    ray_enable_default_cuda_fmha();

    char config_path[PATH_BUF];
    char default_weights[PATH_BUF];
    char default_vae_weights[PATH_BUF];
    char model_parent[PATH_BUF];
    char sibling_model_dir[PATH_BUF];
    char sibling_vae_weights[PATH_BUF];
    if (join_path(config_path, sizeof(config_path), model_dir, "config.yaml")) return 1;
    if (!weights) {
        if (join_path(default_weights, sizeof(default_weights), model_dir, "model.safetensors")) return 1;
        weights = default_weights;
    }
    if (!vae_weights) {
        if (join_path(default_vae_weights, sizeof(default_vae_weights), model_dir, "vae/diffusion_pytorch_model.safetensors")) return 1;
        if (ray_file_exists(default_vae_weights)) {
            vae_weights = default_vae_weights;
        } else if (!ray_dirname(model_parent, sizeof(model_parent), model_dir) &&
                   !join_path(sibling_model_dir, sizeof(sibling_model_dir), model_parent, "Waypoint-1.5-1B") &&
                   !join_path(sibling_vae_weights, sizeof(sibling_vae_weights), sibling_model_dir,
                              "vae/diffusion_pytorch_model.safetensors") &&
                   ray_file_exists(sibling_vae_weights)) {
            vae_weights = sibling_vae_weights;
            fprintf(stderr, "using sibling VAE weights: %s\n", vae_weights);
        } else {
            vae_weights = default_vae_weights;
        }
    }

    WorldConfig cfg;
    if (world_config_load(&cfg, config_path)) return 1;
    if (fast_realtime) {
        if (steps_to_run < 0) steps_to_run = 1;
        if (!cache_window_explicit) cache_window_override = 2;
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
    WorldVaeEncoderWeights vae_encoder;
    WorldRuntime *rt = NULL;
    float *seed_latent = NULL;
    float *seed_image_rgb = NULL;
    memset(&st, 0, sizeof(st));
    memset(&model, 0, sizeof(model));
    memset(&vae, 0, sizeof(vae));
    memset(&vae_encoder, 0, sizeof(vae_encoder));
    int rc = 1;

    size_t seed_latent_elems = (size_t)cfg.channels * (size_t)(cfg.height * cfg.patch_h) * (size_t)(cfg.width * cfg.patch_w);
    int vae_image_w = cfg.width * cfg.patch_w * 16;
    int vae_image_h = cfg.height * cfg.patch_h * 16;
    if (seed_latent_path) {
        if (read_f32_file_exact(seed_latent_path, seed_latent_elems, &seed_latent)) goto cleanup_before_window;
        fprintf(stderr, "loaded seed latent: %s elems=%zu\n", seed_latent_path, seed_latent_elems);
    }
    if (seed_image_path) {
        if (ray_load_seed_image(seed_image_path, vae_image_w, vae_image_h, &seed_image_rgb)) {
            goto cleanup_before_window;
        }
    }

    fprintf(stderr, "loading transformer safetensors: %s\n", weights);
    if (safetensors_open(&st, weights)) goto cleanup_before_window;
    if (load_live_model_weights(&st, &cfg, layers_to_run, &model)) goto cleanup_before_window;
    safetensors_close(&st);
    memset(&st, 0, sizeof(st));
    if (load_vae_decoder_weights(vae_weights, &vae)) goto cleanup_before_window;
    if (load_vae_encoder_weights(vae_weights, &vae_encoder)) goto cleanup_before_window;

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
        if (world_runtime_init_vae_encoder(prewarm_rt, &vae_encoder)) {
            world_runtime_destroy(prewarm_rt);
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        {
            size_t image_elems = (size_t)vae_image_w * vae_image_h * 3;
            float *encoder_warm_image = seed_image_rgb;
            float *encoder_warm_latent = seed_image_rgb ? seed_latent : NULL;
            int free_warm_image = 0;
            int free_warm_latent = 0;
            if (!encoder_warm_image) {
                encoder_warm_image = (float *)calloc(image_elems, sizeof(float));
                free_warm_image = 1;
            }
            if (!encoder_warm_latent) {
                encoder_warm_latent = (float *)malloc(seed_latent_elems * sizeof(float));
                if (seed_image_rgb) {
                    seed_latent = encoder_warm_latent;
                } else {
                    free_warm_latent = 1;
                }
            }
            float encoder_seconds = 0.0f;
            if (!encoder_warm_image || !encoder_warm_latent ||
                    world_runtime_encode_image_rgb(prewarm_rt, encoder_warm_image,
                        vae_image_w, vae_image_h, encoder_warm_latent, &encoder_seconds)) {
                if (free_warm_image) free(encoder_warm_image);
                if (free_warm_latent) free(encoder_warm_latent);
                world_runtime_destroy(prewarm_rt);
                free(seed_control);
                free(warm_control);
                goto cleanup_before_window;
            }
            fprintf(stderr, "prewarmed image encoder in %.3fms%s\n",
                    encoder_seconds * 1000.0f, seed_image_rgb ? " using startup image" : " using black RGB");
            if (free_warm_image) free(encoder_warm_image);
            if (free_warm_latent) free(encoder_warm_latent);
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
    if (world_runtime_init_vae_encoder(rt, &vae_encoder)) {
        free(seed_control);
        free(warm_control);
        goto cleanup_before_window;
    }
    if (seed_image_rgb && !seed_latent) {
        float encoder_seconds = 0.0f;
        seed_latent = (float *)malloc(seed_latent_elems * sizeof(float));
        if (!seed_latent || world_runtime_encode_image_rgb(rt, seed_image_rgb,
                    vae_image_w, vae_image_h, seed_latent, &encoder_seconds)) {
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        fprintf(stderr, "encoded startup image in resident runtime: %.3fms\n",
                encoder_seconds * 1000.0f);
    }
    free_loaded_model(&model);
    free_vae_decoder_weights(&vae);
    free_vae_encoder_weights(&vae_encoder);

    if (seed_latent) {
        fprintf(stderr, "seeding runtime KV cache from latent\n");
        if (world_runtime_seed_latent_pixels(rt, seed_latent, seed_control, &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds)) {
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        fprintf(stderr, "generating initial chunk after seed cache pass\n");
        if (world_runtime_step_pixels(rt, seed_control, &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds)) {
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
    if (headless_reset_check) {
        if (!seed_image_rgb || !warm_pixels || rgb_w <= 0 || rgb_h <= 0 || rgb_frames <= 0) {
            fprintf(stderr, "--headless-reset-check requires a valid input image\n");
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        int expected_w = rgb_w;
        int expected_h = rgb_h;
        int expected_frames = rgb_frames;
        size_t check_bytes = (size_t)rgb_w * rgb_h * WORLD_BACKEND_BYTES_PER_PIXEL * rgb_frames;
        unsigned char *expected_pixels = (unsigned char *)malloc(check_bytes);
        float *check_latent = (float *)malloc(seed_latent_elems * sizeof(float));
        if (!expected_pixels || !check_latent) {
            free(expected_pixels);
            free(check_latent);
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        memcpy(expected_pixels, warm_pixels, check_bytes);
        float encode_seconds = 0.0f;
        int check_failed = world_runtime_encode_image_rgb(rt, seed_image_rgb,
                    vae_image_w, vae_image_h, check_latent, &encode_seconds) ||
            world_runtime_reset(rt, frame_idx, seed) ||
            world_runtime_seed_latent_pixels(rt, check_latent, seed_control,
                    &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds) ||
            world_runtime_step_pixels(rt, seed_control,
                    &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds);
        if (!check_failed &&
                (rgb_w != expected_w || rgb_h != expected_h || rgb_frames != expected_frames)) {
            fprintf(stderr,
                    "runtime reset check changed output shape: %dx%dx%d -> %dx%dx%d\n",
                    expected_w, expected_h, expected_frames, rgb_w, rgb_h, rgb_frames);
            check_failed = 1;
        }
        if (!check_failed && memcmp(expected_pixels, warm_pixels, check_bytes) != 0) {
            size_t mismatches = 0;
            int max_diff = 0;
            for (size_t i = 0; i < check_bytes; ++i) {
                int diff = abs((int)expected_pixels[i] - (int)warm_pixels[i]);
                if (diff) ++mismatches;
                if (diff > max_diff) max_diff = diff;
            }
            fprintf(stderr, "runtime reset check failed: mismatches=%zu max_u8_diff=%d\n",
                    mismatches, max_diff);
            check_failed = 1;
        }
        free(expected_pixels);
        free(check_latent);
        if (check_failed) {
            free(seed_control);
            free(warm_control);
            goto cleanup_before_window;
        }
        fprintf(stderr,
                "runtime reset check passed: repeated seeded rollout is byte-identical (encode %.3fms)\n",
                encode_seconds * 1000.0f);
    }
    if (headless_smoke) {
        for (int i = 0; i < headless_generate_chunks; ++i) {
            fprintf(stderr, "headless generated chunk %d/%d\n", i + 1, headless_generate_chunks);
            if (world_runtime_step_pixels(rt, warm_control, &warm_pixels, &rgb_w, &rgb_h, &rgb_frames, &warm_seconds)) {
                free(seed_control);
                free(warm_control);
                goto cleanup_before_window;
            }
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
            free(seed_image_rgb);
            free(seed_latent);
            world_runtime_destroy(rt);
            return 1;
        }
        fprintf(stderr,
                "headless smoke: resident runtime produced %d %s frame(s) %dx%d in %.3fs%s\n",
                rgb_frames, WORLD_BACKEND_OUTPUT_LABEL, rgb_w, rgb_h, warm_seconds,
                headless_out_path ? ", wrote debug frames" : ", no image files written");
        free(display_rgb);
        free(seed_image_rgb);
        free(seed_latent);
        world_runtime_destroy(rt);
        return 0;
    }

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(window_width, window_height, "worldmodel.c");
    SetWindowMinSize(640, 360);
    MaximizeWindow();
    SetExitKey(KEY_NULL);
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
    {
        const char *control_debug_env = getenv("WORLD_CONTROL_DEBUG");
        shared.control_debug_enabled = control_debug_env ? control_debug_env[0] != '0' : 0;
        if (shared.control_debug_enabled) {
            fprintf(stderr, "raylib control debug enabled by WORLD_CONTROL_DEBUG=1\n");
        }
    }
    shared.latent_elems = seed_latent_elems;
    shared.reset_frame_idx = frame_idx;
    shared.seed = seed;
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
    int playback_frame = rgb_frames - 1;
    float playback_timer = 0.0f;
    float playback_interval = playback_interval_seconds(warm_seconds, rgb_frames);
    float hud_mouse_x = 0.0f;
    float hud_mouse_y = 0.0f;
    int paused = 0;
    int image_loading = 0;
    int suppress_mouse_frames = 0;
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
        int pause_toggled = IsKeyPressed(KEY_ESCAPE);
        if (pause_toggled) {
            paused = !paused;
            if (paused) {
                EnableCursor();
            } else {
                DisableCursor();
                suppress_mouse_frames = 2;
            }
        }
        if (IsFileDropped()) {
            FilePathList dropped = LoadDroppedFiles();
            float *dropped_rgb = NULL;
            const char *loaded_path = NULL;
            for (unsigned int i = 0; i < dropped.count && !dropped_rgb; ++i) {
                if (ray_load_seed_image(dropped.paths[i], vae_image_w, vae_image_h, &dropped_rgb) == 0) {
                    loaded_path = dropped.paths[i];
                }
            }
            if (dropped_rgb) {
                fprintf(stderr, "queued dropped seed image: %s\n", loaded_path ? loaded_path : "<drop>");
                world_mutex_lock(&shared.mutex);
                free(shared.pending_image);
                shared.pending_image = dropped_rgb;
                shared.pending_image_width = vae_image_w;
                shared.pending_image_height = vae_image_h;
                shared.image_pending = 1;
                shared.ready = 0;
                shared.paused = 0;
                memset(shared.control, 0, (size_t)shared.ctrl_dim * sizeof(float));
                world_mutex_unlock(&shared.mutex);
                paused = 0;
                DisableCursor();
                suppress_mouse_frames = 2;
            }
            UnloadDroppedFiles(dropped);
        }
        fill_raylib_control(frame_control, ctrl_dim, cfg.n_buttons, mouse_scale);
        if (suppress_mouse_frames > 0) {
            frame_control[0] = 0.0f;
            frame_control[1] = 0.0f;
            suppress_mouse_frames -= 1;
        }
        float hud_follow = 1.0f - expf(-fminf(frame_seconds, 0.1f) * 22.0f);
        hud_mouse_x += (frame_control[0] - hud_mouse_x) * hud_follow;
        hud_mouse_y += (frame_control[1] - hud_mouse_y) * hud_follow;
        world_mutex_lock(&shared.mutex);
        if (pause_toggled) {
            shared.paused = paused;
            if (paused) {
                memset(shared.control, 0, (size_t)shared.ctrl_dim * sizeof(float));
            }
        }
        if (!shared.paused) {
            merge_frame_control(&shared, frame_control);
        }
        paused = shared.paused;
        image_loading = shared.image_loading || shared.image_pending;
        int failed = shared.failed;
        if (shared.ready && !paused) {
            memcpy(display_rgb, shared.rgb, rgb_bytes);
            last_generation_seconds = shared.generation_seconds;
            shared.ready = 0;
            playback_frame = 0;
            playback_timer = 0.0f;
            playback_interval = playback_interval_seconds(last_generation_seconds, rgb_frames);
            UpdateTexture(texture, display_rgb);
        }
        world_mutex_unlock(&shared.mutex);
        if (failed) break;

        if (!paused && playback_frame < rgb_frames - 1) {
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

        BeginDrawing();
        ClearBackground(BLACK);
        Rectangle src = {0.0f, 0.0f, (float)rgb_w, (float)rgb_h};
        Rectangle dst = fit_rect(GetScreenWidth(), GetScreenHeight(), rgb_w, rgb_h);
        Vector2 origin = {0.0f, 0.0f};
        DrawTexturePro(texture, src, dst, origin, 0.0f, WHITE);
        if (paused) {
            DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), (Color){0, 0, 0, 52});
        }
        draw_live_hud(GetScreenWidth(), GetScreenHeight(),
                      hud_mouse_x, hud_mouse_y,
                      frame_control[2 + cfg.n_buttons], paused, image_loading,
                      rgb_frames, last_generation_seconds);
        EndDrawing();
    }

    world_mutex_lock(&shared.mutex);
    shared.stop = 1;
    int final_failed = shared.failed;
    world_mutex_unlock(&shared.mutex);
    world_thread_join(worker);
    world_mutex_destroy(&shared.mutex);
    free(frame_control);
    free(shared.pending_image);
    free(shared.control);
    free(shared.rgb);
    UnloadTexture(texture);
    EnableCursor();
    CloseWindow();
    free(display_rgb);
    rc = final_failed ? 1 : 0;

cleanup_runtime_only:
    free(seed_image_rgb);
    free(seed_latent);
    world_runtime_destroy(rt);
    return rc;

cleanup_before_window:
    safetensors_close(&st);
    free_loaded_model(&model);
    free_vae_decoder_weights(&vae);
    free_vae_encoder_weights(&vae_encoder);
    free(seed_image_rgb);
    free(seed_latent);
    world_runtime_destroy(rt);
    return rc;
}
