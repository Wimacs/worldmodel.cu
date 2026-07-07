#define _FILE_OFFSET_BITS 64

#include "safetensors.h"
#include "world_config.h"
#include "world_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATH_BUF 4096

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--model-dir DIR] [--weights FILE] [--seed N]\n"
            "\n"
            "Standalone C+CUDA probe. Loads Waypoint config and safetensors without PyTorch,\n"
            "then starts the generation path with latent -> patchify -> layer0 q_proj.\n",
            argv0);
}

static int join_path(char *out, size_t out_size, const char *a, const char *b) {
    int n = snprintf(out, out_size, "%s/%s", a, b);
    return n < 0 || (size_t)n >= out_size;
}

static int expect_shape(const SafeTensorEntry *e, const int64_t *shape, int ndim) {
    if (!e || e->ndim != ndim) return 1;
    for (int i = 0; i < ndim; ++i) {
        if (e->shape[i] != shape[i]) return 1;
    }
    return 0;
}

static int load_required_f32(const SafeTensors *st, const char *name, const int64_t *shape, int ndim, float **out) {
    const SafeTensorEntry *e = safetensors_find(st, name);
    if (!e) {
        fprintf(stderr, "missing tensor: %s\n", name);
        return 1;
    }
    safetensors_print_entry(e);
    if (strcmp(e->dtype, "F32") != 0) {
        fprintf(stderr, "expected F32 tensor for standalone probe: %s has %s\n", name, e->dtype);
        return 1;
    }
    if (expect_shape(e, shape, ndim)) {
        fprintf(stderr, "shape mismatch for %s\n", name);
        return 1;
    }
    void *data = NULL;
    size_t bytes = 0;
    if (safetensors_read_tensor(st, e, &data, &bytes)) {
        fprintf(stderr, "failed to read tensor: %s\n", name);
        return 1;
    }
    *out = (float *)data;
    return 0;
}

int main(int argc, char **argv) {
    const char *model_dir = "../Waypoint-1.5-1B";
    const char *weights = NULL;
    unsigned int seed = 1234;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    char config_path[PATH_BUF];
    if (join_path(config_path, sizeof(config_path), model_dir, "config.yaml")) {
        fprintf(stderr, "config path too long\n");
        return 1;
    }

    char default_weights[PATH_BUF];
    if (!weights) {
        if (join_path(default_weights, sizeof(default_weights), model_dir, "transformer/diffusion_pytorch_model.safetensors")) {
            fprintf(stderr, "weights path too long\n");
            return 1;
        }
        weights = default_weights;
    }

    WorldConfig cfg;
    if (world_config_load(&cfg, config_path)) return 1;
    world_config_print(&cfg);

    SafeTensors st;
    fprintf(stderr, "loading safetensors index: %s\n", weights);
    if (safetensors_open(&st, weights)) return 1;
    fprintf(stderr, "safetensors tensors: %d\n", st.count);

    int64_t patch_shape[4] = {cfg.d_model, cfg.channels, cfg.patch_h, cfg.patch_w};
    int64_t q_shape[2] = {cfg.d_model, cfg.d_model};
    float *patchify_weight = NULL;
    float *q_proj_weight = NULL;

    if (load_required_f32(&st, "patchify.weight", patch_shape, 4, &patchify_weight)) {
        safetensors_close(&st);
        return 1;
    }
    if (load_required_f32(&st, "transformer.blocks.0.attn.q_proj.weight", q_shape, 2, &q_proj_weight)) {
        free(patchify_weight);
        safetensors_close(&st);
        return 1;
    }

    int rc = world_cuda_generation_probe(&cfg, patchify_weight, q_proj_weight, seed);

    free(patchify_weight);
    free(q_proj_weight);
    safetensors_close(&st);
    return rc;
}
