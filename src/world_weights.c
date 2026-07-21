#include "world_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int world_path_join(char *out, size_t out_size, const char *a, const char *b) {
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
int world_load_tensor_as_f32(const SafeTensors *st, const char *name, const int64_t *shape, int ndim, float **out) {
    const SafeTensorEntry *e = safetensors_find(st, name);
    if (!e) {
        fprintf(stderr, "missing tensor: %s\n", name);
        return 1;
    }
    safetensors_print_entry(e);
    if (strcmp(e->dtype, "F32") != 0 && strcmp(e->dtype, "F16") != 0 && strcmp(e->dtype, "BF16") != 0) {
        fprintf(stderr, "expected floating tensor for %s, got %s\n", name, e->dtype);
        return 1;
    }
    if (expect_shape(e, shape, ndim)) {
        fprintf(stderr, "shape mismatch for %s\n", name);
        return 1;
    }
    size_t elems = 0;
    if (safetensors_read_tensor_f32(st, e, out, &elems)) {
        fprintf(stderr, "failed to read tensor as f32: %s\n", name);
        return 1;
    }
    return 0;
}
int world_read_f32_file_exact(const char *path, size_t elems, float **out) {
    *out = NULL;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open f32 file: %s\n", path);
        return 1;
    }
    float *data = (float *)malloc(elems * sizeof(float));
    if (!data) {
        fclose(f);
        fprintf(stderr, "failed to allocate f32 file buffer\n");
        return 1;
    }
    size_t got = fread(data, sizeof(float), elems, f);
    int extra = fgetc(f);
    fclose(f);
    if (got != elems || extra != EOF) {
        fprintf(stderr, "expected %zu f32 values in %s, got %zu%s\n",
                elems, path, got, extra == EOF ? "" : "+");
        free(data);
        return 1;
    }
    *out = data;
    return 0;
}

void world_free_layer_weights(WorldLayerWeights *layers, int n_layers) {
    if (!layers) return;
    for (int i = 0; i < n_layers; ++i) {
        free((void *)layers[i].cond_bias);
        free((void *)layers[i].attn_cond_s_weight);
        free((void *)layers[i].attn_cond_b_weight);
        free((void *)layers[i].attn_cond_g_weight);
        free((void *)layers[i].q_proj_weight);
        free((void *)layers[i].k_proj_weight);
        free((void *)layers[i].v_proj_weight);
        free((void *)layers[i].out_proj_weight);
        free((void *)layers[i].v_lamb);
        free((void *)layers[i].mlp_cond_s_weight);
        free((void *)layers[i].mlp_cond_b_weight);
        free((void *)layers[i].mlp_cond_g_weight);
        free((void *)layers[i].ctrl_fc1_x_weight);
        free((void *)layers[i].ctrl_fc1_c_weight);
        free((void *)layers[i].ctrl_fc2_weight);
        free((void *)layers[i].dit_mlp_fc1_weight);
        free((void *)layers[i].dit_mlp_fc2_weight);
    }
    free(layers);
}

typedef struct {
    int index;
    const char *base;
    int out_c;
    int in_c;
    int kernel;
    int has_bias;
} VaeConvSpec;

static const VaeConvSpec k_vae_decoder_specs[WORLD_VAE_DECODER_CONV_COUNT] = {
    {WORLD_VAE_DEC_CONV_IN, "decoder.1", 256, 32, 3, 1},
    {WORLD_VAE_DEC_MB3_0, "decoder.3.conv.0", 256, 512, 3, 1},
    {WORLD_VAE_DEC_MB3_2, "decoder.3.conv.2", 256, 256, 3, 1},
    {WORLD_VAE_DEC_MB3_4, "decoder.3.conv.4", 256, 256, 3, 1},
    {WORLD_VAE_DEC_MB4_0, "decoder.4.conv.0", 256, 512, 3, 1},
    {WORLD_VAE_DEC_MB4_2, "decoder.4.conv.2", 256, 256, 3, 1},
    {WORLD_VAE_DEC_MB4_4, "decoder.4.conv.4", 256, 256, 3, 1},
    {WORLD_VAE_DEC_MB5_0, "decoder.5.conv.0", 256, 512, 3, 1},
    {WORLD_VAE_DEC_MB5_2, "decoder.5.conv.2", 256, 256, 3, 1},
    {WORLD_VAE_DEC_MB5_4, "decoder.5.conv.4", 256, 256, 3, 1},
    {WORLD_VAE_DEC_TGROW7, "decoder.7.conv", 256, 256, 1, 0},
    {WORLD_VAE_DEC_CONV8, "decoder.8", 128, 256, 3, 0},
    {WORLD_VAE_DEC_MB9_0, "decoder.9.conv.0", 128, 256, 3, 1},
    {WORLD_VAE_DEC_MB9_2, "decoder.9.conv.2", 128, 128, 3, 1},
    {WORLD_VAE_DEC_MB9_4, "decoder.9.conv.4", 128, 128, 3, 1},
    {WORLD_VAE_DEC_MB10_0, "decoder.10.conv.0", 128, 256, 3, 1},
    {WORLD_VAE_DEC_MB10_2, "decoder.10.conv.2", 128, 128, 3, 1},
    {WORLD_VAE_DEC_MB10_4, "decoder.10.conv.4", 128, 128, 3, 1},
    {WORLD_VAE_DEC_MB11_0, "decoder.11.conv.0", 128, 256, 3, 1},
    {WORLD_VAE_DEC_MB11_2, "decoder.11.conv.2", 128, 128, 3, 1},
    {WORLD_VAE_DEC_MB11_4, "decoder.11.conv.4", 128, 128, 3, 1},
    {WORLD_VAE_DEC_TGROW13, "decoder.13.conv", 256, 128, 1, 0},
    {WORLD_VAE_DEC_CONV14, "decoder.14", 64, 128, 3, 0},
    {WORLD_VAE_DEC_MB15_0, "decoder.15.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_DEC_MB15_2, "decoder.15.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_DEC_MB15_4, "decoder.15.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_DEC_MB16_0, "decoder.16.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_DEC_MB16_2, "decoder.16.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_DEC_MB16_4, "decoder.16.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_DEC_MB17_0, "decoder.17.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_DEC_MB17_2, "decoder.17.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_DEC_MB17_4, "decoder.17.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_DEC_TGROW19, "decoder.19.conv", 128, 64, 1, 0},
    {WORLD_VAE_DEC_CONV20, "decoder.20", 64, 64, 3, 0},
    {WORLD_VAE_DEC_CONV_OUT, "decoder.22", 12, 64, 3, 1},
};

static const VaeConvSpec k_vae_encoder_specs[WORLD_VAE_ENCODER_CONV_COUNT] = {
    {WORLD_VAE_ENC_CONV_IN, "encoder.0", 64, 12, 3, 1},
    {WORLD_VAE_ENC_TPOOL2, "encoder.2.conv", 64, 128, 1, 0},
    {WORLD_VAE_ENC_CONV3, "encoder.3", 64, 64, 3, 0},
    {WORLD_VAE_ENC_MB4_0, "encoder.4.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB4_2, "encoder.4.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB4_4, "encoder.4.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB5_0, "encoder.5.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB5_2, "encoder.5.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB5_4, "encoder.5.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB6_0, "encoder.6.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB6_2, "encoder.6.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB6_4, "encoder.6.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_TPOOL7, "encoder.7.conv", 64, 128, 1, 0},
    {WORLD_VAE_ENC_CONV8, "encoder.8", 64, 64, 3, 0},
    {WORLD_VAE_ENC_MB9_0, "encoder.9.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB9_2, "encoder.9.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB9_4, "encoder.9.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB10_0, "encoder.10.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB10_2, "encoder.10.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB10_4, "encoder.10.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB11_0, "encoder.11.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB11_2, "encoder.11.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB11_4, "encoder.11.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_TPOOL12, "encoder.12.conv", 64, 64, 1, 0},
    {WORLD_VAE_ENC_CONV13, "encoder.13", 64, 64, 3, 0},
    {WORLD_VAE_ENC_MB14_0, "encoder.14.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB14_2, "encoder.14.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB14_4, "encoder.14.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB15_0, "encoder.15.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB15_2, "encoder.15.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB15_4, "encoder.15.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB16_0, "encoder.16.conv.0", 64, 128, 3, 1},
    {WORLD_VAE_ENC_MB16_2, "encoder.16.conv.2", 64, 64, 3, 1},
    {WORLD_VAE_ENC_MB16_4, "encoder.16.conv.4", 64, 64, 3, 1},
    {WORLD_VAE_ENC_CONV_OUT, "encoder.17", 32, 64, 3, 1},
};

int world_load_vae_decoder_weights(const char *path, WorldVaeDecoderWeights *vae) {
    memset(vae, 0, sizeof(*vae));
    SafeTensors st;
    fprintf(stderr, "loading VAE safetensors index: %s\n", path);
    if (safetensors_open(&st, path)) return 1;
    fprintf(stderr, "VAE safetensors tensors: %d\n", st.count);

    int rc = 1;
    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
        const VaeConvSpec *spec = &k_vae_decoder_specs[i];
        WorldVaeConvWeight *cw = &vae->convs[spec->index];
        char name[256];
        int64_t w_shape[4] = {spec->out_c, spec->in_c, spec->kernel, spec->kernel};
        int n = snprintf(name, sizeof(name), "%s.weight", spec->base);
        if (n < 0 || (size_t)n >= sizeof(name)) goto cleanup;
        if (world_load_tensor_as_f32(&st, name, w_shape, 4, (float **)&cw->weight)) goto cleanup;
        if (spec->has_bias) {
            int64_t b_shape[1] = {spec->out_c};
            n = snprintf(name, sizeof(name), "%s.bias", spec->base);
            if (n < 0 || (size_t)n >= sizeof(name)) goto cleanup;
            if (world_load_tensor_as_f32(&st, name, b_shape, 1, (float **)&cw->bias)) goto cleanup;
        }
        cw->out_c = spec->out_c;
        cw->in_c = spec->in_c;
        cw->kernel = spec->kernel;
        cw->has_bias = spec->has_bias;
    }
    rc = 0;

cleanup:
    safetensors_close(&st);
    return rc;
}

void world_free_vae_decoder_weights(WorldVaeDecoderWeights *vae) {
    if (!vae) return;
    for (int i = 0; i < WORLD_VAE_DECODER_CONV_COUNT; ++i) {
        free((void *)vae->convs[i].weight);
        free((void *)vae->convs[i].bias);
    }
    memset(vae, 0, sizeof(*vae));
}

int world_load_vae_encoder_weights(const char *path, WorldVaeEncoderWeights *encoder) {
    memset(encoder, 0, sizeof(*encoder));
    SafeTensors st;
    fprintf(stderr, "loading VAE encoder safetensors index: %s\n", path);
    if (safetensors_open(&st, path)) return 1;

    int rc = 1;
    for (int i = 0; i < WORLD_VAE_ENCODER_CONV_COUNT; ++i) {
        const VaeConvSpec *spec = &k_vae_encoder_specs[i];
        WorldVaeConvWeight *cw = &encoder->convs[spec->index];
        char name[256];
        int64_t w_shape[4] = {spec->out_c, spec->in_c, spec->kernel, spec->kernel};
        int n = snprintf(name, sizeof(name), "%s.weight", spec->base);
        if (n < 0 || (size_t)n >= sizeof(name)) goto cleanup;
        if (world_load_tensor_as_f32(&st, name, w_shape, 4, (float **)&cw->weight)) goto cleanup;
        if (spec->has_bias) {
            int64_t b_shape[1] = {spec->out_c};
            n = snprintf(name, sizeof(name), "%s.bias", spec->base);
            if (n < 0 || (size_t)n >= sizeof(name)) goto cleanup;
            if (world_load_tensor_as_f32(&st, name, b_shape, 1, (float **)&cw->bias)) goto cleanup;
        }
        cw->out_c = spec->out_c;
        cw->in_c = spec->in_c;
        cw->kernel = spec->kernel;
        cw->has_bias = spec->has_bias;
    }
    rc = 0;

cleanup:
    safetensors_close(&st);
    return rc;
}

void world_free_vae_encoder_weights(WorldVaeEncoderWeights *encoder) {
    if (!encoder) return;
    for (int i = 0; i < WORLD_VAE_ENCODER_CONV_COUNT; ++i) {
        free((void *)encoder->convs[i].weight);
        free((void *)encoder->convs[i].bias);
    }
    memset(encoder, 0, sizeof(*encoder));
}
