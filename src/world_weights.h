#ifndef WORLD_WEIGHTS_H
#define WORLD_WEIGHTS_H

#include "safetensors.h"
#include "world_model.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WORLD_PATH_MAX 4096

int world_path_join(char *out, size_t out_size, const char *a, const char *b);

int world_load_tensor_as_f32(
        const SafeTensors *st,
        const char *name,
        const int64_t *shape,
        int ndim,
        float **out);

int world_read_f32_file_exact(const char *path, size_t elems, float **out);

void world_free_layer_weights(WorldLayerWeights *layers, int n_layers);

int world_load_vae_decoder_weights(
        const char *path,
        WorldVaeDecoderWeights *vae);
void world_free_vae_decoder_weights(WorldVaeDecoderWeights *vae);

int world_load_vae_encoder_weights(
        const char *path,
        WorldVaeEncoderWeights *encoder);
void world_free_vae_encoder_weights(WorldVaeEncoderWeights *encoder);

#ifdef __cplusplus
}
#endif

#endif
