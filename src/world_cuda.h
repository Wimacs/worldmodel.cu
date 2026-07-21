#ifndef WORLD_CUDA_H
#define WORLD_CUDA_H

#include "world_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WorldCudaRuntime WorldCudaRuntime;

int world_cuda_runtime_create(
        WorldCudaRuntime **out,
        const WorldConfig *cfg,
        const WorldModelWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frame_idx,
        unsigned int seed,
        int noise_mode,
        const WorldVaeDecoderWeights *vae);

int world_cuda_runtime_step_rgb(
        WorldCudaRuntime *rt,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out);

int world_cuda_runtime_seed_latent_rgb(
        WorldCudaRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out);

int world_cuda_runtime_init_vae_encoder(
        WorldCudaRuntime *rt,
        const WorldVaeEncoderWeights *encoder);

int world_cuda_runtime_encode_image_rgb(
        WorldCudaRuntime *rt,
        const float *rgb,
        int width,
        int height,
        float *latent_out,
        float *seconds_out);

int world_cuda_runtime_reset(WorldCudaRuntime *rt, int frame_idx, unsigned int seed);

void world_cuda_runtime_destroy(WorldCudaRuntime *rt);

#ifdef __cplusplus
}
#endif

#endif
