#ifndef WORLD_CUDA_VAE_CUH
#define WORLD_CUDA_VAE_CUH

#include "world_model.h"

typedef struct WorldCudaVae WorldCudaVae;

int wm_cuda_vae_create(
        WorldCudaVae **out,
        const WorldConfig *cfg,
        const WorldVaeDecoderWeights *decoder);
void wm_cuda_vae_destroy(WorldCudaVae *vae);

int wm_cuda_vae_reset(WorldCudaVae *vae);
int wm_cuda_vae_init_encoder(
        WorldCudaVae *vae,
        const WorldVaeEncoderWeights *encoder);
int wm_cuda_vae_encode_rgb(
        WorldCudaVae *vae,
        const float *rgb,
        int width,
        int height,
        float *latent_out);
int wm_cuda_vae_decode_rgb(
        WorldCudaVae *vae,
        const float *device_latent,
        const unsigned char **rgb_out,
        int *frame_count_out,
        int *width_out,
        int *height_out);
#endif
