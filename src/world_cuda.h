#ifndef WORLD_CUDA_H
#define WORLD_CUDA_H

#include "world_config.h"

#ifdef __cplusplus
extern "C" {
#endif

int world_cuda_generation_probe(
        const WorldConfig *cfg,
        const float *patchify_weight,
        const float *q_proj_weight,
        unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif
