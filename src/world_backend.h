#ifndef WORLD_BACKEND_H
#define WORLD_BACKEND_H

#include "world_cuda.h"

#if defined(WORLD_BACKEND_VULKAN) && WORLD_BACKEND_VULKAN
#include "world_vulkan.h"
typedef WorldVulkanRuntime WorldRuntime;
#define WORLD_BACKEND_NAME "Vulkan"
#define world_runtime_create world_vulkan_runtime_create
#define world_runtime_step_rgb world_vulkan_runtime_step_rgb
#define world_runtime_seed_latent_rgb world_vulkan_runtime_seed_latent_rgb
#define world_runtime_destroy world_vulkan_runtime_destroy
#else
typedef WorldCudaRuntime WorldRuntime;
#define WORLD_BACKEND_NAME "CUDA"
#define world_runtime_create world_cuda_runtime_create
#define world_runtime_step_rgb world_cuda_runtime_step_rgb
#define world_runtime_seed_latent_rgb world_cuda_runtime_seed_latent_rgb
#define world_runtime_destroy world_cuda_runtime_destroy
#endif

#endif
