#include "world_vulkan.h"

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef WORLD_VULKAN_SHADER_DIR
#define WORLD_VULKAN_SHADER_DIR "shaders/vulkan"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t frames;
    uint32_t frame_ordinal;
    float control_x;
    float control_y;
} WorldVulkanFillPush;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t inner;
    uint32_t has_bias;
} WorldVulkanLinearPush;

typedef struct {
    uint32_t n;
} WorldVulkanSiluPush;

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float eps;
} WorldVulkanRmsNormPush;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t N;
    uint32_t D;
    float eps;
} WorldVulkanAdaRmsNormPush;

typedef struct {
    uint32_t B;
    uint32_t H;
    uint32_t T;
    uint32_t D;
    uint32_t width;
    uint32_t height;
} WorldVulkanOrthoRopePush;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t D;
    uint32_t width;
    uint32_t height;
    float eps;
} WorldVulkanQkvRmsRopePush;

typedef struct {
    uint32_t B;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t Tq;
    uint32_t Tk;
    uint32_t D;
    float scale;
} WorldVulkanMaskedAttentionPush;

typedef struct {
    uint32_t capacity;
    uint32_t T;
    uint32_t base;
    uint32_t write_step;
} WorldVulkanKvCacheMaskPush;

typedef struct {
    uint32_t B;
    uint32_t H;
    uint32_t T;
    uint32_t D;
    uint32_t L;
    uint32_t base;
    uint32_t write_step;
    uint32_t frozen;
} WorldVulkanKvCacheUpsertPush;

typedef struct {
    uint32_t capacity;
    uint32_t T;
    uint32_t base;
    uint32_t write_step;
} WorldVulkanCacheFrameIndicesPush;

typedef struct {
    uint32_t B;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t D;
    uint32_t ph;
    uint32_t pw;
    uint32_t Hp;
    uint32_t Wp;
} WorldVulkanPatchifyPush;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t D;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t ph;
    uint32_t pw;
    uint32_t Hp;
    uint32_t Wp;
    uint32_t out_dim;
} WorldVulkanUnpatchifyPush;

typedef struct {
    uint32_t T;
    uint32_t D;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t ph;
    uint32_t pw;
    uint32_t Wp;
    uint32_t out_dim;
} WorldVulkanUnpatchifyOrigPush;

typedef struct {
    uint32_t out_width;
    uint32_t out_height;
    uint32_t frames;
    uint32_t latent_c;
    uint32_t latent_h;
    uint32_t latent_w;
    uint32_t frame_ordinal;
    float control_x;
    float control_y;
} WorldVulkanLatentRgbaPush;

typedef struct {
    uint32_t B;
    uint32_t Hq;
    uint32_t Hkv;
    uint32_t Tq;
    uint32_t Nkv;
    uint32_t Tk;
    uint32_t D;
    float scale;
} WorldVulkanIndexedAttentionPush;

struct WorldVulkanRuntime {
    WorldConfig cfg;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkShaderModule fill_shader;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline fill_pipeline;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkBuffer output_buffer;
    VkDeviceMemory output_memory;
    void *output_mapped;
    unsigned char *rgb_host;
    size_t pixel_count;
    size_t rgb_bytes;
    int width;
    int height;
    int frames;
    int frame_ordinal;
    unsigned int seed;
    int model_slice_enabled;
    int C;
    int H;
    int W;
    int D;
    int ph;
    int pw;
    int T;
    int out_dim;
    size_t latent_elems;
    size_t token_elems;
    int use_external_latent_once;
    VkShaderModule patchify_shader;
    VkDescriptorSetLayout patchify_set_layout;
    VkPipelineLayout patchify_pipeline_layout;
    VkPipeline patchify_pipeline;
    VkDescriptorPool patchify_descriptor_pool;
    VkDescriptorSet patchify_descriptor_set;
    VkShaderModule unpatch_orig_shader;
    VkDescriptorSetLayout unpatch_orig_set_layout;
    VkPipelineLayout unpatch_orig_pipeline_layout;
    VkPipeline unpatch_orig_pipeline;
    VkDescriptorPool unpatch_orig_descriptor_pool;
    VkDescriptorSet unpatch_orig_descriptor_set;
    VkShaderModule latent_rgba_shader;
    VkDescriptorSetLayout latent_rgba_set_layout;
    VkPipelineLayout latent_rgba_pipeline_layout;
    VkPipeline latent_rgba_pipeline;
    VkDescriptorPool latent_rgba_descriptor_pool;
    VkDescriptorSet latent_rgba_descriptor_set;
    VkBuffer latent_buffer;
    VkDeviceMemory latent_memory;
    void *latent_mapped;
    VkBuffer patch_weight_buffer;
    VkDeviceMemory patch_weight_memory;
    void *patch_weight_mapped;
    VkBuffer tokens_buffer;
    VkDeviceMemory tokens_memory;
    void *tokens_mapped;
    VkBuffer unpatch_weight_buffer;
    VkDeviceMemory unpatch_weight_memory;
    void *unpatch_weight_mapped;
    VkBuffer unpatch_bias_buffer;
    VkDeviceMemory unpatch_bias_memory;
    void *unpatch_bias_mapped;
    VkBuffer latent_out_buffer;
    VkDeviceMemory latent_out_memory;
    void *latent_out_mapped;
};

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static const char *vk_result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "VK_RESULT_UNKNOWN";
    }
}

static int vk_check(VkResult r, const char *expr, const char *file, int line) {
    if (r == VK_SUCCESS) return 0;
    fprintf(stderr, "Vulkan error at %s:%d: %s -> %s (%d)\n", file, line, expr, vk_result_name(r), (int)r);
    return 1;
}

#define VK_CALL(expr) do { if (vk_check((expr), #expr, __FILE__, __LINE__)) goto fail; } while (0)
#define VK_CALL_RET(expr) do { if (vk_check((expr), #expr, __FILE__, __LINE__)) return 1; } while (0)

static int read_file_bytes(const char *path, void **data_out, size_t *bytes_out) {
    *data_out = NULL;
    *bytes_out = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open Vulkan shader: %s\n", path);
        return 1;
    }
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return 1;
    }
    long n = ftell(f);
    if (n <= 0 || (n % 4) != 0) {
        fclose(f);
        fprintf(stderr, "invalid SPIR-V size for %s: %ld\n", path, n);
        return 1;
    }
    rewind(f);
    void *data = malloc((size_t)n);
    if (!data) {
        fclose(f);
        return 1;
    }
    size_t got = fread(data, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(data);
        fprintf(stderr, "short read for Vulkan shader: %s\n", path);
        return 1;
    }
    *data_out = data;
    *bytes_out = (size_t)n;
    return 0;
}

static int find_queue_family(VkPhysicalDevice dev, uint32_t *family_out) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, NULL);
    if (count == 0) return 1;
    VkQueueFamilyProperties *props = (VkQueueFamilyProperties *)calloc(count, sizeof(*props));
    if (!props) return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props);
    int found = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            *family_out = i;
            found = 1;
            break;
        }
    }
    free(props);
    return found ? 0 : 1;
}

static int pick_physical_device(WorldVulkanRuntime *rt) {
    uint32_t count = 0;
    VK_CALL_RET(vkEnumeratePhysicalDevices(rt->instance, &count, NULL));
    if (count == 0) {
        fprintf(stderr, "no Vulkan physical devices found\n");
        return 1;
    }
    VkPhysicalDevice *devices = (VkPhysicalDevice *)calloc(count, sizeof(*devices));
    if (!devices) return 1;
    if (vkEnumeratePhysicalDevices(rt->instance, &count, devices) != VK_SUCCESS) {
        free(devices);
        return 1;
    }
    int best = -1;
    uint32_t best_family = 0;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props;
        uint32_t family = 0;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (find_queue_family(devices[i], &family)) continue;
        if (best < 0 || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            best = (int)i;
            best_family = family;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
        }
    }
    if (best < 0) {
        free(devices);
        fprintf(stderr, "no Vulkan compute-capable device found\n");
        return 1;
    }
    rt->physical_device = devices[best];
    rt->queue_family = best_family;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(rt->physical_device, &props);
    fprintf(stderr, "Vulkan device: %s queue_family=%u\n", props.deviceName, rt->queue_family);
    free(devices);
    return 0;
}

static int find_memory_type(WorldVulkanRuntime *rt, uint32_t type_bits, VkMemoryPropertyFlags flags, uint32_t *type_out) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(rt->physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            *type_out = i;
            return 0;
        }
    }
    return 1;
}

static int create_host_buffer(
        WorldVulkanRuntime *rt,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkBuffer *buffer,
        VkDeviceMemory *memory,
        void **mapped) {
    *buffer = VK_NULL_HANDLE;
    *memory = VK_NULL_HANDLE;
    *mapped = NULL;

    VkBufferCreateInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CALL_RET(vkCreateBuffer(rt->device, &buffer_info, NULL, buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(rt->device, *buffer, &req);
    uint32_t type_index = 0;
    if (find_memory_type(rt, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &type_index)) {
        fprintf(stderr, "failed to find host-visible coherent Vulkan memory\n");
        return 1;
    }

    VkMemoryAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = type_index;
    VK_CALL_RET(vkAllocateMemory(rt->device, &alloc_info, NULL, memory));
    VK_CALL_RET(vkBindBufferMemory(rt->device, *buffer, *memory, 0));
    VK_CALL_RET(vkMapMemory(rt->device, *memory, 0, size, 0, mapped));
    return 0;
}

static void destroy_host_buffer(WorldVulkanRuntime *rt, VkBuffer buffer, VkDeviceMemory memory, void *mapped) {
    if (mapped) vkUnmapMemory(rt->device, memory);
    if (buffer) vkDestroyBuffer(rt->device, buffer, NULL);
    if (memory) vkFreeMemory(rt->device, memory, NULL);
}

static int create_output_buffer(WorldVulkanRuntime *rt) {
    return create_host_buffer(rt,
            rt->pixel_count * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            &rt->output_buffer,
            &rt->output_memory,
            &rt->output_mapped);
}

static int create_shader_module_from_name(WorldVulkanRuntime *rt, const char *shader_name, VkShaderModule *module_out) {
    *module_out = VK_NULL_HANDLE;
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s.spv", WORLD_VULKAN_SHADER_DIR, shader_name);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "Vulkan shader path too long\n");
        return 1;
    }

    void *spv = NULL;
    size_t spv_bytes = 0;
    if (read_file_bytes(path, &spv, &spv_bytes)) return 1;

    VkShaderModuleCreateInfo shader_info;
    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spv_bytes;
    shader_info.pCode = (const uint32_t *)spv;
    VkResult shader_result = vkCreateShaderModule(rt->device, &shader_info, NULL, module_out);
    free(spv);
    if (vk_check(shader_result, "vkCreateShaderModule", __FILE__, __LINE__)) return 1;
    return 0;
}

static int create_fill_pipeline(WorldVulkanRuntime *rt) {
    if (create_shader_module_from_name(rt, "fill_rgba.comp", &rt->fill_shader)) return 1;

    VkDescriptorSetLayoutBinding binding;
    memset(&binding, 0, sizeof(binding));
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo set_info;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 1;
    set_info.pBindings = &binding;
    VK_CALL_RET(vkCreateDescriptorSetLayout(rt->device, &set_info, NULL, &rt->descriptor_set_layout));

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(WorldVulkanFillPush);

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &rt->descriptor_set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VK_CALL_RET(vkCreatePipelineLayout(rt->device, &layout_info, NULL, &rt->pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = rt->fill_shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = rt->pipeline_layout;
    VK_CALL_RET(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &rt->fill_pipeline));
    return 0;
}

static int create_descriptors(WorldVulkanRuntime *rt) {
    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CALL_RET(vkCreateDescriptorPool(rt->device, &pool_info, NULL, &rt->descriptor_pool));

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = rt->descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &rt->descriptor_set_layout;
    VK_CALL_RET(vkAllocateDescriptorSets(rt->device, &alloc_info, &rt->descriptor_set));

    VkDescriptorBufferInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.buffer = rt->output_buffer;
    buffer_info.offset = 0;
    buffer_info.range = rt->pixel_count * sizeof(uint32_t);

    VkWriteDescriptorSet write;
    memset(&write, 0, sizeof(write));
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = rt->descriptor_set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(rt->device, 1, &write, 0, NULL);
    return 0;
}

static int create_commands(WorldVulkanRuntime *rt) {
    VkCommandPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = rt->queue_family;
    VK_CALL_RET(vkCreateCommandPool(rt->device, &pool_info, NULL, &rt->command_pool));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = rt->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VK_CALL_RET(vkAllocateCommandBuffers(rt->device, &alloc_info, &rt->command_buffer));

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CALL_RET(vkCreateFence(rt->device, &fence_info, NULL, &rt->fence));
    return 0;
}

static int create_storage_pipeline(
        WorldVulkanRuntime *rt,
        const char *shader_name,
        uint32_t binding_count,
        uint32_t push_size,
        VkShaderModule *shader,
        VkDescriptorSetLayout *set_layout,
        VkPipelineLayout *pipeline_layout,
        VkPipeline *pipeline) {
    *shader = VK_NULL_HANDLE;
    *set_layout = VK_NULL_HANDLE;
    *pipeline_layout = VK_NULL_HANDLE;
    *pipeline = VK_NULL_HANDLE;
    if (create_shader_module_from_name(rt, shader_name, shader)) return 1;

    VkDescriptorSetLayoutBinding *bindings =
        (VkDescriptorSetLayoutBinding *)calloc(binding_count, sizeof(*bindings));
    if (!bindings) return 1;
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set_info;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = binding_count;
    set_info.pBindings = bindings;
    VkResult set_result = vkCreateDescriptorSetLayout(rt->device, &set_info, NULL, set_layout);
    free(bindings);
    if (vk_check(set_result, "vkCreateDescriptorSetLayout", __FILE__, __LINE__)) return 1;

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_size;

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = set_layout;
    layout_info.pushConstantRangeCount = push_size ? 1 : 0;
    layout_info.pPushConstantRanges = push_size ? &push_range : NULL;
    VK_CALL_RET(vkCreatePipelineLayout(rt->device, &layout_info, NULL, pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = *shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = *pipeline_layout;
    VK_CALL_RET(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline));
    return 0;
}

static int create_storage_descriptor_set(
        WorldVulkanRuntime *rt,
        VkDescriptorSetLayout set_layout,
        uint32_t binding_count,
        const VkBuffer *buffers,
        const VkDeviceSize *sizes,
        VkDescriptorPool *descriptor_pool,
        VkDescriptorSet *descriptor_set) {
    *descriptor_pool = VK_NULL_HANDLE;
    *descriptor_set = VK_NULL_HANDLE;
    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = binding_count;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CALL_RET(vkCreateDescriptorPool(rt->device, &pool_info, NULL, descriptor_pool));

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = *descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;
    VK_CALL_RET(vkAllocateDescriptorSets(rt->device, &alloc_info, descriptor_set));

    VkDescriptorBufferInfo *buffer_infos =
        (VkDescriptorBufferInfo *)calloc(binding_count, sizeof(*buffer_infos));
    VkWriteDescriptorSet *writes =
        (VkWriteDescriptorSet *)calloc(binding_count, sizeof(*writes));
    if (!buffer_infos || !writes) {
        free(buffer_infos);
        free(writes);
        return 1;
    }
    for (uint32_t i = 0; i < binding_count; ++i) {
        buffer_infos[i].buffer = buffers[i];
        buffer_infos[i].range = sizes[i];
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = *descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(rt->device, binding_count, writes, 0, NULL);
    free(buffer_infos);
    free(writes);
    return 0;
}

static int submit_compute(
        WorldVulkanRuntime *rt,
        VkPipeline pipeline,
        VkPipelineLayout pipeline_layout,
        VkDescriptorSet descriptor_set,
        const void *push,
        uint32_t push_size,
        uint32_t groups_x,
        uint32_t groups_y,
        uint32_t groups_z) {
    VK_CALL_RET(vkResetFences(rt->device, 1, &rt->fence));
    VK_CALL_RET(vkResetCommandBuffer(rt->command_buffer, 0));
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL_RET(vkBeginCommandBuffer(rt->command_buffer, &begin_info));
    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    if (push && push_size) {
        vkCmdPushConstants(rt->command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, push_size, push);
    }
    vkCmdDispatch(rt->command_buffer, groups_x, groups_y, groups_z);
    VK_CALL_RET(vkEndCommandBuffer(rt->command_buffer));

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    VK_CALL_RET(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
    VK_CALL_RET(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));
    return 0;
}

static void fill_runtime_latent(WorldVulkanRuntime *rt, const float *control_input) {
    float *latent = (float *)rt->latent_mapped;
    float cx = control_input ? control_input[0] : 0.0f;
    float cy = control_input ? control_input[1] : 0.0f;
    float frame = (float)rt->frame_ordinal;
    for (int c = 0; c < rt->C; ++c) {
        for (int y = 0; y < rt->H; ++y) {
            for (int x = 0; x < rt->W; ++x) {
                float fx = ((float)x + 0.5f) / (float)rt->W;
                float fy = ((float)y + 0.5f) / (float)rt->H;
                float fc = (float)(c + 1);
                float v = sinf(17.0f * fx + 11.0f * fy + 0.07f * frame + 0.13f * fc);
                v += 0.35f * cosf(9.0f * fx * fc + 0.19f * frame + 0.2f * cx);
                v += 0.25f * sinf(8.0f * fy + 0.15f * cy + 0.03f * (float)rt->seed);
                latent[((c * rt->H + y) * rt->W + x)] = v;
            }
        }
    }
}

static void cmd_shader_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
}

static int create_runtime_model_slice(
        WorldVulkanRuntime *rt,
        const WorldModelProbeWeights *weights) {
    if (!weights || !weights->patchify_weight || !weights->unpatchify_weight || !weights->unpatchify_bias) {
        return 0;
    }

    rt->C = rt->cfg.channels;
    rt->H = rt->cfg.height * rt->cfg.patch_h;
    rt->W = rt->cfg.width * rt->cfg.patch_w;
    rt->D = rt->cfg.d_model;
    rt->ph = rt->cfg.patch_h;
    rt->pw = rt->cfg.patch_w;
    rt->T = rt->cfg.height * rt->cfg.width;
    rt->out_dim = rt->C * rt->ph * rt->pw;
    rt->latent_elems = (size_t)rt->C * rt->H * rt->W;
    rt->token_elems = (size_t)rt->T * rt->D;

    size_t latent_bytes = rt->latent_elems * sizeof(float);
    size_t token_bytes = rt->token_elems * sizeof(float);
    size_t patch_weight_bytes = (size_t)rt->D * rt->C * rt->ph * rt->pw * sizeof(float);
    size_t unpatch_bias_bytes = (size_t)rt->C * sizeof(float);

    if (create_host_buffer(rt, latent_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->latent_buffer, &rt->latent_memory, &rt->latent_mapped)) return 1;
    if (create_host_buffer(rt, patch_weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->patch_weight_buffer, &rt->patch_weight_memory, &rt->patch_weight_mapped)) return 1;
    if (create_host_buffer(rt, token_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->tokens_buffer, &rt->tokens_memory, &rt->tokens_mapped)) return 1;
    if (create_host_buffer(rt, patch_weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->unpatch_weight_buffer, &rt->unpatch_weight_memory, &rt->unpatch_weight_mapped)) return 1;
    if (create_host_buffer(rt, unpatch_bias_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->unpatch_bias_buffer, &rt->unpatch_bias_memory, &rt->unpatch_bias_mapped)) return 1;
    if (create_host_buffer(rt, latent_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &rt->latent_out_buffer, &rt->latent_out_memory, &rt->latent_out_mapped)) return 1;

    memcpy(rt->patch_weight_mapped, weights->patchify_weight, patch_weight_bytes);
    memcpy(rt->unpatch_weight_mapped, weights->unpatchify_weight, patch_weight_bytes);
    memcpy(rt->unpatch_bias_mapped, weights->unpatchify_bias, unpatch_bias_bytes);
    memset(rt->latent_mapped, 0, latent_bytes);
    memset(rt->tokens_mapped, 0, token_bytes);
    memset(rt->latent_out_mapped, 0, latent_bytes);

    if (create_storage_pipeline(rt, "patchify_f32.comp", 3, sizeof(WorldVulkanPatchifyPush),
                &rt->patchify_shader, &rt->patchify_set_layout,
                &rt->patchify_pipeline_layout, &rt->patchify_pipeline)) return 1;
    if (create_storage_pipeline(rt, "unpatchify_orig_f32.comp", 4, sizeof(WorldVulkanUnpatchifyOrigPush),
                &rt->unpatch_orig_shader, &rt->unpatch_orig_set_layout,
                &rt->unpatch_orig_pipeline_layout, &rt->unpatch_orig_pipeline)) return 1;
    if (create_storage_pipeline(rt, "latent_to_rgba.comp", 2, sizeof(WorldVulkanLatentRgbaPush),
                &rt->latent_rgba_shader, &rt->latent_rgba_set_layout,
                &rt->latent_rgba_pipeline_layout, &rt->latent_rgba_pipeline)) return 1;

    {
        VkBuffer buffers[3] = {rt->latent_buffer, rt->patch_weight_buffer, rt->tokens_buffer};
        VkDeviceSize sizes[3] = {latent_bytes, patch_weight_bytes, token_bytes};
        if (create_storage_descriptor_set(rt, rt->patchify_set_layout, 3, buffers, sizes,
                    &rt->patchify_descriptor_pool, &rt->patchify_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[4] = {
            rt->tokens_buffer, rt->unpatch_weight_buffer, rt->unpatch_bias_buffer, rt->latent_out_buffer
        };
        VkDeviceSize sizes[4] = {token_bytes, patch_weight_bytes, unpatch_bias_bytes, latent_bytes};
        if (create_storage_descriptor_set(rt, rt->unpatch_orig_set_layout, 4, buffers, sizes,
                    &rt->unpatch_orig_descriptor_pool, &rt->unpatch_orig_descriptor_set)) return 1;
    }
    {
        VkBuffer buffers[2] = {rt->latent_out_buffer, rt->output_buffer};
        VkDeviceSize sizes[2] = {latent_bytes, rt->pixel_count * sizeof(uint32_t)};
        if (create_storage_descriptor_set(rt, rt->latent_rgba_set_layout, 2, buffers, sizes,
                    &rt->latent_rgba_descriptor_pool, &rt->latent_rgba_descriptor_set)) return 1;
    }

    rt->model_slice_enabled = 1;
    fprintf(stderr,
            "Vulkan resident latent slice enabled: C=%d H=%d W=%d T=%d D=%d bytes(latent)=%.2f MiB\n",
            rt->C, rt->H, rt->W, rt->T, rt->D,
            (double)latent_bytes / (1024.0 * 1024.0));
    return 0;
}

static int record_runtime_model_slice(
        WorldVulkanRuntime *rt,
        const float *control_input) {
    WorldVulkanPatchifyPush patch_push;
    memset(&patch_push, 0, sizeof(patch_push));
    patch_push.B = 1;
    patch_push.C = (uint32_t)rt->C;
    patch_push.H = (uint32_t)rt->H;
    patch_push.W = (uint32_t)rt->W;
    patch_push.D = (uint32_t)rt->D;
    patch_push.ph = (uint32_t)rt->ph;
    patch_push.pw = (uint32_t)rt->pw;
    patch_push.Hp = (uint32_t)rt->cfg.height;
    patch_push.Wp = (uint32_t)rt->cfg.width;

    WorldVulkanUnpatchifyOrigPush unpatch_push;
    memset(&unpatch_push, 0, sizeof(unpatch_push));
    unpatch_push.T = (uint32_t)rt->T;
    unpatch_push.D = (uint32_t)rt->D;
    unpatch_push.C = (uint32_t)rt->C;
    unpatch_push.H = (uint32_t)rt->H;
    unpatch_push.W = (uint32_t)rt->W;
    unpatch_push.ph = (uint32_t)rt->ph;
    unpatch_push.pw = (uint32_t)rt->pw;
    unpatch_push.Wp = (uint32_t)rt->cfg.width;
    unpatch_push.out_dim = (uint32_t)rt->out_dim;

    WorldVulkanLatentRgbaPush rgba_push;
    memset(&rgba_push, 0, sizeof(rgba_push));
    rgba_push.out_width = (uint32_t)rt->width;
    rgba_push.out_height = (uint32_t)rt->height;
    rgba_push.frames = (uint32_t)rt->frames;
    rgba_push.latent_c = (uint32_t)rt->C;
    rgba_push.latent_h = (uint32_t)rt->H;
    rgba_push.latent_w = (uint32_t)rt->W;
    rgba_push.frame_ordinal = (uint32_t)rt->frame_ordinal;
    if (control_input) {
        rgba_push.control_x = control_input[0];
        rgba_push.control_y = control_input[1];
    }

    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->patchify_pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            rt->patchify_pipeline_layout, 0, 1, &rt->patchify_descriptor_set, 0, NULL);
    vkCmdPushConstants(rt->command_buffer, rt->patchify_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(patch_push), &patch_push);
    vkCmdDispatch(rt->command_buffer, (uint32_t)(rt->T * rt->D), 1, 1);
    cmd_shader_barrier(rt->command_buffer);

    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->unpatch_orig_pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            rt->unpatch_orig_pipeline_layout, 0, 1, &rt->unpatch_orig_descriptor_set, 0, NULL);
    vkCmdPushConstants(rt->command_buffer, rt->unpatch_orig_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(unpatch_push), &unpatch_push);
    vkCmdDispatch(rt->command_buffer, (uint32_t)(rt->T * rt->out_dim), 1, 1);
    cmd_shader_barrier(rt->command_buffer);

    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->latent_rgba_pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            rt->latent_rgba_pipeline_layout, 0, 1, &rt->latent_rgba_descriptor_set, 0, NULL);
    vkCmdPushConstants(rt->command_buffer, rt->latent_rgba_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(rgba_push), &rgba_push);
    vkCmdDispatch(rt->command_buffer,
            ((uint32_t)rt->width + 15u) / 16u,
            ((uint32_t)rt->height + 15u) / 16u,
            (uint32_t)rt->frames);
    return 0;
}

int world_vulkan_runtime_create(
        WorldVulkanRuntime **out,
        const WorldConfig *cfg,
        const WorldModelProbeWeights *weights,
        int layers_to_run,
        int steps_to_run,
        int frame_idx,
        unsigned int seed,
        int noise_mode,
        const WorldVaeDecoderWeights *vae) {
    (void)layers_to_run;
    (void)steps_to_run;
    (void)noise_mode;
    (void)vae;
    if (!out || !cfg) return 1;
    *out = NULL;

    WorldVulkanRuntime *rt = (WorldVulkanRuntime *)calloc(1, sizeof(*rt));
    if (!rt) return 1;
    rt->cfg = *cfg;
    rt->width = cfg->width * cfg->patch_w * 16;
    rt->height = cfg->height * cfg->patch_h * 16;
    rt->frames = 4;
    rt->frame_ordinal = frame_idx;
    rt->seed = seed;
    rt->pixel_count = (size_t)rt->width * (size_t)rt->height * (size_t)rt->frames;
    rt->rgb_bytes = rt->pixel_count * 3;
    rt->rgb_host = (unsigned char *)malloc(rt->rgb_bytes);
    if (!rt->rgb_host) goto fail;

    fprintf(stderr,
            "creating resident Vulkan runtime scaffold: RGB %dx%d frames=%d shader_dir=%s\n",
            rt->width, rt->height, rt->frames, WORLD_VULKAN_SHADER_DIR);
    fprintf(stderr,
            "warning: Vulkan backend is partial; transformer/VAE kernels are being ported incrementally\n");

    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "worldmodel.cu";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "worldmodel.cu";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    VK_CALL(vkCreateInstance(&instance_info, NULL, &rt->instance));

    if (pick_physical_device(rt)) goto fail;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info;
    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = rt->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info;
    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VK_CALL(vkCreateDevice(rt->physical_device, &device_info, NULL, &rt->device));
    vkGetDeviceQueue(rt->device, rt->queue_family, 0, &rt->queue);

    if (create_output_buffer(rt)) goto fail;
    if (create_fill_pipeline(rt)) goto fail;
    if (create_descriptors(rt)) goto fail;
    if (create_runtime_model_slice(rt, weights)) goto fail;
    if (create_commands(rt)) goto fail;

    *out = rt;
    return 0;

fail:
    world_vulkan_runtime_destroy(rt);
    return 1;
}

static void convert_rgba_to_rgb(WorldVulkanRuntime *rt) {
    const uint32_t *rgba = (const uint32_t *)rt->output_mapped;
    unsigned char *rgb = rt->rgb_host;
    for (size_t i = 0; i < rt->pixel_count; ++i) {
        uint32_t p = rgba[i];
        rgb[i * 3 + 0] = (unsigned char)(p & 0xffu);
        rgb[i * 3 + 1] = (unsigned char)((p >> 8) & 0xffu);
        rgb[i * 3 + 2] = (unsigned char)((p >> 16) & 0xffu);
    }
}

int world_vulkan_runtime_step_rgb(
        WorldVulkanRuntime *rt,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    if (!rt || !rgb_out || !width_out || !height_out || !frames_out) return 1;
    double t0 = now_seconds();
    if (rt->model_slice_enabled) {
        if (rt->use_external_latent_once) {
            rt->use_external_latent_once = 0;
        } else {
            fill_runtime_latent(rt, control_input);
        }
    }

    WorldVulkanFillPush push;
    memset(&push, 0, sizeof(push));
    push.width = (uint32_t)rt->width;
    push.height = (uint32_t)rt->height;
    push.frames = (uint32_t)rt->frames;
    push.frame_ordinal = (uint32_t)rt->frame_ordinal;
    if (control_input) {
        push.control_x = control_input[0];
        push.control_y = control_input[1];
    }

    VK_CALL_RET(vkResetFences(rt->device, 1, &rt->fence));
    VK_CALL_RET(vkResetCommandBuffer(rt->command_buffer, 0));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL_RET(vkBeginCommandBuffer(rt->command_buffer, &begin_info));
    if (rt->model_slice_enabled) {
        if (record_runtime_model_slice(rt, control_input)) return 1;
    } else {
        vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->fill_pipeline);
        vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                rt->pipeline_layout, 0, 1, &rt->descriptor_set, 0, NULL);
        vkCmdPushConstants(rt->command_buffer, rt->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(push), &push);
        vkCmdDispatch(rt->command_buffer,
                ((uint32_t)rt->width + 15u) / 16u,
                ((uint32_t)rt->height + 15u) / 16u,
                (uint32_t)rt->frames);
    }
    VK_CALL_RET(vkEndCommandBuffer(rt->command_buffer));

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    VK_CALL_RET(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
    VK_CALL_RET(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));

    convert_rgba_to_rgb(rt);
    rt->frame_ordinal += 1;

    *rgb_out = rt->rgb_host;
    *width_out = rt->width;
    *height_out = rt->height;
    *frames_out = rt->frames;
    if (seconds_out) *seconds_out = (float)(now_seconds() - t0);
    fprintf(stderr, "Vulkan %s timing: total=%.3fms rgb_fps=%.3f\n",
            rt->model_slice_enabled ? "resident-slice" : "scaffold",
            (now_seconds() - t0) * 1000.0,
            (double)rt->frames / (now_seconds() - t0));
    return 0;
}

int world_vulkan_runtime_seed_latent_rgb(
        WorldVulkanRuntime *rt,
        const float *latent,
        const float *control_input,
        const unsigned char **rgb_out,
        int *width_out,
        int *height_out,
        int *frames_out,
        float *seconds_out) {
    if (rt && latent && rt->model_slice_enabled && rt->latent_mapped) {
        memcpy(rt->latent_mapped, latent, rt->latent_elems * sizeof(float));
        rt->use_external_latent_once = 1;
    }
    return world_vulkan_runtime_step_rgb(rt, control_input, rgb_out, width_out, height_out, frames_out, seconds_out);
}

static float probe_value(int i, float scale) {
    int v = (i * 37 + 17) % 29;
    return ((float)v - 14.0f) * scale;
}

static void fill_probe_rope_tables(float *xy, float *inv_t, int D, int height, int width) {
    int d_t = D / 4;
    int d_xy = D / 8;
    float max_freq = (float)(height < width ? height : width) * 0.8f;
    int n = (d_xy + 1) / 2;
    for (int i = 0; i < d_xy; ++i) {
        int k = i / 2;
        float a = n > 1 ? (float)k / (float)(n - 1) : 0.0f;
        xy[i] = (1.0f + (max_freq * 0.5f - 1.0f) * a) * (float)M_PI;
    }
    for (int i = 0; i < d_t; ++i) {
        int even = (i / 2) * 2;
        float exponent = (float)even / (float)d_t;
        inv_t[i] = 1.0f / powf(10000.0f, exponent);
    }
}

static float probe_rope_phase(
        int pair_id,
        uint32_t x_pos,
        uint32_t y_pos,
        uint32_t t_pos,
        const float *xy,
        const float *inv_t,
        int D,
        int width,
        int height) {
    int d_xy = D / 8;
    if (pair_id < d_xy) {
        float x = (2.0f * (float)x_pos + 1.0f) / (float)width - 1.0f;
        return x * xy[pair_id];
    }
    if (pair_id < 2 * d_xy) {
        float y = (2.0f * (float)y_pos + 1.0f) / (float)height - 1.0f;
        return y * xy[pair_id - d_xy];
    }
    return (float)t_pos * inv_t[pair_id - 2 * d_xy];
}

int world_vulkan_linear_f32_probe(void) {
    enum { rows = 7, cols = 11, inner = 13 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer b_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory b_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *b_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;

    size_t x_bytes = (size_t)rows * inner * sizeof(float);
    size_t w_bytes = (size_t)cols * inner * sizeof(float);
    size_t b_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_buffer, &b_memory, &b_mapped)) goto cleanup;
    if (create_host_buffer(rt, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    float *w = (float *)w_mapped;
    float *b = (float *)b_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * inner; ++i) x[i] = probe_value(i, 0.03125f);
    for (int i = 0; i < cols * inner; ++i) w[i] = probe_value(i + 101, 0.0234375f);
    for (int i = 0; i < cols; ++i) b[i] = probe_value(i + 211, 0.015625f);
    memset(y, 0, y_bytes);

    if (create_shader_module_from_name(rt, "linear_f32.comp", &shader)) goto cleanup;

    VkDescriptorSetLayoutBinding bindings[4];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set_info;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 4;
    set_info.pBindings = bindings;
    VK_CALL(vkCreateDescriptorSetLayout(rt->device, &set_info, NULL, &set_layout));

    VkPushConstantRange push_range;
    memset(&push_range, 0, sizeof(push_range));
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(WorldVulkanLinearPush);

    VkPipelineLayoutCreateInfo layout_info;
    memset(&layout_info, 0, sizeof(layout_info));
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VK_CALL(vkCreatePipelineLayout(rt->device, &layout_info, NULL, &pipeline_layout));

    VkPipelineShaderStageCreateInfo stage_info;
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    VK_CALL(vkCreateComputePipelines(rt->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));

    VkDescriptorPoolSize pool_size;
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 4;

    VkDescriptorPoolCreateInfo pool_info;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CALL(vkCreateDescriptorPool(rt->device, &pool_info, NULL, &descriptor_pool));

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;
    VK_CALL(vkAllocateDescriptorSets(rt->device, &alloc_info, &descriptor_set));

    VkDescriptorBufferInfo buffer_infos[4];
    memset(buffer_infos, 0, sizeof(buffer_infos));
    buffer_infos[0].buffer = x_buffer;
    buffer_infos[0].range = x_bytes;
    buffer_infos[1].buffer = w_buffer;
    buffer_infos[1].range = w_bytes;
    buffer_infos[2].buffer = b_buffer;
    buffer_infos[2].range = b_bytes;
    buffer_infos[3].buffer = y_buffer;
    buffer_infos[3].range = y_bytes;

    VkWriteDescriptorSet writes[4];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(rt->device, 4, writes, 0, NULL);

    WorldVulkanLinearPush push;
    push.rows = rows;
    push.cols = cols;
    push.inner = inner;
    push.has_bias = 1;

    VK_CALL(vkResetFences(rt->device, 1, &rt->fence));
    VK_CALL(vkResetCommandBuffer(rt->command_buffer, 0));
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(rt->command_buffer, &begin_info));
    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
    vkCmdPushConstants(rt->command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(push), &push);
    vkCmdDispatch(rt->command_buffer, (cols + 7) / 8, (rows + 7) / 8, 1);
    VK_CALL(vkEndCommandBuffer(rt->command_buffer));

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &rt->command_buffer;
    VK_CALL(vkQueueSubmit(rt->queue, 1, &submit_info, rt->fence));
    VK_CALL(vkWaitForFences(rt->device, 1, &rt->fence, VK_TRUE, UINT64_MAX));

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float ref = b[c];
            for (int k = 0; k < inner; ++k) {
                ref += x[r * inner + k] * w[c * inner + k];
            }
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan linear_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 1.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, b_buffer, b_memory, b_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;

fail:
    rc = 1;
    goto cleanup;
}

int world_vulkan_silu_f32_probe(void) {
    enum { n = 513 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t bytes = (size_t)n * sizeof(float);
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < n; ++i) x[i] = probe_value(i + 31, 0.125f);
    memset(y, 0, bytes);

    if (create_storage_pipeline(rt, "silu_f32.comp", 2, sizeof(WorldVulkanSiluPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[2] = {x_buffer, y_buffer};
    VkDeviceSize sizes[2] = {bytes, bytes};
    if (create_storage_descriptor_set(rt, set_layout, 2, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanSiluPush push;
    push.n = n;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (n + 255u) / 256u, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = x[i] / (1.0f + expf(-x[i]));
        float diff = fabsf(y[i] - ref);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= (float)n;
    fprintf(stderr, "vulkan silu_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 2.0e-6f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_rms_norm_f32_probe(void) {
    enum { rows = 5, cols = 257 };
    const float eps = 1.0e-6f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer w_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory w_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *w_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)rows * cols * sizeof(float);
    size_t w_bytes = (size_t)cols * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &w_buffer, &w_memory, &w_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *w = (float *)w_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * cols; ++i) x[i] = probe_value(i + 71, 0.0625f);
    for (int i = 0; i < cols; ++i) w[i] = 1.0f + probe_value(i + 151, 0.01171875f);
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "rms_norm_f32.comp", 3, sizeof(WorldVulkanRmsNormPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, w_buffer, y_buffer};
    VkDeviceSize sizes[3] = {x_bytes, w_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanRmsNormPush push;
    push.rows = rows;
    push.cols = cols;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                rows, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sum += v * v;
        }
        float scale = 1.0f / sqrtf(sum / (float)cols + eps);
        for (int c = 0; c < cols; ++c) {
            float ref = x[r * cols + c] * scale * w[c];
            float diff = fabsf(y[r * cols + c] - ref);
            if (diff > max_abs) max_abs = diff;
            mean_abs += diff;
        }
    }
    mean_abs /= (float)(rows * cols);
    fprintf(stderr, "vulkan rms_norm_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, w_buffer, w_memory, w_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_ada_rms_norm_f32_probe(void) {
    enum { B = 2, N = 3, M = 5, T = N * M, D = 257 };
    const float eps = 1.0e-6f;
    const int rows = B * T;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer scale_buffer = VK_NULL_HANDLE;
    VkBuffer bias_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory scale_memory = VK_NULL_HANDLE;
    VkDeviceMemory bias_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *scale_mapped = NULL;
    void *bias_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)rows * D * sizeof(float);
    size_t sb_bytes = (size_t)B * N * D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &scale_buffer, &scale_memory, &scale_mapped)) goto cleanup;
    if (create_host_buffer(rt, sb_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bias_buffer, &bias_memory, &bias_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;
    float *x = (float *)x_mapped;
    float *scale = (float *)scale_mapped;
    float *bias = (float *)bias_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < rows * D; ++i) x[i] = probe_value(i + 211, 0.046875f);
    for (int i = 0; i < B * N * D; ++i) {
        scale[i] = probe_value(i + 307, 0.0078125f);
        bias[i] = probe_value(i + 409, 0.01171875f);
    }
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "ada_rms_norm_f32.comp", 4, sizeof(WorldVulkanAdaRmsNormPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[4] = {x_buffer, scale_buffer, bias_buffer, y_buffer};
    VkDeviceSize sizes[4] = {x_bytes, sb_bytes, sb_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 4, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanAdaRmsNormPush push;
    push.B = B;
    push.T = T;
    push.N = N;
    push.D = D;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                rows, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int row = b * T + t;
            int n = t / M;
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v = x[row * D + d];
                sum += v * v;
            }
            float inv = 1.0f / sqrtf(sum / (float)D + eps);
            for (int d = 0; d < D; ++d) {
                int idx = row * D + d;
                int sb = (b * N + n) * D + d;
                float ref = x[idx] * inv * (1.0f + scale[sb]) + bias[sb];
                float diff = fabsf(y[idx] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(rows * D);
    fprintf(stderr, "vulkan ada_rms_norm_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, scale_buffer, scale_memory, scale_mapped);
        destroy_host_buffer(rt, bias_buffer, bias_memory, bias_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_ortho_rope_f32_probe(void) {
    enum { B = 2, H = 3, T = 11, D = 128, width = 20, height = 18 };
    enum { half_d = D / 2, d_xy = D / 8, d_t = D / 4 };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer x_pos_buffer = VK_NULL_HANDLE;
    VkBuffer y_pos_buffer = VK_NULL_HANDLE;
    VkBuffer t_pos_buffer = VK_NULL_HANDLE;
    VkBuffer xy_buffer = VK_NULL_HANDLE;
    VkBuffer inv_t_buffer = VK_NULL_HANDLE;
    VkBuffer y_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory x_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory t_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory xy_memory = VK_NULL_HANDLE;
    VkDeviceMemory inv_t_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *x_pos_mapped = NULL;
    void *y_pos_mapped = NULL;
    void *t_pos_mapped = NULL;
    void *xy_mapped = NULL;
    void *inv_t_mapped = NULL;
    void *y_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)B * H * T * D * sizeof(float);
    size_t pos_bytes = (size_t)T * sizeof(uint32_t);
    size_t xy_bytes = (size_t)d_xy * sizeof(float);
    size_t inv_t_bytes = (size_t)d_t * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_pos_buffer, &x_pos_memory, &x_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_pos_buffer, &y_pos_memory, &y_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &t_pos_buffer, &t_pos_memory, &t_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, xy_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &xy_buffer, &xy_memory, &xy_mapped)) goto cleanup;
    if (create_host_buffer(rt, inv_t_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &inv_t_buffer, &inv_t_memory, &inv_t_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_buffer, &y_memory, &y_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    uint32_t *x_pos = (uint32_t *)x_pos_mapped;
    uint32_t *y_pos = (uint32_t *)y_pos_mapped;
    uint32_t *t_pos = (uint32_t *)t_pos_mapped;
    float *xy = (float *)xy_mapped;
    float *inv_t = (float *)inv_t_mapped;
    float *y = (float *)y_mapped;
    for (int i = 0; i < B * H * T * D; ++i) x[i] = probe_value(i + 503, 0.03125f);
    for (int t = 0; t < T; ++t) {
        y_pos[t] = (uint32_t)(t % height);
        x_pos[t] = (uint32_t)((t * 3) % width);
        t_pos[t] = (uint32_t)(t * 7);
    }
    fill_probe_rope_tables(xy, inv_t, D, height, width);
    memset(y, 0, x_bytes);

    if (create_storage_pipeline(rt, "ortho_rope_f32.comp", 7, sizeof(WorldVulkanOrthoRopePush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[7] = {
        x_buffer, x_pos_buffer, y_pos_buffer, t_pos_buffer, xy_buffer, inv_t_buffer, y_buffer
    };
    VkDeviceSize sizes[7] = {
        x_bytes, pos_bytes, pos_bytes, pos_bytes, xy_bytes, inv_t_bytes, x_bytes
    };
    if (create_storage_descriptor_set(rt, set_layout, 7, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanOrthoRopePush push;
    push.B = B;
    push.H = H;
    push.T = T;
    push.D = D;
    push.width = width;
    push.height = height;
    uint32_t total = (uint32_t)(B * H * T * half_d);
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                (total + 255u) / 256u, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                size_t base = (size_t)(((b * H + h) * T + t) * D);
                for (int p = 0; p < half_d; ++p) {
                    float phase = probe_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, D, width, height);
                    float c = cosf(phase);
                    float s = sinf(phase);
                    float x0 = x[base + 2 * p];
                    float x1 = x[base + 2 * p + 1];
                    float ref0 = x0 * c - x1 * s;
                    float ref1 = x1 * c + x0 * s;
                    float diff0 = fabsf(y[base + p] - ref0);
                    float diff1 = fabsf(y[base + half_d + p] - ref1);
                    if (diff0 > max_abs) max_abs = diff0;
                    if (diff1 > max_abs) max_abs = diff1;
                    mean_abs += diff0 + diff1;
                }
            }
        }
    }
    mean_abs /= (float)(B * H * T * D);
    fprintf(stderr, "vulkan ortho_rope_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 5.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, x_pos_buffer, x_pos_memory, x_pos_mapped);
        destroy_host_buffer(rt, y_pos_buffer, y_pos_memory, y_pos_mapped);
        destroy_host_buffer(rt, t_pos_buffer, t_pos_memory, t_pos_mapped);
        destroy_host_buffer(rt, xy_buffer, xy_memory, xy_mapped);
        destroy_host_buffer(rt, inv_t_buffer, inv_t_memory, inv_t_mapped);
        destroy_host_buffer(rt, y_buffer, y_memory, y_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_qkv_rms_rope_f32_probe(void) {
    enum { B = 2, T = 13, n_heads = 6, n_kv_heads = 2, D = 128, width = 19, height = 16 };
    enum { total_heads = n_heads + 2 * n_kv_heads, half_d = D / 2, d_xy = D / 8, d_t = D / 4 };
    const float eps = 1.0e-6f;
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer qkv_buffer = VK_NULL_HANDLE;
    VkBuffer x_pos_buffer = VK_NULL_HANDLE;
    VkBuffer y_pos_buffer = VK_NULL_HANDLE;
    VkBuffer t_pos_buffer = VK_NULL_HANDLE;
    VkBuffer xy_buffer = VK_NULL_HANDLE;
    VkBuffer inv_t_buffer = VK_NULL_HANDLE;
    VkBuffer q_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkDeviceMemory qkv_memory = VK_NULL_HANDLE;
    VkDeviceMemory x_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory y_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory t_pos_memory = VK_NULL_HANDLE;
    VkDeviceMemory xy_memory = VK_NULL_HANDLE;
    VkDeviceMemory inv_t_memory = VK_NULL_HANDLE;
    VkDeviceMemory q_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    void *qkv_mapped = NULL;
    void *x_pos_mapped = NULL;
    void *y_pos_mapped = NULL;
    void *t_pos_mapped = NULL;
    void *xy_mapped = NULL;
    void *inv_t_mapped = NULL;
    void *q_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t qkv_bytes = (size_t)B * T * total_heads * D * sizeof(float);
    size_t q_bytes = (size_t)B * n_heads * T * D * sizeof(float);
    size_t kv_bytes = (size_t)B * n_kv_heads * T * D * sizeof(float);
    size_t pos_bytes = (size_t)T * sizeof(uint32_t);
    size_t xy_bytes = (size_t)d_xy * sizeof(float);
    size_t inv_t_bytes = (size_t)d_t * sizeof(float);
    if (create_host_buffer(rt, qkv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &qkv_buffer, &qkv_memory, &qkv_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_pos_buffer, &x_pos_memory, &x_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &y_pos_buffer, &y_pos_memory, &y_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, pos_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &t_pos_buffer, &t_pos_memory, &t_pos_mapped)) goto cleanup;
    if (create_host_buffer(rt, xy_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &xy_buffer, &xy_memory, &xy_mapped)) goto cleanup;
    if (create_host_buffer(rt, inv_t_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &inv_t_buffer, &inv_t_memory, &inv_t_mapped)) goto cleanup;
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &q_buffer, &q_memory, &q_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v_buffer, &v_memory, &v_mapped)) goto cleanup;

    float *qkv = (float *)qkv_mapped;
    uint32_t *x_pos = (uint32_t *)x_pos_mapped;
    uint32_t *y_pos = (uint32_t *)y_pos_mapped;
    uint32_t *t_pos = (uint32_t *)t_pos_mapped;
    float *xy = (float *)xy_mapped;
    float *inv_t = (float *)inv_t_mapped;
    float *q = (float *)q_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    for (int i = 0; i < B * T * total_heads * D; ++i) qkv[i] = probe_value(i + 601, 0.02734375f);
    for (int t = 0; t < T; ++t) {
        y_pos[t] = (uint32_t)(t % height);
        x_pos[t] = (uint32_t)((t * 5) % width);
        t_pos[t] = (uint32_t)(t * 2);
    }
    fill_probe_rope_tables(xy, inv_t, D, height, width);
    memset(q, 0, q_bytes);
    memset(k, 0, kv_bytes);
    memset(v, 0, kv_bytes);

    if (create_storage_pipeline(rt, "qkv_rms_rope_f32.comp", 9, sizeof(WorldVulkanQkvRmsRopePush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[9] = {
        qkv_buffer, x_pos_buffer, y_pos_buffer, t_pos_buffer, xy_buffer, inv_t_buffer, q_buffer, k_buffer, v_buffer
    };
    VkDeviceSize sizes[9] = {
        qkv_bytes, pos_bytes, pos_bytes, pos_bytes, xy_bytes, inv_t_bytes, q_bytes, kv_bytes, kv_bytes
    };
    if (create_storage_descriptor_set(rt, set_layout, 9, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanQkvRmsRopePush push;
    push.B = B;
    push.T = T;
    push.n_heads = n_heads;
    push.n_kv_heads = n_kv_heads;
    push.D = D;
    push.width = width;
    push.height = height;
    push.eps = eps;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * T, total_heads, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int compared = 0;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int role = 0; role < n_heads + n_kv_heads; ++role) {
                int is_k = role >= n_heads;
                int h = is_k ? role - n_heads : role;
                int src_head = is_k ? n_heads + h : h;
                const float *src = qkv + ((b * T + t) * total_heads + src_head) * D;
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float z = src[d];
                    sum += z * z;
                }
                float inv = 1.0f / sqrtf(sum / (float)D + eps);
                float *dst = is_k
                    ? k + ((b * n_kv_heads + h) * T + t) * D
                    : q + ((b * n_heads + h) * T + t) * D;
                for (int p = 0; p < half_d; ++p) {
                    float phase = probe_rope_phase(p, x_pos[t], y_pos[t], t_pos[t], xy, inv_t, D, width, height);
                    float c = cosf(phase);
                    float s = sinf(phase);
                    float a = src[2 * p] * inv;
                    float bb = src[2 * p + 1] * inv;
                    float ref0 = a * c - bb * s;
                    float ref1 = bb * c + a * s;
                    float diff0 = fabsf(dst[p] - ref0);
                    float diff1 = fabsf(dst[half_d + p] - ref1);
                    if (diff0 > max_abs) max_abs = diff0;
                    if (diff1 > max_abs) max_abs = diff1;
                    mean_abs += diff0 + diff1;
                    compared += 2;
                }
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                const float *src = qkv + ((b * T + t) * total_heads + n_heads + n_kv_heads + h) * D;
                const float *dst = v + ((b * n_kv_heads + h) * T + t) * D;
                for (int d = 0; d < D; ++d) {
                    float diff = fabsf(dst[d] - src[d]);
                    if (diff > max_abs) max_abs = diff;
                    mean_abs += diff;
                    compared += 1;
                }
            }
        }
    }
    mean_abs /= (float)compared;
    fprintf(stderr, "vulkan qkv_rms_rope_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 5.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, qkv_buffer, qkv_memory, qkv_mapped);
        destroy_host_buffer(rt, x_pos_buffer, x_pos_memory, x_pos_mapped);
        destroy_host_buffer(rt, y_pos_buffer, y_pos_memory, y_pos_mapped);
        destroy_host_buffer(rt, t_pos_buffer, t_pos_memory, t_pos_mapped);
        destroy_host_buffer(rt, xy_buffer, xy_memory, xy_mapped);
        destroy_host_buffer(rt, inv_t_buffer, inv_t_memory, inv_t_mapped);
        destroy_host_buffer(rt, q_buffer, q_memory, q_mapped);
        destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
        destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_masked_attention_f32_probe(void) {
    enum { B = 2, Hq = 6, Hkv = 2, Tq = 7, Tk = 11, D = 64 };
    const float scale = 1.0f / sqrtf((float)D);
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer q_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkBuffer written_buffer = VK_NULL_HANDLE;
    VkBuffer out_buffer = VK_NULL_HANDLE;
    VkDeviceMemory q_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    VkDeviceMemory written_memory = VK_NULL_HANDLE;
    VkDeviceMemory out_memory = VK_NULL_HANDLE;
    void *q_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;
    void *written_mapped = NULL;
    void *out_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t q_bytes = (size_t)B * Hq * Tq * D * sizeof(float);
    size_t kv_bytes = (size_t)B * Hkv * Tk * D * sizeof(float);
    size_t written_bytes = (size_t)Tk * sizeof(uint32_t);
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &q_buffer, &q_memory, &q_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v_buffer, &v_memory, &v_mapped)) goto cleanup;
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &written_buffer, &written_memory, &written_mapped)) goto cleanup;
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) goto cleanup;

    float *q = (float *)q_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    uint32_t *written = (uint32_t *)written_mapped;
    float *out = (float *)out_mapped;
    for (int i = 0; i < B * Hq * Tq * D; ++i) q[i] = probe_value(i + 701, 0.0234375f);
    for (int i = 0; i < B * Hkv * Tk * D; ++i) {
        k[i] = probe_value(i + 809, 0.021484375f);
        v[i] = probe_value(i + 907, 0.033203125f);
    }
    const uint32_t written_ref[Tk] = {1u, 1u, 0u, 1u, 0u, 1u, 1u, 0u, 1u, 0u, 1u};
    memcpy(written, written_ref, written_bytes);
    memset(out, 0, q_bytes);

    if (create_storage_pipeline(rt, "masked_attention_f32.comp", 5, sizeof(WorldVulkanMaskedAttentionPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[5] = {q_buffer, k_buffer, v_buffer, written_buffer, out_buffer};
    VkDeviceSize sizes[5] = {q_bytes, kv_bytes, kv_bytes, written_bytes, q_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 5, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanMaskedAttentionPush push;
    push.B = B;
    push.Hq = Hq;
    push.Hkv = Hkv;
    push.Tq = Tq;
    push.Tk = Tk;
    push.D = D;
    push.scale = scale;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * Hq * Tq, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int group = Hq / Hkv;
    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < Hq; ++hq) {
            int hk = hq / group;
            for (int tq = 0; tq < Tq; ++tq) {
                const float *qrow = q + ((b * Hq + hq) * Tq + tq) * D;
                const float *kbase = k + (b * Hkv + hk) * Tk * D;
                const float *vbase = v + (b * Hkv + hk) * Tk * D;
                float scores[Tk];
                float max_score = -INFINITY;
                for (int tk = 0; tk < Tk; ++tk) {
                    if (!written[tk]) {
                        scores[tk] = -INFINITY;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += qrow[d] * kbase[tk * D + d];
                    }
                    scores[tk] = dot * scale;
                    if (scores[tk] > max_score) max_score = scores[tk];
                }
                float denom = 0.0f;
                for (int tk = 0; tk < Tk; ++tk) {
                    if (written[tk]) denom += expf(scores[tk] - max_score);
                }
                for (int d = 0; d < D; ++d) {
                    float ref = 0.0f;
                    if (denom > 0.0f) {
                        for (int tk = 0; tk < Tk; ++tk) {
                            if (written[tk]) {
                                ref += expf(scores[tk] - max_score) * vbase[tk * D + d];
                            }
                        }
                        ref /= denom;
                    }
                    size_t idx = (size_t)(((b * Hq + hq) * Tq + tq) * D + d);
                    float diff = fabsf(out[idx] - ref);
                    if (diff > max_abs) max_abs = diff;
                    mean_abs += diff;
                }
            }
        }
    }
    mean_abs /= (float)(B * Hq * Tq * D);
    fprintf(stderr, "vulkan masked_attention_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, q_buffer, q_memory, q_mapped);
        destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
        destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
        destroy_host_buffer(rt, written_buffer, written_memory, written_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

static int run_kv_cache_upsert_probe_case(
        WorldVulkanRuntime *rt,
        VkDescriptorSetLayout mask_set_layout,
        VkPipelineLayout mask_pipeline_layout,
        VkPipeline mask_pipeline,
        VkDescriptorSetLayout upsert_set_layout,
        VkPipelineLayout upsert_pipeline_layout,
        VkPipeline upsert_pipeline,
        const char *name,
        uint32_t B,
        uint32_t H,
        uint32_t T,
        uint32_t D,
        uint32_t L,
        uint32_t frame_idx,
        uint32_t pinned_dilation,
        uint32_t frozen) {
    int rc = 1;
    uint32_t capacity = L + T;
    uint32_t bucket = (frame_idx + pinned_dilation - 1u) / pinned_dilation;
    uint32_t num_buckets = (L / T) / pinned_dilation;
    uint32_t base = (bucket % num_buckets) * T;
    uint32_t write_step = (frame_idx % pinned_dilation) == 0u ? 1u : 0u;
    size_t cache_values = (size_t)B * H * capacity * D;
    size_t kv_values = (size_t)B * H * T * D;
    size_t cache_bytes = cache_values * sizeof(float);
    size_t kv_bytes = kv_values * sizeof(float);
    size_t written_bytes = (size_t)capacity * sizeof(uint32_t);

    VkDescriptorPool mask_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet mask_descriptor_set = VK_NULL_HANDLE;
    VkDescriptorPool upsert_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet upsert_descriptor_set = VK_NULL_HANDLE;
    VkBuffer cache_k_buffer = VK_NULL_HANDLE;
    VkBuffer cache_v_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkBuffer written_buffer = VK_NULL_HANDLE;
    VkBuffer mask_buffer = VK_NULL_HANDLE;
    VkDeviceMemory cache_k_memory = VK_NULL_HANDLE;
    VkDeviceMemory cache_v_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    VkDeviceMemory written_memory = VK_NULL_HANDLE;
    VkDeviceMemory mask_memory = VK_NULL_HANDLE;
    void *cache_k_mapped = NULL;
    void *cache_v_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;
    void *written_mapped = NULL;
    void *mask_mapped = NULL;
    float *ref_cache_k = NULL;
    float *ref_cache_v = NULL;
    uint32_t *ref_written = NULL;
    uint32_t *ref_mask = NULL;

    if (create_host_buffer(rt, cache_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &cache_k_buffer, &cache_k_memory, &cache_k_mapped)) goto cleanup;
    if (create_host_buffer(rt, cache_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &cache_v_buffer, &cache_v_memory, &cache_v_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &v_buffer, &v_memory, &v_mapped)) goto cleanup;
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &written_buffer, &written_memory, &written_mapped)) goto cleanup;
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &mask_buffer, &mask_memory, &mask_mapped)) goto cleanup;

    float *cache_k = (float *)cache_k_mapped;
    float *cache_v = (float *)cache_v_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    uint32_t *written = (uint32_t *)written_mapped;
    uint32_t *mask = (uint32_t *)mask_mapped;

    for (size_t i = 0; i < cache_values; ++i) {
        cache_k[i] = frozen ? probe_value((int)i + 1009, 0.017578125f) : 0.0f;
        cache_v[i] = frozen ? probe_value((int)i + 1201, 0.01953125f) : 0.0f;
    }
    for (size_t i = 0; i < kv_values; ++i) {
        k[i] = probe_value((int)i + 1409, 0.025390625f);
        v[i] = probe_value((int)i + 1601, 0.029296875f);
    }
    memset(written, 0, written_bytes);
    uint32_t initially_written = frozen ? T : 2u * T;
    for (uint32_t i = 0; i < initially_written && i < capacity; ++i) written[i] = 1u;
    for (uint32_t i = L; i < capacity; ++i) written[i] = 1u;
    memset(mask, 0, written_bytes);

    ref_cache_k = (float *)malloc(cache_bytes);
    ref_cache_v = (float *)malloc(cache_bytes);
    ref_written = (uint32_t *)malloc(written_bytes);
    ref_mask = (uint32_t *)malloc(written_bytes);
    if (!ref_cache_k || !ref_cache_v || !ref_written || !ref_mask) goto cleanup;
    memcpy(ref_cache_k, cache_k, cache_bytes);
    memcpy(ref_cache_v, cache_v, cache_bytes);
    memcpy(ref_written, written, written_bytes);
    memcpy(ref_mask, written, written_bytes);

    if (write_step) {
        for (uint32_t t = 0; t < T; ++t) ref_mask[base + t] = 0u;
    }
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t t = 0; t < T; ++t) {
                uint32_t tail_idx = L + t;
                uint32_t ring_idx = base + t;
                uint32_t dst_idx = (!frozen && write_step) ? ring_idx : tail_idx;
                for (uint32_t d = 0; d < D; ++d) {
                    size_t src = (size_t)(((b * H + h) * T + t) * D + d);
                    size_t tail = (size_t)(((b * H + h) * capacity + tail_idx) * D + d);
                    size_t dst = (size_t)(((b * H + h) * capacity + dst_idx) * D + d);
                    ref_cache_k[tail] = k[src];
                    ref_cache_v[tail] = v[src];
                    if (!frozen) {
                        ref_cache_k[dst] = k[src];
                        ref_cache_v[dst] = v[src];
                    }
                }
            }
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t tail_idx = L + t;
        uint32_t dst_idx = (!frozen && write_step) ? base + t : tail_idx;
        ref_written[tail_idx] = 1u;
        if (!frozen) ref_written[dst_idx] = 1u;
    }

    VkBuffer mask_buffers[2] = {written_buffer, mask_buffer};
    VkDeviceSize mask_sizes[2] = {written_bytes, written_bytes};
    if (create_storage_descriptor_set(rt, mask_set_layout, 2, mask_buffers, mask_sizes,
                &mask_descriptor_pool, &mask_descriptor_set)) goto cleanup;
    WorldVulkanKvCacheMaskPush mask_push;
    mask_push.capacity = capacity;
    mask_push.T = T;
    mask_push.base = base;
    mask_push.write_step = write_step;
    if (submit_compute(rt, mask_pipeline, mask_pipeline_layout, mask_descriptor_set,
                &mask_push, sizeof(mask_push), (capacity + 255u) / 256u, 1, 1)) goto cleanup;

    VkBuffer upsert_buffers[5] = {cache_k_buffer, cache_v_buffer, k_buffer, v_buffer, written_buffer};
    VkDeviceSize upsert_sizes[5] = {cache_bytes, cache_bytes, kv_bytes, kv_bytes, written_bytes};
    if (create_storage_descriptor_set(rt, upsert_set_layout, 5, upsert_buffers, upsert_sizes,
                &upsert_descriptor_pool, &upsert_descriptor_set)) goto cleanup;
    WorldVulkanKvCacheUpsertPush upsert_push;
    upsert_push.B = B;
    upsert_push.H = H;
    upsert_push.T = T;
    upsert_push.D = D;
    upsert_push.L = L;
    upsert_push.base = base;
    upsert_push.write_step = write_step;
    upsert_push.frozen = frozen;
    if (submit_compute(rt, upsert_pipeline, upsert_pipeline_layout, upsert_descriptor_set,
                &upsert_push, sizeof(upsert_push), (uint32_t)((kv_values + 255u) / 256u), 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    uint32_t mismatches = 0;
    for (size_t i = 0; i < cache_values; ++i) {
        float dk = fabsf(cache_k[i] - ref_cache_k[i]);
        float dv = fabsf(cache_v[i] - ref_cache_v[i]);
        if (dk > max_abs) max_abs = dk;
        if (dv > max_abs) max_abs = dv;
        if (dk != 0.0f || dv != 0.0f) ++mismatches;
    }
    for (uint32_t i = 0; i < capacity; ++i) {
        if (written[i] != ref_written[i]) ++mismatches;
        if (mask[i] != ref_mask[i]) ++mismatches;
    }
    fprintf(stderr, "vulkan kv_cache_upsert_f32 probe (%s): max_abs=%g mismatches=%u\n",
            name, max_abs, mismatches);
    if (mismatches != 0 || max_abs != 0.0f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (mask_descriptor_pool) vkDestroyDescriptorPool(rt->device, mask_descriptor_pool, NULL);
    if (upsert_descriptor_pool) vkDestroyDescriptorPool(rt->device, upsert_descriptor_pool, NULL);
    destroy_host_buffer(rt, cache_k_buffer, cache_k_memory, cache_k_mapped);
    destroy_host_buffer(rt, cache_v_buffer, cache_v_memory, cache_v_mapped);
    destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
    destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
    destroy_host_buffer(rt, written_buffer, written_memory, written_mapped);
    destroy_host_buffer(rt, mask_buffer, mask_memory, mask_mapped);
    free(ref_cache_k);
    free(ref_cache_v);
    free(ref_written);
    free(ref_mask);
    return rc;
}

int world_vulkan_kv_cache_upsert_f32_probe(void) {
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule mask_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout mask_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout mask_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline mask_pipeline = VK_NULL_HANDLE;
    VkShaderModule upsert_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout upsert_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout upsert_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline upsert_pipeline = VK_NULL_HANDLE;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    if (create_storage_pipeline(rt, "kv_cache_mask.comp", 2, sizeof(WorldVulkanKvCacheMaskPush),
                &mask_shader, &mask_set_layout, &mask_pipeline_layout, &mask_pipeline)) goto cleanup;
    if (create_storage_pipeline(rt, "kv_cache_upsert_copy_f32.comp", 5, sizeof(WorldVulkanKvCacheUpsertPush),
                &upsert_shader, &upsert_set_layout, &upsert_pipeline_layout, &upsert_pipeline)) goto cleanup;

    if (run_kv_cache_upsert_probe_case(rt, mask_set_layout, mask_pipeline_layout, mask_pipeline,
                upsert_set_layout, upsert_pipeline_layout, upsert_pipeline,
                "frozen_write_step", 1, 2, 4, 8, 16, 4, 1, 1)) goto cleanup;
    if (run_kv_cache_upsert_probe_case(rt, mask_set_layout, mask_pipeline_layout, mask_pipeline,
                upsert_set_layout, upsert_pipeline_layout, upsert_pipeline,
                "unfrozen_pinned_dilation", 1, 3, 4, 16, 32, 5, 2, 0)) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (mask_pipeline) vkDestroyPipeline(rt->device, mask_pipeline, NULL);
    if (mask_pipeline_layout) vkDestroyPipelineLayout(rt->device, mask_pipeline_layout, NULL);
    if (mask_set_layout) vkDestroyDescriptorSetLayout(rt->device, mask_set_layout, NULL);
    if (mask_shader) vkDestroyShaderModule(rt->device, mask_shader, NULL);
    if (upsert_pipeline) vkDestroyPipeline(rt->device, upsert_pipeline, NULL);
    if (upsert_pipeline_layout) vkDestroyPipelineLayout(rt->device, upsert_pipeline_layout, NULL);
    if (upsert_set_layout) vkDestroyDescriptorSetLayout(rt->device, upsert_set_layout, NULL);
    if (upsert_shader) vkDestroyShaderModule(rt->device, upsert_shader, NULL);
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_cache_frame_indices_probe(void) {
    enum { T = 4, slots = 9, capacity = T * slots };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer written_buffer = VK_NULL_HANDLE;
    VkBuffer indices_buffer = VK_NULL_HANDLE;
    VkBuffer count_buffer = VK_NULL_HANDLE;
    VkDeviceMemory written_memory = VK_NULL_HANDLE;
    VkDeviceMemory indices_memory = VK_NULL_HANDLE;
    VkDeviceMemory count_memory = VK_NULL_HANDLE;
    void *written_mapped = NULL;
    void *indices_mapped = NULL;
    void *count_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t written_bytes = (size_t)capacity * sizeof(uint32_t);
    size_t indices_bytes = (size_t)capacity * sizeof(uint32_t);
    size_t count_bytes = sizeof(uint32_t);
    if (create_host_buffer(rt, written_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &written_buffer, &written_memory, &written_mapped)) goto cleanup;
    if (create_host_buffer(rt, indices_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &indices_buffer, &indices_memory, &indices_mapped)) goto cleanup;
    if (create_host_buffer(rt, count_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &count_buffer, &count_memory, &count_mapped)) goto cleanup;

    uint32_t *written = (uint32_t *)written_mapped;
    uint32_t *indices = (uint32_t *)indices_mapped;
    uint32_t *count = (uint32_t *)count_mapped;
    memset(written, 0, written_bytes);
    const uint32_t written_slots[4] = {0u, 2u, 5u, 8u};
    for (uint32_t s = 0; s < 4u; ++s) {
        uint32_t slot = written_slots[s];
        for (uint32_t t = 0; t < T; ++t) written[slot * T + t] = 1u;
    }

    if (create_storage_pipeline(rt, "cache_frame_indices.comp", 3, sizeof(WorldVulkanCacheFrameIndicesPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {written_buffer, indices_buffer, count_buffer};
    VkDeviceSize sizes[3] = {written_bytes, indices_bytes, count_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;

    for (uint32_t case_id = 0; case_id < 2u; ++case_id) {
        uint32_t base = case_id == 0u ? 2u * T : 0u;
        uint32_t write_step = case_id == 0u ? 1u : 0u;
        for (uint32_t i = 0; i < capacity; ++i) indices[i] = 0xffffffffu;
        count[0] = 0u;

        WorldVulkanCacheFrameIndicesPush push;
        push.capacity = capacity;
        push.T = T;
        push.base = base;
        push.write_step = write_step;
        if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                    slots, 1, 1)) goto cleanup;

        uint32_t ref_indices[capacity];
        uint32_t ref_count = 0u;
        for (uint32_t slot = 0; slot < slots; ++slot) {
            uint32_t slot_base = slot * T;
            uint32_t slot_written = written[slot_base] && !(write_step && slot_base == base);
            if (slot_written) {
                for (uint32_t t = 0; t < T; ++t) ref_indices[ref_count++] = slot_base + t;
            }
        }

        uint32_t mismatches = count[0] == ref_count ? 0u : 1u;
        for (uint32_t i = 0; i < ref_count; ++i) {
            if (indices[i] != ref_indices[i]) ++mismatches;
        }
        fprintf(stderr, "vulkan cache_frame_indices probe (%s): count=%u mismatches=%u\n",
                write_step ? "write_step" : "read_step", count[0], mismatches);
        if (mismatches != 0u) goto cleanup;
    }
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, written_buffer, written_memory, written_mapped);
        destroy_host_buffer(rt, indices_buffer, indices_memory, indices_mapped);
        destroy_host_buffer(rt, count_buffer, count_memory, count_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_patchify_f32_probe(void) {
    enum { B = 2, C = 5, H = 8, W = 10, D = 7, ph = 2, pw = 2 };
    enum { Hp = H / ph, Wp = W / pw, T = Hp * Wp };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkBuffer weight_buffer = VK_NULL_HANDLE;
    VkBuffer tokens_buffer = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    VkDeviceMemory weight_memory = VK_NULL_HANDLE;
    VkDeviceMemory tokens_memory = VK_NULL_HANDLE;
    void *x_mapped = NULL;
    void *weight_mapped = NULL;
    void *tokens_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t x_bytes = (size_t)B * C * H * W * sizeof(float);
    size_t weight_bytes = (size_t)D * C * ph * pw * sizeof(float);
    size_t tokens_bytes = (size_t)B * T * D * sizeof(float);
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;
    if (create_host_buffer(rt, weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &weight_buffer, &weight_memory, &weight_mapped)) goto cleanup;
    if (create_host_buffer(rt, tokens_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tokens_buffer, &tokens_memory, &tokens_mapped)) goto cleanup;

    float *x = (float *)x_mapped;
    float *weight = (float *)weight_mapped;
    float *tokens = (float *)tokens_mapped;
    for (int i = 0; i < B * C * H * W; ++i) x[i] = probe_value(i + 1801, 0.02734375f);
    for (int i = 0; i < D * C * ph * pw; ++i) weight[i] = probe_value(i + 1907, 0.033203125f);
    memset(tokens, 0, tokens_bytes);

    if (create_storage_pipeline(rt, "patchify_f32.comp", 3, sizeof(WorldVulkanPatchifyPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[3] = {x_buffer, weight_buffer, tokens_buffer};
    VkDeviceSize sizes[3] = {x_bytes, weight_bytes, tokens_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 3, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanPatchifyPush push;
    push.B = B;
    push.C = C;
    push.H = H;
    push.W = W;
    push.D = D;
    push.ph = ph;
    push.pw = pw;
    push.Hp = Hp;
    push.Wp = Wp;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * T * D, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int token = 0; token < T; ++token) {
            int oy = token / Wp;
            int ox = token - oy * Wp;
            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                for (int c = 0; c < C; ++c) {
                    for (int dy = 0; dy < ph; ++dy) {
                        for (int dx = 0; dx < pw; ++dx) {
                            int iy = oy * ph + dy;
                            int ix = ox * pw + dx;
                            float xv = x[((b * C + c) * H + iy) * W + ix];
                            float wv = weight[(((d * C + c) * ph + dy) * pw + dx)];
                            ref += xv * wv;
                        }
                    }
                }
                size_t idx = (size_t)((b * T + token) * D + d);
                float diff = fabsf(tokens[idx] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(B * T * D);
    fprintf(stderr, "vulkan patchify_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 3.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
        destroy_host_buffer(rt, weight_buffer, weight_memory, weight_mapped);
        destroy_host_buffer(rt, tokens_buffer, tokens_memory, tokens_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_unpatchify_f32_probe(void) {
    enum { B = 2, C = 4, H = 6, W = 8, ph = 2, pw = 2, D = 9 };
    enum { Hp = H / ph, Wp = W / pw, T = Hp * Wp, out_dim = C * ph * pw };
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer tokens_buffer = VK_NULL_HANDLE;
    VkBuffer weight_buffer = VK_NULL_HANDLE;
    VkBuffer bias_buffer = VK_NULL_HANDLE;
    VkBuffer x_buffer = VK_NULL_HANDLE;
    VkDeviceMemory tokens_memory = VK_NULL_HANDLE;
    VkDeviceMemory weight_memory = VK_NULL_HANDLE;
    VkDeviceMemory bias_memory = VK_NULL_HANDLE;
    VkDeviceMemory x_memory = VK_NULL_HANDLE;
    void *tokens_mapped = NULL;
    void *weight_mapped = NULL;
    void *bias_mapped = NULL;
    void *x_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t tokens_bytes = (size_t)B * T * D * sizeof(float);
    size_t weight_bytes = (size_t)out_dim * D * sizeof(float);
    size_t bias_bytes = (size_t)out_dim * sizeof(float);
    size_t x_bytes = (size_t)B * C * H * W * sizeof(float);
    if (create_host_buffer(rt, tokens_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tokens_buffer, &tokens_memory, &tokens_mapped)) goto cleanup;
    if (create_host_buffer(rt, weight_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &weight_buffer, &weight_memory, &weight_mapped)) goto cleanup;
    if (create_host_buffer(rt, bias_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bias_buffer, &bias_memory, &bias_mapped)) goto cleanup;
    if (create_host_buffer(rt, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &x_buffer, &x_memory, &x_mapped)) goto cleanup;

    float *tokens = (float *)tokens_mapped;
    float *weight = (float *)weight_mapped;
    float *bias = (float *)bias_mapped;
    float *x = (float *)x_mapped;
    for (int i = 0; i < B * T * D; ++i) tokens[i] = probe_value(i + 2003, 0.029296875f);
    for (int i = 0; i < out_dim * D; ++i) weight[i] = probe_value(i + 2111, 0.025390625f);
    for (int i = 0; i < out_dim; ++i) bias[i] = probe_value(i + 2203, 0.017578125f);
    memset(x, 0, x_bytes);

    if (create_storage_pipeline(rt, "unpatchify_f32.comp", 4, sizeof(WorldVulkanUnpatchifyPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[4] = {tokens_buffer, weight_buffer, bias_buffer, x_buffer};
    VkDeviceSize sizes[4] = {tokens_bytes, weight_bytes, bias_bytes, x_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 4, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanUnpatchifyPush push;
    push.B = B;
    push.T = T;
    push.D = D;
    push.C = C;
    push.H = H;
    push.W = W;
    push.ph = ph;
    push.pw = pw;
    push.Hp = Hp;
    push.Wp = Wp;
    push.out_dim = out_dim;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * T * out_dim, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
        for (int token = 0; token < T; ++token) {
            int oy = token / Wp;
            int ox = token - oy * Wp;
            for (int o = 0; o < out_dim; ++o) {
                float ref = bias[o];
                for (int d = 0; d < D; ++d) {
                    ref += tokens[(b * T + token) * D + d] * weight[o * D + d];
                }
                int p = o;
                int dx = p % pw;
                p /= pw;
                int dy = p % ph;
                p /= ph;
                int c = p;
                int iy = oy * ph + dy;
                int ix = ox * pw + dx;
                size_t idx = (size_t)(((b * C + c) * H + iy) * W + ix);
                float diff = fabsf(x[idx] - ref);
                if (diff > max_abs) max_abs = diff;
                mean_abs += diff;
            }
        }
    }
    mean_abs /= (float)(B * C * H * W);
    fprintf(stderr, "vulkan unpatchify_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 3.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, tokens_buffer, tokens_memory, tokens_mapped);
        destroy_host_buffer(rt, weight_buffer, weight_memory, weight_mapped);
        destroy_host_buffer(rt, bias_buffer, bias_memory, bias_mapped);
        destroy_host_buffer(rt, x_buffer, x_memory, x_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

int world_vulkan_indexed_attention_f32_probe(void) {
    enum { B = 1, Hq = 8, Hkv = 2, Tq = 9, Tk = 17, D = 64, Nkv = 5 };
    const float scale = 1.0f / sqrtf((float)D);
    int rc = 1;
    WorldConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 32;
    cfg.height = 1;
    cfg.width = 1;
    cfg.patch_h = 1;
    cfg.patch_w = 1;

    WorldVulkanRuntime *rt = NULL;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkBuffer q_buffer = VK_NULL_HANDLE;
    VkBuffer k_buffer = VK_NULL_HANDLE;
    VkBuffer v_buffer = VK_NULL_HANDLE;
    VkBuffer indices_buffer = VK_NULL_HANDLE;
    VkBuffer out_buffer = VK_NULL_HANDLE;
    VkDeviceMemory q_memory = VK_NULL_HANDLE;
    VkDeviceMemory k_memory = VK_NULL_HANDLE;
    VkDeviceMemory v_memory = VK_NULL_HANDLE;
    VkDeviceMemory indices_memory = VK_NULL_HANDLE;
    VkDeviceMemory out_memory = VK_NULL_HANDLE;
    void *q_mapped = NULL;
    void *k_mapped = NULL;
    void *v_mapped = NULL;
    void *indices_mapped = NULL;
    void *out_mapped = NULL;

    if (world_vulkan_runtime_create(&rt, &cfg, NULL, 0, 0, 0, 1234, WORLD_NOISE_NORMAL, NULL)) goto cleanup;
    size_t q_bytes = (size_t)B * Hq * Tq * D * sizeof(float);
    size_t kv_bytes = (size_t)B * Hkv * Tk * D * sizeof(float);
    size_t indices_bytes = (size_t)Nkv * sizeof(uint32_t);
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &q_buffer, &q_memory, &q_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_buffer, &k_memory, &k_mapped)) goto cleanup;
    if (create_host_buffer(rt, kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v_buffer, &v_memory, &v_mapped)) goto cleanup;
    if (create_host_buffer(rt, indices_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &indices_buffer, &indices_memory, &indices_mapped)) goto cleanup;
    if (create_host_buffer(rt, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_buffer, &out_memory, &out_mapped)) goto cleanup;

    float *q = (float *)q_mapped;
    float *k = (float *)k_mapped;
    float *v = (float *)v_mapped;
    uint32_t *indices = (uint32_t *)indices_mapped;
    float *out = (float *)out_mapped;
    for (int i = 0; i < B * Hq * Tq * D; ++i) q[i] = probe_value(i + 2309, 0.021484375f);
    for (int i = 0; i < B * Hkv * Tk * D; ++i) {
        k[i] = probe_value(i + 2411, 0.01953125f);
        v[i] = probe_value(i + 2503, 0.02734375f);
    }
    const uint32_t indices_ref[Nkv] = {0u, 3u, 4u, 9u, 16u};
    memcpy(indices, indices_ref, indices_bytes);
    memset(out, 0, q_bytes);

    if (create_storage_pipeline(rt, "indexed_attention_f32.comp", 5, sizeof(WorldVulkanIndexedAttentionPush),
                &shader, &set_layout, &pipeline_layout, &pipeline)) goto cleanup;
    VkBuffer buffers[5] = {q_buffer, k_buffer, v_buffer, indices_buffer, out_buffer};
    VkDeviceSize sizes[5] = {q_bytes, kv_bytes, kv_bytes, indices_bytes, q_bytes};
    if (create_storage_descriptor_set(rt, set_layout, 5, buffers, sizes, &descriptor_pool, &descriptor_set)) goto cleanup;
    WorldVulkanIndexedAttentionPush push;
    push.B = B;
    push.Hq = Hq;
    push.Hkv = Hkv;
    push.Tq = Tq;
    push.Nkv = Nkv;
    push.Tk = Tk;
    push.D = D;
    push.scale = scale;
    if (submit_compute(rt, pipeline, pipeline_layout, descriptor_set, &push, sizeof(push),
                B * Hq * Tq, 1, 1)) goto cleanup;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int group = Hq / Hkv;
    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < Hq; ++hq) {
            int hk = hq / group;
            for (int tq = 0; tq < Tq; ++tq) {
                const float *qrow = q + ((b * Hq + hq) * Tq + tq) * D;
                const float *kbase = k + (b * Hkv + hk) * Tk * D;
                const float *vbase = v + (b * Hkv + hk) * Tk * D;
                float scores[Nkv];
                float max_score = -INFINITY;
                for (int n = 0; n < Nkv; ++n) {
                    int tk = (int)indices[n];
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += qrow[d] * kbase[tk * D + d];
                    scores[n] = dot * scale;
                    if (scores[n] > max_score) max_score = scores[n];
                }
                float denom = 0.0f;
                for (int n = 0; n < Nkv; ++n) denom += expf(scores[n] - max_score);
                for (int d = 0; d < D; ++d) {
                    float ref = 0.0f;
                    for (int n = 0; n < Nkv; ++n) {
                        int tk = (int)indices[n];
                        ref += expf(scores[n] - max_score) * vbase[tk * D + d];
                    }
                    ref /= denom;
                    size_t idx = (size_t)(((b * Hq + hq) * Tq + tq) * D + d);
                    float diff = fabsf(out[idx] - ref);
                    if (diff > max_abs) max_abs = diff;
                    mean_abs += diff;
                }
            }
        }
    }
    mean_abs /= (float)(B * Hq * Tq * D);
    fprintf(stderr, "vulkan indexed_attention_f32 probe: max_abs=%g mean_abs=%g\n", max_abs, mean_abs);
    if (max_abs > 4.0e-5f) goto cleanup;
    rc = 0;

cleanup:
    if (rt && rt->device) vkDeviceWaitIdle(rt->device);
    if (pipeline) vkDestroyPipeline(rt->device, pipeline, NULL);
    if (pipeline_layout) vkDestroyPipelineLayout(rt->device, pipeline_layout, NULL);
    if (descriptor_pool) vkDestroyDescriptorPool(rt->device, descriptor_pool, NULL);
    if (set_layout) vkDestroyDescriptorSetLayout(rt->device, set_layout, NULL);
    if (shader) vkDestroyShaderModule(rt->device, shader, NULL);
    if (rt) {
        destroy_host_buffer(rt, q_buffer, q_memory, q_mapped);
        destroy_host_buffer(rt, k_buffer, k_memory, k_mapped);
        destroy_host_buffer(rt, v_buffer, v_memory, v_mapped);
        destroy_host_buffer(rt, indices_buffer, indices_memory, indices_mapped);
        destroy_host_buffer(rt, out_buffer, out_memory, out_mapped);
    }
    world_vulkan_runtime_destroy(rt);
    return rc;
}

void world_vulkan_runtime_destroy(WorldVulkanRuntime *rt) {
    if (!rt) return;
    if (rt->device) vkDeviceWaitIdle(rt->device);
    if (rt->patchify_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->patchify_descriptor_pool, NULL);
    if (rt->unpatch_orig_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->unpatch_orig_descriptor_pool, NULL);
    if (rt->latent_rgba_descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->latent_rgba_descriptor_pool, NULL);
    if (rt->patchify_pipeline) vkDestroyPipeline(rt->device, rt->patchify_pipeline, NULL);
    if (rt->unpatch_orig_pipeline) vkDestroyPipeline(rt->device, rt->unpatch_orig_pipeline, NULL);
    if (rt->latent_rgba_pipeline) vkDestroyPipeline(rt->device, rt->latent_rgba_pipeline, NULL);
    if (rt->patchify_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->patchify_pipeline_layout, NULL);
    if (rt->unpatch_orig_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->unpatch_orig_pipeline_layout, NULL);
    if (rt->latent_rgba_pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->latent_rgba_pipeline_layout, NULL);
    if (rt->patchify_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->patchify_set_layout, NULL);
    if (rt->unpatch_orig_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->unpatch_orig_set_layout, NULL);
    if (rt->latent_rgba_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->latent_rgba_set_layout, NULL);
    if (rt->patchify_shader) vkDestroyShaderModule(rt->device, rt->patchify_shader, NULL);
    if (rt->unpatch_orig_shader) vkDestroyShaderModule(rt->device, rt->unpatch_orig_shader, NULL);
    if (rt->latent_rgba_shader) vkDestroyShaderModule(rt->device, rt->latent_rgba_shader, NULL);
    destroy_host_buffer(rt, rt->latent_buffer, rt->latent_memory, rt->latent_mapped);
    destroy_host_buffer(rt, rt->patch_weight_buffer, rt->patch_weight_memory, rt->patch_weight_mapped);
    destroy_host_buffer(rt, rt->tokens_buffer, rt->tokens_memory, rt->tokens_mapped);
    destroy_host_buffer(rt, rt->unpatch_weight_buffer, rt->unpatch_weight_memory, rt->unpatch_weight_mapped);
    destroy_host_buffer(rt, rt->unpatch_bias_buffer, rt->unpatch_bias_memory, rt->unpatch_bias_mapped);
    destroy_host_buffer(rt, rt->latent_out_buffer, rt->latent_out_memory, rt->latent_out_mapped);
    if (rt->output_mapped) vkUnmapMemory(rt->device, rt->output_memory);
    if (rt->fence) vkDestroyFence(rt->device, rt->fence, NULL);
    if (rt->command_pool) vkDestroyCommandPool(rt->device, rt->command_pool, NULL);
    if (rt->descriptor_pool) vkDestroyDescriptorPool(rt->device, rt->descriptor_pool, NULL);
    if (rt->fill_pipeline) vkDestroyPipeline(rt->device, rt->fill_pipeline, NULL);
    if (rt->pipeline_layout) vkDestroyPipelineLayout(rt->device, rt->pipeline_layout, NULL);
    if (rt->descriptor_set_layout) vkDestroyDescriptorSetLayout(rt->device, rt->descriptor_set_layout, NULL);
    if (rt->fill_shader) vkDestroyShaderModule(rt->device, rt->fill_shader, NULL);
    if (rt->output_buffer) vkDestroyBuffer(rt->device, rt->output_buffer, NULL);
    if (rt->output_memory) vkFreeMemory(rt->device, rt->output_memory, NULL);
    if (rt->device) vkDestroyDevice(rt->device, NULL);
    if (rt->instance) vkDestroyInstance(rt->instance, NULL);
    free(rt->rgb_host);
    free(rt);
}
