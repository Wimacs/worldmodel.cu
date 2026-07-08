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
    (void)weights;
    (void)layers_to_run;
    (void)steps_to_run;
    (void)seed;
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
    rt->pixel_count = (size_t)rt->width * (size_t)rt->height * (size_t)rt->frames;
    rt->rgb_bytes = rt->pixel_count * 3;
    rt->rgb_host = (unsigned char *)malloc(rt->rgb_bytes);
    if (!rt->rgb_host) goto fail;

    fprintf(stderr,
            "creating resident Vulkan runtime scaffold: RGB %dx%d frames=%d shader_dir=%s\n",
            rt->width, rt->height, rt->frames, WORLD_VULKAN_SHADER_DIR);
    fprintf(stderr,
            "warning: Vulkan backend currently runs the compute-pipeline scaffold; model kernels are being ported incrementally\n");

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
    vkCmdBindPipeline(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, rt->fill_pipeline);
    vkCmdBindDescriptorSets(rt->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            rt->pipeline_layout, 0, 1, &rt->descriptor_set, 0, NULL);
    vkCmdPushConstants(rt->command_buffer, rt->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(push), &push);
    vkCmdDispatch(rt->command_buffer,
            ((uint32_t)rt->width + 15u) / 16u,
            ((uint32_t)rt->height + 15u) / 16u,
            (uint32_t)rt->frames);
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
    fprintf(stderr, "Vulkan scaffold timing: total=%.3fms rgb_fps=%.3f\n",
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
    (void)latent;
    return world_vulkan_runtime_step_rgb(rt, control_input, rgb_out, width_out, height_out, frames_out, seconds_out);
}

static float probe_value(int i, float scale) {
    int v = (i * 37 + 17) % 29;
    return ((float)v - 14.0f) * scale;
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

void world_vulkan_runtime_destroy(WorldVulkanRuntime *rt) {
    if (!rt) return;
    if (rt->device) vkDeviceWaitIdle(rt->device);
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
