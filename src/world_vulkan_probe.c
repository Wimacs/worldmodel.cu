#include "world_vulkan.h"

#include <stdio.h>

int world_vulkan_linear_f32_probe(void);

int main(void) {
    if (world_vulkan_linear_f32_probe()) {
        fprintf(stderr, "world_vulkan_linear_f32_probe: failed\n");
        return 1;
    }
    fprintf(stderr, "world_vulkan_linear_f32_probe: ok\n");
    return 0;
}
