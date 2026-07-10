cmake_minimum_required(VERSION 3.24)

project(worldmodel_cu LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(WORLD_ENABLE_CUDA "Build CUDA backend targets" ON)
option(WORLD_ENABLE_VULKAN "Build Vulkan backend targets" OFF)
set(WORLD_CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS checkout. If empty, common local checkouts are probed and then FetchContent is used.")

if(WORLD_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()

find_package(Threads REQUIRED)

set(WORLD_PLATFORM_LIBS)
if(UNIX)
  list(APPEND WORLD_PLATFORM_LIBS m)
endif()

function(world_enable_large_objects target_name)
  if(WIN32)
    target_compile_definitions(${target_name} PRIVATE
      WIN32_LEAN_AND_MEAN
      NOMINMAX
      _CRT_SECURE_NO_WARNINGS
    )
  endif()
  if(MSVC)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:C>:/bigobj>
    )
    if(CMAKE_CUDA_COMPILER)
      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
      )
    endif()
  endif()
endfunction()

function(world_enable_cuda_options target_name)
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
  world_enable_large_objects(${target_name})
endfunction()

if(WORLD_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  find_package(CUDAToolkit REQUIRED)
endif()

if(WORLD_ENABLE_CUDA)
  include(FetchContent)
  set(WORLD_CUTLASS_HINTS
    "${WORLD_CUTLASS_DIR}"
    "$ENV{CUTLASS_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd/cutlass"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../flux2/third_party/cutlass"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../flux.cu/third_party/cutlass"
  )
  set(WORLD_CUTLASS_INCLUDE_DIR "")
  foreach(candidate IN LISTS WORLD_CUTLASS_HINTS)
    if(candidate AND EXISTS "${candidate}/include/cutlass/cutlass.h")
      set(WORLD_CUTLASS_INCLUDE_DIR "${candidate}/include")
      break()
    endif()
  endforeach()
  if(NOT WORLD_CUTLASS_INCLUDE_DIR)
    FetchContent_Declare(
      cutlass
      GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG v3.5.1
      GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(cutlass)
    if(NOT cutlass_POPULATED)
      FetchContent_Populate(cutlass)
    endif()
    set(WORLD_CUTLASS_INCLUDE_DIR "${cutlass_SOURCE_DIR}/include")
  endif()
  message(STATUS "CUTLASS include: ${WORLD_CUTLASS_INCLUDE_DIR}")
endif()

function(world_enable_cutlass target_name)
  target_include_directories(${target_name} PRIVATE ${WORLD_CUTLASS_INCLUDE_DIR})
endfunction()

if(WORLD_ENABLE_CUDA)
  add_executable(worldmodel_cuda
    src/world_cli.c
    src/world_config.c
    src/safetensors.c
    src/world_cuda.cu
  )

  target_include_directories(worldmodel_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  world_enable_cutlass(worldmodel_cuda)

  target_link_libraries(worldmodel_cuda PRIVATE
    CUDA::cudart
    ${WORLD_PLATFORM_LIBS}
  )
  world_enable_cuda_options(worldmodel_cuda)

  add_executable(worldmodel_cuda_gemm_probe
    src/world_cuda_gemm_probe.cu
  )
  world_enable_cutlass(worldmodel_cuda_gemm_probe)
  target_link_libraries(worldmodel_cuda_gemm_probe PRIVATE
    CUDA::cudart
    ${WORLD_PLATFORM_LIBS}
  )
  world_enable_cuda_options(worldmodel_cuda_gemm_probe)
endif()

if(WORLD_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLC_EXECUTABLE
    NAMES glslc glslc.exe
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32"
    REQUIRED
  )

  set(WORLD_VULKAN_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/vulkan)
  file(TO_CMAKE_PATH "${WORLD_VULKAN_SHADER_OUTPUT_DIR}" WORLD_VULKAN_SHADER_DEFINE_DIR)
  set(WORLD_VULKAN_SHADER_INCLUDE_DEPS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/gemm_wf16_coopmat_template.comp
  )
  set(WORLD_VULKAN_SHADER_SOURCES
    add_bias_silu_f32.comp
    add_channel_silu_f32.comp
    add_f32.comp
    ada_rms_norm_f32.comp
    cache_frame_indices.comp
    fill_rgba.comp
    gated_residual_add_f32.comp
    cast_f32_to_f16.comp
    indexed_attention_d64_flash24w_f32.comp
    indexed_attention_d64_flash24w_k96_f32.comp
    indexed_attention_d64_flash32w_f32.comp
    indexed_attention_d64_flash96_f32.comp
    indexed_attention_d64_flash_kvf16_f32.comp
    indexed_attention_d64_flash_f32.comp
    indexed_attention_d64_splitk32_part_f32.comp
    indexed_attention_d64_splitk64_part_f32.comp
    indexed_attention_d64_splitk64_w32_part_f32.comp
    indexed_attention_d64_splitk64_part_kvf16_f32.comp
    indexed_attention_d64_splitk96_part_f32.comp
    indexed_attention_d64_splitk64_reduce_f32.comp
    indexed_attention_d64_warp_f32.comp
    indexed_attention_f32.comp
    kv_cache_mask.comp
    kv_cache_upsert_copy_f16.comp
    kv_cache_upsert_copy_f32.comp
    lerp_inplace_f32.comp
    gemm_f16x_wf16_n8.comp
    gemm_f16x_wf16_n16.comp
    gemm_f16x_wf16_n32.comp
    gemm_f32x_wf16_n8_stageb.comp
    gemm_f32x_wf16_n16_stageb.comp
    gemm_f32x_wf16_n32.comp
    gemm_f32x_wf16_silu_f16_n8_stageb.comp
    gemm_f32x_wf16_silu_f16_n16_stageb.comp
    gemm_f32x_wf16_silu_f16_n32.comp
    gemm_f16x_wf16_silu_f16_n8.comp
    gemm_f16x_wf16_silu_f16_n16.comp
    gemm_f16x_wf16_silu_f16_n32.comp
    gemm_f32x_wf16_gated_n8_stageb.comp
    gemm_f32x_wf16_gated_n16_stageb.comp
    gemm_f32x_wf16_gated_n32.comp
    gemm_f16x_wf16_gated_n8.comp
    gemm_f16x_wf16_gated_n16.comp
    gemm_f16x_wf16_gated_n32.comp
    linear_f16_coopmat.comp
    linear_f16x_wf16_gated_residual.comp
    linear_f16x_wf16_gated_residual_n16.comp
    linear_f16x_wf16_gated_residual_n32.comp
    linear_f16x_wf16_gated_residual_wg4.comp
    linear_f16x_wf16_coopmat.comp
    linear_f32_coopmat.comp
    linear_f32_wf16_silu_f16_n32.comp
    linear_f16x_wf16_silu_f16_n32.comp
    linear_f32_wf16_coopmat.comp
    linear_f32_wf16_coopmat_n16.comp
    linear_f32_wf16_coopmat_n32.comp
    linear_f32_wf16_gated_residual_n32.comp
    linear_f32.comp
    linear_f32_tile64.comp
    latent_to_rgba.comp
    masked_attention_f32.comp
    ortho_rope_f32.comp
    out_norm_silu_f32.comp
    patchify_f32.comp
    qkv_rms_rope_f32.comp
    rms_norm_f32.comp
    silu_f32_to_f16.comp
    silu_f32.comp
    taehv_add_relu_f32.comp
    taehv_concat_past_nchw_f32.comp
    taehv_conv2d_nchw_1x1_c4_f32.comp
    taehv_conv2d_nchw_c4_f32.comp
    taehv_conv2d_nchw_f32.comp
    taehv_conv_out_rgba_f32.comp
    taehv_pixel_shuffle_rgba_f32.comp
    taehv_relu_f32.comp
    taehv_repeat_clamp_f32.comp
    taehv_tgrow_reshape_nchw_f32.comp
    taehv_upsample2_nchw_f32.comp
    latent_update_f32.comp
    unpatchify_f32.comp
    unpatchify_orig_f32.comp
  )
  set(WORLD_VULKAN_SHADER_SPV)
  foreach(shader_name IN LISTS WORLD_VULKAN_SHADER_SOURCES)
    set(shader_src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/${shader_name})
    set(shader_spv ${WORLD_VULKAN_SHADER_OUTPUT_DIR}/${shader_name}.spv)
    add_custom_command(
      OUTPUT ${shader_spv}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${WORLD_VULKAN_SHADER_OUTPUT_DIR}
      COMMAND ${GLSLC_EXECUTABLE} -O --target-env=vulkan1.2 -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan ${shader_src} -o ${shader_spv}
      DEPENDS ${shader_src} ${WORLD_VULKAN_SHADER_INCLUDE_DEPS}
      VERBATIM
    )
    list(APPEND WORLD_VULKAN_SHADER_SPV ${shader_spv})
  endforeach()
  add_custom_target(worldmodel_vulkan_shaders DEPENDS ${WORLD_VULKAN_SHADER_SPV})

  add_executable(worldmodel_vulkan_probe
    src/world_vulkan_probe.c
    src/world_config.c
    src/world_vulkan.c
  )
  add_dependencies(worldmodel_vulkan_probe worldmodel_vulkan_shaders)
  target_include_directories(worldmodel_vulkan_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_compile_definitions(worldmodel_vulkan_probe PRIVATE
    WORLD_VULKAN_SHADER_DIR="${WORLD_VULKAN_SHADER_DEFINE_DIR}"
  )
  world_enable_large_objects(worldmodel_vulkan_probe)
  target_link_libraries(worldmodel_vulkan_probe PRIVATE
    Vulkan::Vulkan
    ${WORLD_PLATFORM_LIBS}
  )

  add_executable(worldmodel_vulkan_gemm_autotune
    src/world_vulkan_gemm_autotune.c
    src/world_config.c
    src/world_vulkan.c
  )
  add_dependencies(worldmodel_vulkan_gemm_autotune worldmodel_vulkan_shaders)
  target_include_directories(worldmodel_vulkan_gemm_autotune PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_compile_definitions(worldmodel_vulkan_gemm_autotune PRIVATE
    WORLD_VULKAN_SHADER_DIR="${WORLD_VULKAN_SHADER_DEFINE_DIR}"
  )
  world_enable_large_objects(worldmodel_vulkan_gemm_autotune)
  target_link_libraries(worldmodel_vulkan_gemm_autotune PRIVATE
    Vulkan::Vulkan
    ${WORLD_PLATFORM_LIBS}
  )
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/CMakeLists.txt)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUSTOMIZE_BUILD ON CACHE BOOL "" FORCE)
  set(USE_AUDIO OFF CACHE BOOL "" FORCE)
  set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)
  set(SUPPORT_MODULE_RMODELS OFF CACHE BOOL "" FORCE)
  add_subdirectory(3rd/raylib EXCLUDE_FROM_ALL)

  if(WORLD_ENABLE_CUDA)
    add_executable(worldmodel_raylib
      src/world_raylib.c
      src/world_config.c
      src/safetensors.c
      src/world_cuda.cu
    )

    target_include_directories(worldmodel_raylib PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/src
    )
    world_enable_cutlass(worldmodel_raylib)

    target_link_libraries(worldmodel_raylib PRIVATE
      raylib
      CUDA::cudart
      Threads::Threads
      ${WORLD_PLATFORM_LIBS}
    )
    world_enable_cuda_options(worldmodel_raylib)
  endif()

  if(WORLD_ENABLE_VULKAN)
    add_executable(worldmodel_raylib_vulkan
      src/world_raylib.c
      src/world_config.c
      src/safetensors.c
      src/world_vulkan.c
    )

    add_dependencies(worldmodel_raylib_vulkan worldmodel_vulkan_shaders)

    target_include_directories(worldmodel_raylib_vulkan PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/src
    )

    target_compile_definitions(worldmodel_raylib_vulkan PRIVATE
      WORLD_BACKEND_VULKAN=1
      WORLD_VULKAN_SHADER_DIR="${WORLD_VULKAN_SHADER_DEFINE_DIR}"
    )
    world_enable_large_objects(worldmodel_raylib_vulkan)

    target_link_libraries(worldmodel_raylib_vulkan PRIVATE
      raylib
      Vulkan::Vulkan
      Threads::Threads
      ${WORLD_PLATFORM_LIBS}
    )
  endif()
endif()
