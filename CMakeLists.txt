cmake_minimum_required(VERSION 3.24)

project(worldmodel_cu LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(WORLD_ENABLE_CUDA "Build CUDA backend targets" ON)
option(WORLD_ENABLE_VULKAN "Build Vulkan backend targets" OFF)

find_package(Threads REQUIRED)

if(WORLD_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

if(WORLD_ENABLE_CUDA AND Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pathlib; import nvidia.cudnn as c; p=pathlib.Path(c.__file__).resolve().parent; print(p/'include'); print(p/'lib')"
    RESULT_VARIABLE CUDNN_PY_RESULT
    OUTPUT_VARIABLE CUDNN_PY_OUTPUT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(CUDNN_PY_RESULT EQUAL 0)
    string(REPLACE "\n" ";" CUDNN_PY_HINTS "${CUDNN_PY_OUTPUT}")
    list(GET CUDNN_PY_HINTS 0 CUDNN_INCLUDE_HINT)
    list(GET CUDNN_PY_HINTS 1 CUDNN_LIBRARY_HINT)
  endif()
endif()

if(WORLD_ENABLE_CUDA)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_INCLUDE_HINT})
  find_library(CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 HINTS ${CUDNN_LIBRARY_HINT})
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    add_library(CUDNN::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn PROPERTIES
      IMPORTED_LOCATION "${CUDNN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
    get_filename_component(CUDNN_LIBRARY_DIR "${CUDNN_LIBRARY}" DIRECTORY)
    message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
  else()
    message(STATUS "cuDNN not found; VAE decoder will use built-in CUDA conv kernels")
  endif()
endif()

function(enable_optional_cudnn target_name)
  if(TARGET CUDNN::cudnn)
    target_link_libraries(${target_name} PRIVATE CUDNN::cudnn)
    target_compile_definitions(${target_name} PRIVATE WORLD_USE_CUDNN=1)
    set_target_properties(${target_name} PROPERTIES BUILD_RPATH "${CUDNN_LIBRARY_DIR}")
  endif()
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

  target_link_libraries(worldmodel_cuda PRIVATE
    CUDA::cudart
    CUDA::cublas
  )
  enable_optional_cudnn(worldmodel_cuda)

  target_compile_options(worldmodel_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

if(WORLD_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLC_EXECUTABLE glslc REQUIRED)

  set(WORLD_VULKAN_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/vulkan)
  set(WORLD_VULKAN_SHADER_SOURCES
    add_bias_silu_f32.comp
    add_channel_silu_f32.comp
    add_f32.comp
    ada_rms_norm_f32.comp
    cache_frame_indices.comp
    fill_rgba.comp
    gated_residual_add_f32.comp
    indexed_attention_d64_flash24w_f32.comp
    indexed_attention_d64_flash24w_k96_f32.comp
    indexed_attention_d64_flash32w_f32.comp
    indexed_attention_d64_flash96_f32.comp
    indexed_attention_d64_flash_kvf16_f32.comp
    indexed_attention_d64_flash_f32.comp
    indexed_attention_d64_splitk32_part_f32.comp
    indexed_attention_d64_splitk64_part_f32.comp
    indexed_attention_d64_splitk96_part_f32.comp
    indexed_attention_d64_splitk64_reduce_f32.comp
    indexed_attention_d64_warp_f32.comp
    indexed_attention_f32.comp
    kv_cache_mask.comp
    kv_cache_upsert_copy_f16.comp
    kv_cache_upsert_copy_f32.comp
    lerp_inplace_f32.comp
    linear_f16_coopmat.comp
    linear_f16x_wf16_gated_residual.comp
    linear_f16x_wf16_gated_residual_n32.comp
    linear_f16x_wf16_coopmat.comp
    linear_f32_coopmat.comp
    linear_f32_wf16_silu_f16_n32.comp
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
      COMMAND ${GLSLC_EXECUTABLE} -O --target-env=vulkan1.2 ${shader_src} -o ${shader_spv}
      DEPENDS ${shader_src}
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
    WORLD_VULKAN_SHADER_DIR="${WORLD_VULKAN_SHADER_OUTPUT_DIR}"
  )
  target_link_libraries(worldmodel_vulkan_probe PRIVATE
    Vulkan::Vulkan
    m
  )
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/CMakeLists.txt)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)
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

    target_link_libraries(worldmodel_raylib PRIVATE
      raylib
      CUDA::cudart
      CUDA::cublas
      Threads::Threads
    )
    enable_optional_cudnn(worldmodel_raylib)

    target_compile_options(worldmodel_raylib PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
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
      WORLD_VULKAN_SHADER_DIR="${WORLD_VULKAN_SHADER_OUTPUT_DIR}"
    )

    target_link_libraries(worldmodel_raylib_vulkan PRIVATE
      raylib
      Vulkan::Vulkan
      Threads::Threads
      m
    )
  endif()
endif()
