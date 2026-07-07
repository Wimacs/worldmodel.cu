cmake_minimum_required(VERSION 3.24)

project(worldmodel_cu LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

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

target_compile_options(worldmodel_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/CMakeLists.txt)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)
  add_subdirectory(3rd/raylib EXCLUDE_FROM_ALL)

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

  target_compile_options(worldmodel_raylib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()
