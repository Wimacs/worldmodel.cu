cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()

project(worldmodel_c LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(WORLD_BUILD_TOOLS "Build CUDA profiling and diagnostic tools" ON)
set(WORLD_CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS checkout. If empty, common local checkouts are probed and then FetchContent is used.")

include(CTest)
include(FetchContent)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    )
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
    set(WORLD_CUTLASS_SOURCE_DIR "${candidate}")
    break()
  elseif(candidate AND EXISTS "${candidate}/cutlass/cutlass.h")
    set(WORLD_CUTLASS_INCLUDE_DIR "${candidate}")
    get_filename_component(WORLD_CUTLASS_SOURCE_DIR "${candidate}" DIRECTORY)
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
  set(WORLD_CUTLASS_SOURCE_DIR "${cutlass_SOURCE_DIR}")
endif()

set(WORLD_CUTLASS_FMHA_EXAMPLE_DIR "${WORLD_CUTLASS_SOURCE_DIR}/examples/41_fused_multi_head_attention")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTLASS include: ${WORLD_CUTLASS_INCLUDE_DIR}")

function(world_enable_cutlass target_name)
  target_include_directories(${target_name} PRIVATE ${WORLD_CUTLASS_INCLUDE_DIR})
  if(EXISTS "${WORLD_CUTLASS_FMHA_EXAMPLE_DIR}/kernel_forward.h")
    target_include_directories(${target_name} PRIVATE ${WORLD_CUTLASS_FMHA_EXAMPLE_DIR})
  endif()
endfunction()

add_library(worldmodel_cuda_backend STATIC
  src/world_config.c
  src/world_cuda.cu
  src/world_cuda_ops.cu
  src/world_cuda_vae.cu
  src/world_cuda_vae_ops.cu
)
target_include_directories(worldmodel_cuda_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
world_enable_cutlass(worldmodel_cuda_backend)
world_enable_cuda_options(worldmodel_cuda_backend)
target_link_libraries(worldmodel_cuda_backend PUBLIC CUDA::cudart ${WORLD_PLATFORM_LIBS})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/CMakeLists.txt")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUSTOMIZE_BUILD ON CACHE BOOL "" FORCE)
  set(USE_AUDIO OFF CACHE BOOL "" FORCE)
  set(SUPPORT_MODULE_RAUDIO OFF CACHE BOOL "" FORCE)
  set(SUPPORT_MODULE_RMODELS OFF CACHE BOOL "" FORCE)
  set(SUPPORT_FILEFORMAT_JPG ON CACHE BOOL "" FORCE)
  add_subdirectory(3rd/raylib EXCLUDE_FROM_ALL)

  add_executable(worldmodel_raylib
    src/world_raylib.c
    src/world_weights.c
    src/safetensors.c
  )
  target_include_directories(worldmodel_raylib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd/raylib/src
  )
  target_link_libraries(worldmodel_raylib PRIVATE
    worldmodel_cuda_backend
    raylib
    Threads::Threads
    ${WORLD_PLATFORM_LIBS}
  )
  set_property(TARGET worldmodel_raylib PROPERTY LINKER_LANGUAGE CUDA)
  world_enable_large_objects(worldmodel_raylib)
endif()

add_subdirectory(tools)
