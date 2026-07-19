cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()

project(worldmodel_w8a8_probe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(WORLD_CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS checkout or include directory")
get_filename_component(WORLD_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(_cutlass_candidates
  "${WORLD_CUTLASS_DIR}"
  "$ENV{WORLD_CUTLASS_DIR}"
  "$ENV{CUTLASS_DIR}"
  "${WORLD_REPO_ROOT}/3rd/cutlass"
  "${WORLD_REPO_ROOT}/build-win-cudnn/_deps/cutlass-src"
  "${WORLD_REPO_ROOT}/build-win-cuda/_deps/cutlass-src"
)

set(WORLD_CUTLASS_INCLUDE_DIR "")
foreach(candidate IN LISTS _cutlass_candidates)
  if(candidate AND EXISTS "${candidate}/include/cutlass/cutlass.h")
    set(WORLD_CUTLASS_INCLUDE_DIR "${candidate}/include")
    break()
  elseif(candidate AND EXISTS "${candidate}/cutlass/cutlass.h")
    set(WORLD_CUTLASS_INCLUDE_DIR "${candidate}")
    break()
  endif()
endforeach()

if(NOT WORLD_CUTLASS_INCLUDE_DIR)
  include(FetchContent)
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

message(STATUS "W8A8 probe CUTLASS include: ${WORLD_CUTLASS_INCLUDE_DIR}")
message(STATUS "W8A8 probe CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_executable(worldmodel_w8a8_probe
  worldmodel_w8a8_probe.cu
)

target_include_directories(worldmodel_w8a8_probe PRIVATE
  "${WORLD_CUTLASS_INCLUDE_DIR}"
)

target_link_libraries(worldmodel_w8a8_probe PRIVATE CUDA::cudart)

target_compile_options(worldmodel_w8a8_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(MSVC)
  target_compile_definitions(worldmodel_w8a8_probe PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
  )
  target_compile_options(worldmodel_w8a8_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
  )
endif()
