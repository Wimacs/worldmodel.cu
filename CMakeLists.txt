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
find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_Interpreter_FOUND)
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

function(enable_optional_cudnn target_name)
  if(TARGET CUDNN::cudnn)
    target_link_libraries(${target_name} PRIVATE CUDNN::cudnn)
    target_compile_definitions(${target_name} PRIVATE WORLD_USE_CUDNN=1)
    set_target_properties(${target_name} PROPERTIES BUILD_RPATH "${CUDNN_LIBRARY_DIR}")
  endif()
endfunction()

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
  enable_optional_cudnn(worldmodel_raylib)

  target_compile_options(worldmodel_raylib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()
