if(BUILD_TESTING)
  add_executable(worldmodel_config_tests
    tests/test_world_config.c
    ${PROJECT_SOURCE_DIR}/src/world_config.c
  )
  target_include_directories(worldmodel_config_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(worldmodel_config_tests PRIVATE ${WORLD_PLATFORM_LIBS})
  world_enable_large_objects(worldmodel_config_tests)
  add_test(NAME worldmodel_config_validation COMMAND worldmodel_config_tests)

  add_executable(worldmodel_safetensors_tests
    tests/test_safetensors.c
    ${PROJECT_SOURCE_DIR}/src/safetensors.c
  )
  target_include_directories(worldmodel_safetensors_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
  world_enable_large_objects(worldmodel_safetensors_tests)
  add_test(NAME worldmodel_safetensors_validation COMMAND worldmodel_safetensors_tests)
endif()

if(WORLD_BUILD_TOOLS)
  add_executable(worldmodel_cuda_gemm_probe profile/cuda_gemm.cu)
  world_enable_cutlass(worldmodel_cuda_gemm_probe)
  world_enable_cuda_options(worldmodel_cuda_gemm_probe)
  target_link_libraries(worldmodel_cuda_gemm_probe PRIVATE CUDA::cudart ${WORLD_PLATFORM_LIBS})

  add_executable(worldmodel_w8a8_probe profile/w8a8.cu)
  world_enable_cutlass(worldmodel_w8a8_probe)
  world_enable_cuda_options(worldmodel_w8a8_probe)
  target_link_libraries(worldmodel_w8a8_probe PRIVATE CUDA::cudart ${WORLD_PLATFORM_LIBS})
endif()
