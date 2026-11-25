if(NOT TARGET flagcv_nvidia OR NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Live Imgproc benchmarks require the NVIDIA backend and shared FlagCV")
endif()
find_package(OpenCV 5 REQUIRED COMPONENTS core imgproc cudawarping cudaimgproc cudaarithm cudafilters)
find_package(CUDAToolkit REQUIRED)

set(runners "${CMAKE_INSTALL_LIBEXECDIR}/flagcv")
file(RELATIVE_PATH runtime_from_runner "${CMAKE_INSTALL_FULL_LIBEXECDIR}/flagcv"
     "${CMAKE_INSTALL_FULL_LIBDIR}")
foreach(name IN ITEMS first geometry arithmetic_filters batch)
  add_executable(flagcv_imgproc_${name} ${name}.cpp)
  target_compile_features(flagcv_imgproc_${name} PRIVATE cxx_std_17)
  target_include_directories(flagcv_imgproc_${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/cpp"
                             "${PROJECT_BINARY_DIR}/tests")
  target_link_libraries(flagcv_imgproc_${name} PRIVATE flagcv opencv_core opencv_imgproc
                        opencv_cudawarping opencv_cudaimgproc opencv_cudaarithm opencv_cudafilters CUDA::cudart)
  set_target_properties(flagcv_imgproc_${name} PROPERTIES INSTALL_RPATH "$ORIGIN/${runtime_from_runner}")
  install(TARGETS flagcv_imgproc_${name} RUNTIME DESTINATION "${runners}")
endforeach()

foreach(name IN ITEMS first spatial batch)
  add_library(flagcv_imgproc_${name}_bridge SHARED ${name}_bridge.cpp)
  target_compile_features(flagcv_imgproc_${name}_bridge PRIVATE cxx_std_17)
  target_include_directories(flagcv_imgproc_${name}_bridge PRIVATE "${PROJECT_SOURCE_DIR}")
  target_link_libraries(flagcv_imgproc_${name}_bridge PRIVATE flagcv)
  set_target_properties(flagcv_imgproc_${name}_bridge PROPERTIES INSTALL_RPATH "$ORIGIN/${runtime_from_runner}")
  install(TARGETS flagcv_imgproc_${name}_bridge LIBRARY DESTINATION "${runners}")
endforeach()

execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/live-install.json"
  "{\"revision\":\"${revision}\",\"runners\":\"${runners}\",\"libdir\":\"${CMAKE_INSTALL_LIBDIR}\"}\n")
install(FILES first.py geometry.py arithmetic_filters.py batch.py cases.json
              "${CMAKE_CURRENT_BINARY_DIR}/live-install.json"
              "${PROJECT_SOURCE_DIR}/tests/test_projective.py"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/benchmark/imgproc")
install(FILES "${PROJECT_SOURCE_DIR}/tools/check_border_threshold.py"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/flagcv/tools")
