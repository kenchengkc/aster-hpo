cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures (A100: 80)")
project(aster_cuda_test LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(ASTER_BUILD_PYTHON OFF CACHE BOOL "Build Python bindings" FORCE)
set(ASTER_BUILD_TESTS OFF CACHE BOOL "Build native tests" FORCE)
set(ASTER_BUILD_BENCHMARKS OFF CACHE BOOL "Build native benchmark" FORCE)
set(ASTER_ENABLE_SANITIZERS OFF CACHE BOOL "Build CPU sanitizers" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../native" native-core)

find_package(CUDAToolkit REQUIRED)
set(ASTER_CUDA_FLAGS "--fmad=false --ftz=false --prec-div=true --prec-sqrt=true; host: -ffp-contract=off -fno-fast-math")
configure_file(build_metadata.hpp.in generated/build_metadata.hpp @ONLY)
add_executable(aster_cuda hedge_accounting.cu)
target_include_directories(aster_cuda PRIVATE include "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(aster_cuda PRIVATE aster_accounting CUDA::cudart)
target_compile_options(aster_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off,-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>)

include(CTest)
if(BUILD_TESTING)
    add_executable(aster_cuda_comparison_test tests/test_comparison.cpp)
    target_include_directories(aster_cuda_comparison_test PRIVATE include)
    target_compile_options(aster_cuda_comparison_test PRIVATE -ffp-contract=off -fno-fast-math)
    add_test(NAME cuda_comparison COMMAND aster_cuda_comparison_test)
    add_test(NAME cuda_smoke COMMAND aster_cuda --mode smoke)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME cuda_cli COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli.py" "$<TARGET_FILE:aster_cuda>")
endif()
