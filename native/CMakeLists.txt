cmake_minimum_required(VERSION 3.20)
project(aster_native VERSION 0.1.0 LANGUAGES CXX)

option(ASTER_BUILD_PYTHON "Build the optional Python companion" OFF)
option(ASTER_BUILD_TESTS "Build native correctness tests" ON)
option(ASTER_BUILD_BENCHMARKS "Build standalone native benchmarks" ON)
option(ASTER_ENABLE_SANITIZERS "Enable address and undefined behavior checks" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(aster_accounting STATIC src/accounting.cpp)
target_include_directories(aster_accounting PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(aster_accounting PUBLIC cxx_std_20)
set_target_properties(aster_accounting PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(aster_accounting PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/fp:strict>")
  set(ASTER_FP_FLAGS "/fp:strict")
else()
  target_compile_options(aster_accounting PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>" "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>")
  set(ASTER_FP_FLAGS "-fno-fast-math -ffp-contract=off")
endif()
if(ASTER_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Sanitizer configuration requires Clang or GCC")
  endif()
  target_compile_options(aster_accounting PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>")
  target_link_options(aster_accounting PUBLIC -fsanitize=address,undefined)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" ASTER_BUILD_UPPER)
set(ASTER_RECORDED_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${ASTER_BUILD_UPPER}} ${ASTER_FP_FLAGS}")

if(ASTER_BUILD_TESTS)
  enable_testing()
  add_executable(accounting_tests tests/test_accounting.cpp)
  target_link_libraries(accounting_tests PRIVATE aster_accounting)
  add_test(NAME accounting COMMAND accounting_tests)
endif()

if(ASTER_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/accounting_bench.cpp")
  add_executable(accounting_bench benchmarks/accounting_bench.cpp)
  target_link_libraries(accounting_bench PRIVATE aster_accounting)
  target_compile_definitions(accounting_bench PRIVATE
    ASTER_BUILD_TYPE="${CMAKE_BUILD_TYPE}" ASTER_BUILD_FLAGS="${ASTER_RECORDED_FLAGS}")
endif()

if(ASTER_BUILD_PYTHON)
  find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC src/bindings.cpp)
  target_link_libraries(_core PRIVATE aster_accounting)
  target_compile_definitions(_core PRIVATE ASTER_NATIVE_VERSION="${PROJECT_VERSION}"
    ASTER_BUILD_TYPE="${CMAKE_BUILD_TYPE}" ASTER_BUILD_FLAGS="${ASTER_RECORDED_FLAGS}")
  install(TARGETS _core LIBRARY DESTINATION aster_hpo_native)
endif()
