cmake_minimum_required(VERSION 3.16)
project(qpu_llama_runtime LANGUAGES C)

include(CTest)
option(QPU_LLAMA_BUILD_HARDWARE_TESTS "Build tests that require a VideoCore VII render node" OFF)
option(QPU_LLAMA_BUILD_BENCHMARK "Build the native llama.cpp Q4_0 operator benchmark" OFF)
option(QPU_LLAMA_BUILD_GRAPH_PROFILER "Build the serialized llama.cpp graph profiler" OFF)
set(LLAMA_CPP_ROOT "" CACHE PATH "Path to the pinned llama.cpp source tree")

add_library(qpu_llama_runtime STATIC
    src/qpu_llama_q4_0.c
    src/qpu_llama_runtime.c
    src/qpu_llama_sha256.c)
target_compile_features(qpu_llama_runtime PUBLIC c_std_11)
target_include_directories(qpu_llama_runtime PUBLIC include PRIVATE generated)
target_link_libraries(qpu_llama_runtime PUBLIC pthread)
target_compile_options(qpu_llama_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(QPU_LLAMA_BUILD_BENCHMARK)
    if(NOT IS_DIRECTORY "${LLAMA_CPP_ROOT}")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to the pinned llama.cpp source tree")
    endif()
    find_library(GGML_CPU_LIBRARY NAMES ggml-cpu
        PATHS "${LLAMA_CPP_ROOT}/build/bin" NO_DEFAULT_PATH REQUIRED)
    add_executable(qpu_llama_q4_0_bench bench/qpu_llama_q4_0_bench.c)
    target_include_directories(qpu_llama_q4_0_bench PRIVATE
        generated
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_llama_q4_0_bench PRIVATE qpu_llama_runtime "${GGML_CPU_LIBRARY}" m)
    set_target_properties(qpu_llama_q4_0_bench PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_ROOT}/build/bin")
    target_compile_options(qpu_llama_q4_0_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(QPU_LLAMA_BUILD_GRAPH_PROFILER)
    enable_language(CXX)
    if(NOT IS_DIRECTORY "${LLAMA_CPP_ROOT}")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to the pinned llama.cpp source tree")
    endif()
    find_library(LLAMA_LIBRARY NAMES llama
        PATHS "${LLAMA_CPP_ROOT}/build/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY NAMES ggml
        PATHS "${LLAMA_CPP_ROOT}/build/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIBRARY NAMES ggml-base
        PATHS "${LLAMA_CPP_ROOT}/build/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIBRARY_FOR_PROFILE NAMES ggml-cpu
        PATHS "${LLAMA_CPP_ROOT}/build/bin" NO_DEFAULT_PATH REQUIRED)
    add_executable(qpu_llama_graph_profile tools/qpu_llama_graph_profile.cpp)
    target_include_directories(qpu_llama_graph_profile PRIVATE
        "${LLAMA_CPP_ROOT}/include"
        "${LLAMA_CPP_ROOT}/src"
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/vendor")
    target_link_libraries(qpu_llama_graph_profile PRIVATE
        "${LLAMA_LIBRARY}"
        "${GGML_LIBRARY}"
        "${GGML_BASE_LIBRARY}"
        "${GGML_CPU_LIBRARY_FOR_PROFILE}")
    set_target_properties(qpu_llama_graph_profile PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_ROOT}/build/bin")
    target_compile_features(qpu_llama_graph_profile PRIVATE cxx_std_17)
    target_compile_options(qpu_llama_graph_profile PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME qpu_llama_graph_profile_help COMMAND qpu_llama_graph_profile --help)
    endif()
endif()

if(BUILD_TESTING)
    add_executable(test_qpu_llama_runtime tests/test_qpu_llama_runtime.c)
    target_link_libraries(test_qpu_llama_runtime PRIVATE qpu_llama_runtime)
    add_test(NAME qpu_llama_runtime_unit COMMAND test_qpu_llama_runtime)

    add_executable(qpu_llama_runtime_smoke tests/qpu_llama_runtime_smoke.c)
    target_link_libraries(qpu_llama_runtime_smoke PRIVATE qpu_llama_runtime)

    if(QPU_LLAMA_BUILD_HARDWARE_TESTS)
        add_executable(qpu_llama_w8a8_smoke tests/qpu_llama_w8a8_smoke.c)
        target_include_directories(qpu_llama_w8a8_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_w8a8_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_w8a8_hardware COMMAND qpu_llama_w8a8_smoke)
        set_tests_properties(qpu_llama_w8a8_hardware PROPERTIES LABELS hardware)

        add_executable(qpu_llama_q4_0_smoke tests/qpu_llama_q4_0_smoke.c)
        target_include_directories(qpu_llama_q4_0_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_q4_0_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_q4_0_hardware COMMAND qpu_llama_q4_0_smoke)
        set_tests_properties(qpu_llama_q4_0_hardware PROPERTIES LABELS hardware)

        add_executable(qpu_llama_q4_0_m4_smoke tests/qpu_llama_q4_0_m4_smoke.c)
        target_include_directories(qpu_llama_q4_0_m4_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_q4_0_m4_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_q4_0_m4_hardware COMMAND qpu_llama_q4_0_m4_smoke)
        set_tests_properties(qpu_llama_q4_0_m4_hardware PROPERTIES LABELS hardware)
    endif()
endif()
