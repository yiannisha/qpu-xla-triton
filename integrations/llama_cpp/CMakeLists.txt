cmake_minimum_required(VERSION 3.16)
project(qpu_llama_runtime LANGUAGES C)

include(CTest)
option(QPU_LLAMA_BUILD_HARDWARE_TESTS "Build tests that require a VideoCore VII render node" OFF)
option(QPU_LLAMA_BUILD_BENCHMARK "Build the native llama.cpp Q4_0 operator benchmark" OFF)
option(QPU_LLAMA_BUILD_GRAPH_PROFILER "Build the serialized llama.cpp graph profiler" OFF)
option(QPU_LLAMA_BUILD_GGML_BACKEND "Build the out-of-tree GGML QPU backend plugin" OFF)
set(LLAMA_CPP_ROOT "" CACHE PATH "Path to the pinned llama.cpp source tree")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "Path to the pinned llama.cpp build tree")

if(LLAMA_CPP_ROOT AND NOT LLAMA_CPP_BUILD_DIR)
    set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_ROOT}/build" CACHE PATH
        "Path to the pinned llama.cpp build tree" FORCE)
endif()

add_library(qpu_llama_runtime STATIC
    src/qpu_llama_q4_0.c
    src/qpu_llama_q4_k.c
    src/qpu_llama_q6_k.c
    src/qpu_llama_q8_0.c
    src/qpu_llama_runtime.c
    src/qpu_llama_sha256.c)
target_compile_features(qpu_llama_runtime PUBLIC c_std_11)
target_include_directories(qpu_llama_runtime PUBLIC include PRIVATE generated)
target_link_libraries(qpu_llama_runtime PUBLIC pthread)
target_compile_options(qpu_llama_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(qpu_llama_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(QPU_LLAMA_BUILD_GGML_BACKEND)
    enable_language(CXX)
    if(NOT IS_DIRECTORY "${LLAMA_CPP_ROOT}")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to the pinned llama.cpp source tree")
    endif()
    find_library(GGML_BASE_LIBRARY_FOR_QPU_BACKEND NAMES ggml-base
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIBRARY_FOR_QPU_BACKEND NAMES ggml-cpu
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY_FOR_QPU_BACKEND NAMES ggml
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(LLAMA_LIBRARY_FOR_QPU_BACKEND NAMES llama
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    add_library(ggml-qpu MODULE backend/ggml-qpu.cpp)
    target_include_directories(ggml-qpu PRIVATE
        include
        generated
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(ggml-qpu PRIVATE
        qpu_llama_runtime
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(ggml-qpu PRIVATE cxx_std_17)
    target_compile_definitions(ggml-qpu PRIVATE
        GGML_BACKEND_SHARED
        GGML_BACKEND_BUILD
        GGML_BACKEND_DL)
    target_compile_options(ggml-qpu PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(ggml-qpu PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        LIBRARY_OUTPUT_NAME "ggml-qpu")

    add_library(ggml-qpu-inline SHARED
        backend/ggml-qpu-inline.cpp
        backend/ggml-qpu-ffn-island.cpp
        backend/ggml-qpu-up-inline.cpp)
    target_include_directories(ggml-qpu-inline PRIVATE
        include
        generated
        "${LLAMA_CPP_ROOT}/ggml/include")
    target_link_libraries(ggml-qpu-inline PRIVATE
        qpu_llama_runtime
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(ggml-qpu-inline PRIVATE cxx_std_17)
    target_compile_options(ggml-qpu-inline PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(ggml-qpu-inline PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin"
        LIBRARY_OUTPUT_NAME "ggml-qpu-inline")

    add_executable(qpu_ggml_inline_smoke tests/qpu_ggml_inline_smoke.cpp)
    target_include_directories(qpu_ggml_inline_smoke PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include")
    target_link_libraries(qpu_ggml_inline_smoke PRIVATE
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}"
        "${CMAKE_DL_LIBS}")
    target_compile_features(qpu_ggml_inline_smoke PRIVATE cxx_std_17)
    target_compile_options(qpu_ggml_inline_smoke PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qpu_ggml_inline_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    if(BUILD_TESTING AND QPU_LLAMA_BUILD_HARDWARE_TESTS)
        add_test(NAME qpu_ggml_inline_hardware
            COMMAND ${CMAKE_COMMAND} -E env
                LD_PRELOAD=$<TARGET_FILE:ggml-qpu-inline>
                $<TARGET_FILE:qpu_ggml_inline_smoke>)
        set_tests_properties(qpu_ggml_inline_hardware PROPERTIES LABELS hardware)
        add_test(NAME qpu_ggml_inline_hybrid_hardware
            COMMAND ${CMAKE_COMMAND} -E env
                LD_PRELOAD=$<TARGET_FILE:ggml-qpu-inline>
                GGML_QPU_TELEMETRY=0
                $<TARGET_FILE:qpu_ggml_inline_smoke>
                --rows 17
                --columns 12288
                --qpu-rows 1
                --cpu-threads 4
                --warmups 2
                --samples 3)
        set_tests_properties(qpu_ggml_inline_hybrid_hardware PROPERTIES LABELS hardware)
    endif()

    add_executable(qpu_ffn_island_smoke tests/qpu_ffn_island_smoke.cpp)
    target_include_directories(qpu_ffn_island_smoke PRIVATE include)
    target_link_libraries(qpu_ffn_island_smoke PRIVATE ggml-qpu-inline)
    target_compile_features(qpu_ffn_island_smoke PRIVATE cxx_std_17)
    target_compile_options(qpu_ffn_island_smoke PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING AND QPU_LLAMA_BUILD_HARDWARE_TESTS)
        add_test(NAME qpu_ffn_island_hardware COMMAND qpu_ffn_island_smoke)
        set_tests_properties(qpu_ffn_island_hardware PROPERTIES LABELS hardware)
    endif()

    add_executable(qpu_ffn_island_graph_smoke
        tests/qpu_ffn_island_graph_smoke.cpp)
    target_include_directories(qpu_ffn_island_graph_smoke PRIVATE
        include
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_ffn_island_graph_smoke PRIVATE
        ggml-qpu-inline
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(qpu_ffn_island_graph_smoke PRIVATE cxx_std_17)
    target_compile_options(qpu_ffn_island_graph_smoke PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qpu_ffn_island_graph_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    if(BUILD_TESTING AND QPU_LLAMA_BUILD_HARDWARE_TESTS)
        add_test(NAME qpu_ffn_island_graph_hardware
            COMMAND qpu_ffn_island_graph_smoke)
        set_tests_properties(qpu_ffn_island_graph_hardware PROPERTIES LABELS hardware)
    endif()

    add_executable(qpu_repack_stride_smoke tests/qpu_repack_stride_smoke.cpp)
    target_include_directories(qpu_repack_stride_smoke PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_repack_stride_smoke PRIVATE
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(qpu_repack_stride_smoke PRIVATE cxx_std_17)
    target_compile_options(qpu_repack_stride_smoke PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qpu_repack_stride_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    if(BUILD_TESTING)
        add_test(NAME qpu_repack_stride COMMAND qpu_repack_stride_smoke)
    endif()

    add_executable(qpu_ffn_island_model_smoke
        tests/qpu_ffn_island_model_smoke.cpp)
    target_include_directories(qpu_ffn_island_model_smoke PRIVATE
        include
        "${LLAMA_CPP_ROOT}/include"
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/vendor")
    target_link_libraries(qpu_ffn_island_model_smoke PRIVATE
        ggml-qpu-inline
        "${LLAMA_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(qpu_ffn_island_model_smoke PRIVATE cxx_std_17)
    target_compile_options(qpu_ffn_island_model_smoke PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qpu_ffn_island_model_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")

    add_executable(qpu_ggml_backend_smoke tests/qpu_ggml_backend_smoke.cpp)
    target_include_directories(qpu_ggml_backend_smoke PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_ggml_backend_smoke PRIVATE
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(qpu_ggml_backend_smoke PRIVATE cxx_std_17)
    target_compile_options(qpu_ggml_backend_smoke PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qpu_ggml_backend_smoke PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    if(BUILD_TESTING AND QPU_LLAMA_BUILD_HARDWARE_TESTS)
        add_test(NAME qpu_ggml_backend_hardware
            COMMAND qpu_ggml_backend_smoke $<TARGET_FILE:ggml-qpu>)
        set_tests_properties(qpu_ggml_backend_hardware PROPERTIES LABELS hardware)
    endif()

    add_executable(qpu_ggml_geglu_bench bench/qpu_ggml_geglu_bench.cpp)
    target_include_directories(qpu_ggml_geglu_bench PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include")
    target_link_libraries(qpu_ggml_geglu_bench PRIVATE
        "${GGML_CPU_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_LIBRARY_FOR_QPU_BACKEND}"
        "${GGML_BASE_LIBRARY_FOR_QPU_BACKEND}")
    target_compile_features(qpu_ggml_geglu_bench PRIVATE cxx_std_17)
    target_compile_options(qpu_ggml_geglu_bench PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qpu_ggml_geglu_bench PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    if(BUILD_TESTING)
        add_test(NAME qpu_ggml_geglu_bench_help
            COMMAND qpu_ggml_geglu_bench --help)
    endif()
    if(BUILD_TESTING AND QPU_LLAMA_BUILD_HARDWARE_TESTS)
        add_test(NAME qpu_ggml_geglu_hardware
            COMMAND qpu_ggml_geglu_bench
                --plugin $<TARGET_FILE:ggml-qpu>
                --rows 16
                --columns 6144
                --cpu-threads 1
                --warmups 1
                --samples 1)
        set_tests_properties(qpu_ggml_geglu_hardware PROPERTIES LABELS hardware)
    endif()
endif()

if(QPU_LLAMA_BUILD_BENCHMARK)
    enable_language(CXX)
    if(NOT IS_DIRECTORY "${LLAMA_CPP_ROOT}")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to the pinned llama.cpp source tree")
    endif()
    find_library(GGML_CPU_LIBRARY NAMES ggml-cpu
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY_FOR_BENCH NAMES ggml
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIBRARY_FOR_BENCH NAMES ggml-base
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BLAS_LIBRARY NAMES ggml-blas
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH)
    add_executable(qpu_llama_q4_0_bench bench/qpu_llama_q4_0_bench.c)
    target_include_directories(qpu_llama_q4_0_bench PRIVATE
        generated
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_llama_q4_0_bench PRIVATE qpu_llama_runtime "${GGML_CPU_LIBRARY}" m)
    set_target_properties(qpu_llama_q4_0_bench PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    target_compile_options(qpu_llama_q4_0_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(qpu_llama_cpu_repack_bench bench/qpu_llama_cpu_repack_bench.cpp)
    target_include_directories(qpu_llama_cpu_repack_bench PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_llama_cpu_repack_bench PRIVATE
        "${GGML_CPU_LIBRARY}"
        "${GGML_LIBRARY_FOR_BENCH}"
        "${GGML_BASE_LIBRARY_FOR_BENCH}")
    set_target_properties(qpu_llama_cpu_repack_bench PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    target_compile_features(qpu_llama_cpu_repack_bench PRIVATE cxx_std_17)
    target_compile_options(qpu_llama_cpu_repack_bench PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME qpu_llama_cpu_repack_bench_help COMMAND qpu_llama_cpu_repack_bench --help)
    endif()

    add_executable(qpu_llama_cpu_ffn_bench bench/qpu_llama_cpu_ffn_bench.cpp)
    target_include_directories(qpu_llama_cpu_ffn_bench PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_llama_cpu_ffn_bench PRIVATE
        "${GGML_CPU_LIBRARY}"
        "${GGML_LIBRARY_FOR_BENCH}"
        "${GGML_BASE_LIBRARY_FOR_BENCH}")
    if(GGML_BLAS_LIBRARY)
        target_link_libraries(qpu_llama_cpu_ffn_bench PRIVATE "${GGML_BLAS_LIBRARY}")
        target_compile_definitions(qpu_llama_cpu_ffn_bench PRIVATE QPU_LLAMA_HAVE_BLAS=1)
    endif()
    set_target_properties(qpu_llama_cpu_ffn_bench PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    target_compile_features(qpu_llama_cpu_ffn_bench PRIVATE cxx_std_17)
    target_compile_options(qpu_llama_cpu_ffn_bench PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME qpu_llama_cpu_ffn_bench_help COMMAND qpu_llama_cpu_ffn_bench --help)
    endif()

    add_executable(qpu_llama_cpu_flash_attn_bench bench/qpu_llama_cpu_flash_attn_bench.cpp)
    target_include_directories(qpu_llama_cpu_flash_attn_bench PRIVATE
        "${LLAMA_CPP_ROOT}/ggml/include"
        "${LLAMA_CPP_ROOT}/ggml/src"
        "${LLAMA_CPP_ROOT}/ggml/src/ggml-cpu")
    target_link_libraries(qpu_llama_cpu_flash_attn_bench PRIVATE
        "${GGML_CPU_LIBRARY}"
        "${GGML_LIBRARY_FOR_BENCH}"
        "${GGML_BASE_LIBRARY_FOR_BENCH}")
    set_target_properties(qpu_llama_cpu_flash_attn_bench PROPERTIES
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
    target_compile_features(qpu_llama_cpu_flash_attn_bench PRIVATE cxx_std_17)
    target_compile_options(qpu_llama_cpu_flash_attn_bench PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME qpu_llama_cpu_flash_attn_bench_help
            COMMAND qpu_llama_cpu_flash_attn_bench --help)
    endif()
endif()

if(QPU_LLAMA_BUILD_GRAPH_PROFILER)
    enable_language(CXX)
    if(NOT IS_DIRECTORY "${LLAMA_CPP_ROOT}")
        message(FATAL_ERROR "LLAMA_CPP_ROOT must point to the pinned llama.cpp source tree")
    endif()
    find_library(LLAMA_LIBRARY NAMES llama
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIBRARY NAMES ggml
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIBRARY NAMES ggml-base
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIBRARY_FOR_PROFILE NAMES ggml-cpu
        PATHS "${LLAMA_CPP_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
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
        BUILD_RPATH "${LLAMA_CPP_BUILD_DIR}/bin")
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

        add_executable(qpu_llama_q4_0_mx_smoke tests/qpu_llama_q4_0_mx_smoke.c)
        target_include_directories(qpu_llama_q4_0_mx_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_q4_0_mx_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_q4_0_mx_hardware COMMAND qpu_llama_q4_0_mx_smoke)
        set_tests_properties(qpu_llama_q4_0_mx_hardware PROPERTIES LABELS hardware)

        add_executable(qpu_llama_q4_k_smoke tests/qpu_llama_q4_k_smoke.c)
        target_include_directories(qpu_llama_q4_k_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_q4_k_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_q4_k_hardware COMMAND qpu_llama_q4_k_smoke)
        set_tests_properties(qpu_llama_q4_k_hardware PROPERTIES LABELS hardware)

        add_executable(qpu_llama_q6_k_smoke tests/qpu_llama_q6_k_smoke.c)
        target_include_directories(qpu_llama_q6_k_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_q6_k_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_q6_k_hardware COMMAND qpu_llama_q6_k_smoke)
        set_tests_properties(qpu_llama_q6_k_hardware PROPERTIES LABELS hardware)

        add_executable(qpu_llama_q8_0_smoke tests/qpu_llama_q8_0_smoke.c)
        target_include_directories(qpu_llama_q8_0_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_q8_0_smoke PRIVATE qpu_llama_runtime)
        add_test(NAME qpu_llama_q8_0_hardware COMMAND qpu_llama_q8_0_smoke)
        set_tests_properties(qpu_llama_q8_0_hardware PROPERTIES LABELS hardware)

        add_executable(qpu_llama_native_safety_smoke tests/qpu_llama_native_safety_smoke.c)
        target_include_directories(qpu_llama_native_safety_smoke PRIVATE generated)
        target_link_libraries(qpu_llama_native_safety_smoke PRIVATE qpu_llama_runtime)
        target_compile_options(qpu_llama_native_safety_smoke PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME qpu_llama_native_safety_hardware COMMAND qpu_llama_native_safety_smoke)
        set_tests_properties(qpu_llama_native_safety_hardware PROPERTIES LABELS hardware)
    endif()
endif()
