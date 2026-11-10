# ==============================================================================
# CMakeLists.txt for Example 09: Electrodermal Activity (EDA) Stress Detection
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

include(../../pico_sdk_import.cmake)

project(eda_stress_hybrid_pipeline C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(eda_stress_hybrid_pipeline
    main.cpp
)

target_include_directories(eda_stress_hybrid_pipeline PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(eda_stress_hybrid_pipeline
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

pico_enable_stdio_usb(eda_stress_hybrid_pipeline 1)
pico_enable_stdio_uart(eda_stress_hybrid_pipeline 0)

pico_add_extra_outputs(eda_stress_hybrid_pipeline)
