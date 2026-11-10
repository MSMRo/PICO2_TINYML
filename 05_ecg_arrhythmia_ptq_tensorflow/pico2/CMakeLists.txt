# ==============================================================================
# CMakeLists.txt for Example 05: ECG Arrhythmia Classification (INT8 PTQ)
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

include(../../pico_sdk_import.cmake)

project(ecg_arrhythmia_ptq C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(ecg_arrhythmia_ptq
    main.cpp
)

target_include_directories(ecg_arrhythmia_ptq PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ecg_arrhythmia_ptq
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

pico_enable_stdio_usb(ecg_arrhythmia_ptq 1)
pico_enable_stdio_uart(ecg_arrhythmia_ptq 0)

pico_add_extra_outputs(ecg_arrhythmia_ptq)
