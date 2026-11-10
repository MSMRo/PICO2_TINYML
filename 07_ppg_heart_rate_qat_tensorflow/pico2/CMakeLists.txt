# ==============================================================================
# CMakeLists.txt for Example 07: Photoplethysmogram (PPG) Pulse & Quality (QAT INT8)
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

include(../../pico_sdk_import.cmake)

project(ppg_heart_rate_qat C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(ppg_heart_rate_qat
    main.cpp
)

target_include_directories(ppg_heart_rate_qat PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(ppg_heart_rate_qat
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

pico_enable_stdio_usb(ppg_heart_rate_qat 1)
pico_enable_stdio_uart(ppg_heart_rate_qat 0)

pico_add_extra_outputs(ppg_heart_rate_qat)
