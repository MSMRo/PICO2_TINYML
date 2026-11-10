# ==============================================================================
# CMakeLists.txt for Example 06: EMG Gesture Classification (Pruned INT8)
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

include(../../pico_sdk_import.cmake)

project(emg_gesture_pruning C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(emg_gesture_pruning
    main.cpp
)

target_include_directories(emg_gesture_pruning PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(emg_gesture_pruning
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

pico_enable_stdio_usb(emg_gesture_pruning 1)
pico_enable_stdio_uart(emg_gesture_pruning 0)

pico_add_extra_outputs(emg_gesture_pruning)
