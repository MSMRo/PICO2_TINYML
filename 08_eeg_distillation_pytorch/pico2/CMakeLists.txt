# ==============================================================================
# CMakeLists.txt for Example 08: EEG Brain Rhythm Distillation (PyTorch Student INT8)
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

include(../../pico_sdk_import.cmake)

project(eeg_distillation_pytorch C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(eeg_distillation_pytorch
    main.cpp
)

target_include_directories(eeg_distillation_pytorch PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(eeg_distillation_pytorch
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

pico_enable_stdio_usb(eeg_distillation_pytorch 1)
pico_enable_stdio_uart(eeg_distillation_pytorch 0)

pico_add_extra_outputs(eeg_distillation_pytorch)
