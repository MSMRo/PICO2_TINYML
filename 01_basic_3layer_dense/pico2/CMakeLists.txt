# ==============================================================================
# CMakeLists.txt for Example 01: Minimal 3-Layer Neural Network on Pico 2
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

# Include Pico SDK import script from repository root
include(../../pico_sdk_import.cmake)

project(basic_3layer_dense C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Raspberry Pi Pico SDK
pico_sdk_init()

# Add Pico TensorFlow Lite Micro static library
if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(basic_3layer_dense
    main.cpp
)

target_include_directories(basic_3layer_dense PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(basic_3layer_dense
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

# Enable USB CDC stdio output (for minicom / PuTTY / Serial Monitor)
pico_enable_stdio_usb(basic_3layer_dense 1)
pico_enable_stdio_uart(basic_3layer_dense 0)

# Generate UF2, BIN, HEX, and ELF outputs for flashing
pico_add_extra_outputs(basic_3layer_dense)
