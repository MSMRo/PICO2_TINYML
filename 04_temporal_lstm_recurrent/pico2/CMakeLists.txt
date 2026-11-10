# ==============================================================================
# CMakeLists.txt for Example 04: Temporal Recurrent LSTM on Pico 2
# Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

include(../../pico_sdk_import.cmake)

project(temporal_lstm_recurrent C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

if (NOT TARGET pico_tflm)
    add_subdirectory(/../../libraries/pico_tflm pico_tflm_build)
endif()

add_executable(temporal_lstm_recurrent
    main.cpp
)

target_include_directories(temporal_lstm_recurrent PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(temporal_lstm_recurrent
    pico_stdlib
    hardware_timer
    hardware_clocks
    pico_tflm
)

pico_enable_stdio_usb(temporal_lstm_recurrent 1)
pico_enable_stdio_uart(temporal_lstm_recurrent 0)

pico_add_extra_outputs(temporal_lstm_recurrent)
