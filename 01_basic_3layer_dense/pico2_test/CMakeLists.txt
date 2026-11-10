# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

# 1. Modificar el target board para la Pico 2 W
set(PICO_BOARD pico2_w CACHE STRING "Board type")
# 2. Especificar explícitamente la plataforma del procesador (RP2350)
set(PICO_PLATFORM rp2350 CACHE STRING "Target platform")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(pico2_test C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add Pico TensorFlow Lite Micro static library
if (NOT TARGET pico_tflm)
    add_subdirectory(./libraries/pico_tflm pico_tflm_build)
endif()

# Add executable. Default name is the project name, version 0.1
add_executable(pico2_test pico2_test.cpp )

pico_set_program_name(pico2_test "pico2_test")
pico_set_program_version(pico2_test "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(pico2_test 0)
pico_enable_stdio_usb(pico2_test 1)

# Add the standard library to the build
target_link_libraries(pico2_test
        pico_stdlib
        hardware_timer
        hardware_clocks
        pico_tflm
        # 3. (Opcional) Si en tu proyecto vas a usar la conectividad Inalámbrica de la Pico 2 W (WiFi/Bluetooth):
        # pico_cyw43_arch_none # Descomenta para inicializar el chip Wi-Fi sin red.
        # pico_cyw43_arch_lwip_threadsafe_background # Descomenta para uso completo de Wi-Fi
)

# Add the standard include files to the build
target_include_directories(pico2_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(pico2_test)