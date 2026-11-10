# ==============================================================================
# Top-Level CMakeLists.txt for Raspberry Pi Pico 2 TinyML Suite
# Target Board: Raspberry Pi Pico 2 (RP2350 - Cortex-M33) or Pico (RP2040)
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

# Include Pico SDK configuration
include(pico_sdk_import.cmake)

project(pico2_tinyml_suite C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Raspberry Pi Pico SDK
pico_sdk_init()

# 1. TensorFlow Lite Micro Runtime for Raspberry Pi Pico / Pico 2
add_subdirectory(libraries/pico_tflm pico_tflm_build)

# 2. Progressive TinyML Educational & Biosignal Subprojects
# Basic Foundations
add_subdirectory(01_basic_3layer_dense/pico2 01_basic_dense_build)
add_subdirectory(02_deep_mlp_classification/pico2 02_deep_mlp_build)
add_subdirectory(03_sine_regression/pico2 03_regression_build)
add_subdirectory(04_temporal_lstm_recurrent/pico2 04_lstm_build)

# Advanced Biosignal Classifiers
add_subdirectory(05_ecg_arrhythmia_ptq_tensorflow/pico2 05_ecg_build)
add_subdirectory(06_emg_gesture_pruning_tensorflow/pico2 06_emg_build)
add_subdirectory(07_ppg_heart_rate_qat_tensorflow/pico2 07_ppg_build)
add_subdirectory(08_eeg_distillation_pytorch/pico2 08_eeg_build)
add_subdirectory(09_eda_stress_hybrid_pipeline/pico2 09_eda_build)
