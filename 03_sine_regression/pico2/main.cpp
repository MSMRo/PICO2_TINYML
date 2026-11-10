/*
 * ==============================================================================
 * TinyML on Raspberry Pi Pico 2 (RP2350 - ARM Cortex-M33)
 * Example 03: Non-Linear Function Approximation (Sine Wave Regression)
 * Function Target: y = sin(x) * cos(2x)
 * Framework: Raspberry Pi Pico C++ SDK & TensorFlow Lite Micro (CMSIS-NN)
 * ==============================================================================
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/timer.h"

#include "pico_tflm.h"
#include "model_data.h"
#include "test_samples.h"

#ifndef PICO_DEFAULT_LED_PIN
#define PICO_DEFAULT_LED_PIN 25
#endif

// Tensor Arena: 16 KB for 2 hidden layers (32 units each)
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static tflite::AllOpsResolver resolver;

void setup_tinyml() {
    stdio_init_all();

    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 1);

    // Allow time for USB serial monitor connection
    sleep_ms(3000);

    printf("\n========================================================\n");
    printf("  Raspberry Pi Pico 2 (RP2350 ARM Cortex-M33) - TinyML\n");
    printf("  Example 03: Non-Linear Regression [y = sin(x)*cos(2x)]\n");
    printf("========================================================\n");
    printf("CPU Frequency: %lu MHz\n", (unsigned long)(clock_get_hz(clk_sys) / 1000000));
    printf("Flash Model Size: %u bytes\n", g_model_len);
    printf("Allocated Tensor Arena: %d bytes\n", kTensorArenaSize);

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[ERROR] Model schema %ld != runtime %d\n",
               (long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("[ERROR] AllocateTensors() failed!\n");
        return;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    printf("[INFO] Tensor Allocation Successful:\n");
    printf("  - Input Tensor: Dimensions [%d, %d], Type=%d\n",
           input_tensor->dims->data[0], input_tensor->dims->data[1], input_tensor->type);
    printf("    Scale: %e, Zero Point: %ld\n",
           input_tensor->params.scale, (long)input_tensor->params.zero_point);
    printf("  - Output Tensor: Dimensions [%d, %d], Type=%d\n",
           output_tensor->dims->data[0], output_tensor->dims->data[1], output_tensor->type);
    printf("    Scale: %e, Zero Point: %ld\n",
           output_tensor->params.scale, (long)output_tensor->params.zero_point);
    printf("  - Used Arena: %u bytes\n\n", (unsigned)interpreter->arena_used_bytes());

    gpio_put(PICO_DEFAULT_LED_PIN, 0);
}

void run_regression_inference() {
    printf("--- Running Regression Inference on Test Set ---\n");
    float total_abs_error = 0.0f;
    uint64_t total_latency_us = 0;

    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        float x_val = TEST_SAMPLES[i][0];
        float y_true = TEST_TARGETS[i];

        // 1. Quantize continuous input x -> int8
        input_tensor->data.int8[0] = pico_tflm::QuantizeFloatToInt8(
            x_val, input_tensor->params.scale, input_tensor->params.zero_point);

        // 2. High-precision hardware timer benchmark
        uint64_t t_start = time_us_64();
        TfLiteStatus invoke_status = interpreter->Invoke();
        uint64_t t_end = time_us_64();
        uint64_t latency = t_end - t_start;
        total_latency_us += latency;

        if (invoke_status != kTfLiteOk) {
            printf("[ERROR] Invoke failed on sample %d\n", i);
            continue;
        }

        // 3. Dequantize predicted scalar output int8 -> float
        int8_t raw_output = output_tensor->data.int8[0];
        float y_pred = pico_tflm::DequantizeInt8ToFloat(
            raw_output, output_tensor->params.scale, output_tensor->params.zero_point);

        float abs_error = std::fabs(y_pred - y_true);
        total_abs_error += abs_error;

        // Toggle LED to provide physical feedback
        gpio_put(PICO_DEFAULT_LED_PIN, 1);
        sleep_ms(30);
        gpio_put(PICO_DEFAULT_LED_PIN, 0);

        printf("Sample %2d | x = %+7.4f rad | Pred y = %+7.4f | True y = %+7.4f | Err: %6.4f | Latency: %llu us\n",
               i, x_val, y_pred, y_true, abs_error, latency);
    }

    float mean_mae = total_abs_error / NUM_TEST_SAMPLES;
    uint64_t avg_latency = total_latency_us / NUM_TEST_SAMPLES;
    printf("------------------------------------------------------------------------\n");
    printf("Evaluation Summary: Mean Absolute Error (MAE): %6.4f | Avg Latency: %llu us\n\n",
           mean_mae, avg_latency);
}

int main() {
    setup_tinyml();

    while (true) {
        run_regression_inference();
        sleep_ms(2500);
    }

    return 0;
}
