/*
 * ==============================================================================
 * TinyML on Raspberry Pi Pico 2 (RP2350 - ARM Cortex-M33)
 * Example 01: Minimal 3-Layer Neural Network Inference (INT8 PTQ)
 * Framework: Raspberry Pi Pico C++ SDK & TensorFlow Lite for Microcontrollers
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

// Onboard LED pin (GPIO 25 on standard Pico / Pico 2)
#ifndef PICO_DEFAULT_LED_PIN
#define PICO_DEFAULT_LED_PIN 25
#endif

// Tensor Arena: 16 KB is more than enough for this 3-layer MLP
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Global TFLM runtime pointers
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static tflite::AllOpsResolver resolver;

void setup_tinyml() {
    // 1. Initialize stdio for USB Serial Monitor
    stdio_init_all();

    // 2. Configure onboard LED
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 1); // Turn ON during setup

    // Give user 3 seconds to open USB Serial Monitor (e.g. minicom, PuTTY, picotool)
    sleep_ms(3000);

    printf("\n========================================================\n");
    printf("  Raspberry Pi Pico 2 (RP2350 ARM Cortex-M33) - TinyML\n");
    printf("  Example 01: Minimal 3-Layer Neural Network (Dense)\n");
    printf("========================================================\n");
    printf("CPU Frequency: %lu MHz\n", (unsigned long)(clock_get_hz(clk_sys) / 1000000));
    printf("Flash Model Size: %u bytes\n", g_model_len);
    printf("Allocated Tensor Arena: %d bytes\n", kTensorArenaSize);

    // 3. Map FlatBuffer Model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[ERROR] Model schema %ld != runtime schema %d\n",
               (long)model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // 4. Instantiate Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // 5. Allocate memory for tensors
    TfLiteStatus alloc_status = interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        printf("[ERROR] AllocateTensors() failed! Check arena size.\n");
        return;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    printf("\n[Model Architecture Metadata]\n");
    printf("Input Tensor  : [%d, %d], Type: %d (INT8)\n",
           input_tensor->dims->data[0], input_tensor->dims->data[1], input_tensor->type);
    printf("Input Quant   : Scale = %.6f, Zero-Point = %ld\n",
           input_tensor->params.scale, (long)input_tensor->params.zero_point);
    printf("Output Tensor : [%d, %d], Type: %d (INT8)\n",
           output_tensor->dims->data[0], output_tensor->dims->data[1], output_tensor->type);
    printf("Output Quant  : Scale = %.6f, Zero-Point = %ld\n",
           output_tensor->params.scale, (long)output_tensor->params.zero_point);
    printf("Arena Used    : %u bytes / %d bytes (%.1f%%)\n",
           (unsigned int)interpreter->arena_used_bytes(),
           kTensorArenaSize,
           (float)interpreter->arena_used_bytes() * 100.0f / kTensorArenaSize);
    printf("========================================================\n\n");

    gpio_put(PICO_DEFAULT_LED_PIN, 0); // Setup complete
}

void run_single_inference(int sample_idx) {
    if (sample_idx < 0 || sample_idx >= NUM_TEST_SAMPLES) return;

    printf("--- [Sample %d] True Label: %s (%d) ---\n",
           sample_idx, CLASS_NAMES[TEST_LABELS[sample_idx]], TEST_LABELS[sample_idx]);

    // 1. Quantize input float32 -> int8: q = round(x / scale) + zero_point
    float in_scale = input_tensor->params.scale;
    int32_t in_zp = input_tensor->params.zero_point;
    int8_t* in_data = input_tensor->data.int8;

    printf("Input Features (x1, x2): [ ");
    for (int i = 0; i < SAMPLE_LENGTH; i++) {
        float raw_val = TEST_SAMPLES[sample_idx][i];
        in_data[i] = pico_tflm::QuantizeFloatToInt8(raw_val, in_scale, in_zp);
        printf("%.4f ", raw_val);
    }
    printf("]\n");

    // 2. High-precision Latency Benchmark
    gpio_put(PICO_DEFAULT_LED_PIN, 1);
    uint64_t t_start = time_us_64();
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint64_t t_latency_us = time_us_64() - t_start;
    gpio_put(PICO_DEFAULT_LED_PIN, 0);

    if (invoke_status != kTfLiteOk) {
        printf("[ERROR] interpreter->Invoke() failed!\n");
        return;
    }

    // 3. Dequantize output int8 -> float32: f = (q - zero_point) * scale
    float out_scale = output_tensor->params.scale;
    int32_t out_zp = output_tensor->params.zero_point;

    float probs[NUM_CLASSES];
    int predicted_class = 0;
    float max_prob = -1.0f;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int8_t q_val = output_tensor->data.int8[c];
        probs[c] = pico_tflm::DequantizeInt8ToFloat(q_val, out_scale, out_zp);
        if (probs[c] > max_prob) {
            max_prob = probs[c];
            predicted_class = c;
        }
    }

    // 4. Report Results
    printf("Predicted     : %s (%d) | Confidence: %.2f%%\n",
           CLASS_NAMES[predicted_class], predicted_class, max_prob * 100.0f);
    printf("Inference Time: %llu us (%.3f ms) | Throughput: ~%.0f inf/sec\n",
           t_latency_us, (float)t_latency_us / 1000.0f, 1000000.0f / (float)t_latency_us);
    printf("Probabilities : [ ");
    for (int c = 0; c < NUM_CLASSES; c++) {
        printf("%s: %.1f%% ", CLASS_NAMES[c], probs[c] * 100.0f);
    }
    printf("]\n");

    if (predicted_class == TEST_LABELS[sample_idx]) {
        printf("Result Check  : [MATCH / SUCCESS]\n\n");
    } else {
        printf("Result Check  : [MISMATCH]\n\n");
    }
}

int main() {
    setup_tinyml();

    while (true) {
        printf(">>> Starting Continuous Verification Cycle on Pico 2 <<<\n\n");
        for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
            run_single_inference(i);
            sleep_ms(1500);
        }
        printf(">>> Cycle completed. Pausing for 5 seconds...\n\n");
        sleep_ms(5000);
    }

    return 0;
}
