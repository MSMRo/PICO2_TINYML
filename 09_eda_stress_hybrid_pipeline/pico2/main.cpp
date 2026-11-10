/*
 * ==============================================================================
 * TinyML Biosignals on Raspberry Pi Pico 2 (RP2350 - ARM Cortex-M33)
 * Example 09: Electrodermal Activity (EDA/GSR) Sympathetic Stress Detector
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

// Tensor Arena: 32 KB for 1D CNN EDA feature extractor
constexpr int kTensorArenaSize = 32 * 1024;
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

    sleep_ms(3000);

    printf("\n========================================================\n");
    printf("  Raspberry Pi Pico 2 (RP2350 ARM Cortex-M33) - TinyML\n");
    printf("  Example 09: Electrodermal Activity (EDA/GSR) Stress\n");
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
    printf("  - Input Tensor: Dimensions [");
    for (int d = 0; d < input_tensor->dims->size; ++d) {
        printf("%d%s", input_tensor->dims->data[d], d < input_tensor->dims->size - 1 ? ", " : "");
    }
    printf("], Type=%d\n", input_tensor->type);
    printf("    Scale: %e, Zero Point: %ld\n",
           input_tensor->params.scale, (long)input_tensor->params.zero_point);

    printf("  - Output Tensor: Dimensions [");
    for (int d = 0; d < output_tensor->dims->size; ++d) {
        printf("%d%s", output_tensor->dims->data[d], d < output_tensor->dims->size - 1 ? ", " : "");
    }
    printf("], Type=%d\n", output_tensor->type);
    printf("    Scale: %e, Zero Point: %ld\n",
           output_tensor->params.scale, (long)output_tensor->params.zero_point);
    printf("  - Used Arena: %u bytes\n\n", (unsigned)interpreter->arena_used_bytes());

    gpio_put(PICO_DEFAULT_LED_PIN, 0);
}

void run_eda_inference() {
    printf("--- Running Electrodermal Activity (GSR) Stress Inference ---\n");
    int correct_predictions = 0;
    uint64_t total_latency_us = 0;

    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        // 1. Quantize 200-sample skin conductance window into INT8 input tensor
        for (int s = 0; s < SAMPLE_LENGTH; ++s) {
            float val = TEST_SAMPLES[i][s];
            input_tensor->data.int8[s] = pico_tflm::QuantizeFloatToInt8(
                val, input_tensor->params.scale, input_tensor->params.zero_point);
        }

        // 2. High-precision hardware timer benchmark
        uint64_t t_start = time_us_64();
        TfLiteStatus invoke_status = interpreter->Invoke();
        uint64_t t_end = time_us_64();
        uint64_t latency = t_end - t_start;
        total_latency_us += latency;

        if (invoke_status != kTfLiteOk) {
            printf("[ERROR] Invoke failed on EDA sample %d\n", i);
            continue;
        }

        // 3. Dequantize output probabilities
        int best_class = 0;
        float max_prob = -1.0f;
        float probs[NUM_CLASSES];

        for (int c = 0; c < NUM_CLASSES; ++c) {
            int8_t raw_val = output_tensor->data.int8[c];
            probs[c] = pico_tflm::DequantizeInt8ToFloat(
                raw_val, output_tensor->params.scale, output_tensor->params.zero_point);
            if (probs[c] > max_prob) {
                max_prob = probs[c];
                best_class = c;
            }
        }

        int true_label = TEST_LABELS[i];
        bool is_correct = (best_class == true_label);
        if (is_correct) correct_predictions++;

        // Visual alert based on sympathetic stress response
        if (best_class == 1) {
            // Stress / High arousal: rapid urgent flashes
            for (int k = 0; k < 3; ++k) {
                gpio_put(PICO_DEFAULT_LED_PIN, 1); sleep_ms(25);
                gpio_put(PICO_DEFAULT_LED_PIN, 0); sleep_ms(25);
            }
        } else {
            // Calm baseline: soft gentle blip
            gpio_put(PICO_DEFAULT_LED_PIN, 1); sleep_ms(20);
            gpio_put(PICO_DEFAULT_LED_PIN, 0);
        }

        printf("EDA %2d | True: %-20s | Pred: %-20s (Conf: %5.1f%%) | Match: [%s] | Latency: %llu us\n",
               i,
               CLASS_NAMES[true_label],
               CLASS_NAMES[best_class],
               max_prob * 100.0f,
               is_correct ? "OK" : "MISMATCH",
               latency);
    }

    float accuracy = (100.0f * correct_predictions) / NUM_TEST_SAMPLES;
    uint64_t avg_latency = total_latency_us / NUM_TEST_SAMPLES;
    printf("------------------------------------------------------------------------\n");
    printf("EDA Benchmark Summary: Accuracy: %d/%d (%.1f%%) | Avg Latency: %llu us\n\n",
           correct_predictions, NUM_TEST_SAMPLES, accuracy, avg_latency);
}

int main() {
    setup_tinyml();

    while (true) {
        run_eda_inference();
        sleep_ms(3000);
    }

    return 0;
}
