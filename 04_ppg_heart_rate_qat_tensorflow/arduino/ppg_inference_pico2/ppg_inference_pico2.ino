/*
 * ==============================================================================
 * TinyML Biosignals on Raspberry Pi Pico 2 (RP2350 - Cortex-M33)
 * Example 04: PPG Pulse Contour & Arrhythmia Detection with QAT
 *
 * Board: Raspberry Pi Pico 2 / Pico 2 W (RP2350)
 * Core:  Raspberry Pi Pico/RP2040/RP2350 by Earle F. Philhower, III
 * ==============================================================================
 */

#include "model_data.h"
#include "test_samples.h"

#if __has_include(<tflm_cortexm.h>)
  #include <tflm_cortexm.h>
  #include <tensorflow/lite/micro/all_ops_resolver.h>
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/schema/schema_generated.h>
  #define HAS_TFLM 1
#elif __has_include(<tensorflow/lite/micro/micro_interpreter.h>)
  #include <tensorflow/lite/micro/all_ops_resolver.h>
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/schema/schema_generated.h>
  #define HAS_TFLM 1
#elif __has_include(<TensorFlowLite.h>)
  #include <TensorFlowLite.h>
  #include <tensorflow/lite/micro/all_ops_resolver.h>
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/schema/schema_generated.h>
  #define HAS_TFLM 1
#else
  #define HAS_TFLM 0
#endif

constexpr int kTensorArenaSize = 28 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

#if HAS_TFLM
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
tflite::AllOpsResolver resolver;
#endif

void setup() {
  Serial.begin(115200);
  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 4000)) {
    delay(10);
  }

  Serial.println("\n========================================================");
  Serial.println("  TinyML Biosignals - Raspberry Pi Pico 2 (RP2350)");
  Serial.println("  Example 04: PPG Pulse Waveform (Quantization-Aware Training)");
  Serial.println("========================================================");
  Serial.printf("RP2350 Clock: %lu MHz\n", rp2040.f_cpu() / 1000000);
  Serial.printf("Model Array Size in Flash: %u bytes\n", g_model_len);

#if HAS_TFLM
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Schema version mismatch: %ld vs %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() failed!");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("Arena Allocated: %d bytes | Used: %u bytes\n", kTensorArenaSize, interpreter->arena_used_bytes());
  Serial.printf("Quantization: Scale=%.6f, ZeroPoint=%ld\n", input_tensor->params.scale, input_tensor->params.zero_point);
#endif
  Serial.println("========================================================\n");
}

void run_ppg_sample(int sample_index) {
  if (sample_index < 0 || sample_index >= NUM_TEST_SAMPLES) return;

  Serial.printf(">>> PPG Window #%d [Reference: %s] <<<\n",
                sample_index, CLASS_NAMES[TEST_LABELS[sample_index]]);

#if HAS_TFLM
  float in_scale = input_tensor->params.scale;
  int32_t in_zp = input_tensor->params.zero_point;

  if (input_tensor->type == kTfLiteInt8) {
    int8_t* in_data = input_tensor->data.int8;
    for (int i = 0; i < SAMPLE_LENGTH; i++) {
      float val = TEST_SAMPLES[sample_index][i];
      int32_t q = (int32_t)roundf(val / in_scale) + in_zp;
      in_data[i] = (int8_t)constrain(q, -128, 127);
    }
  }

  uint32_t t_start = micros();
  TfLiteStatus status = interpreter->Invoke();
  uint32_t duration = micros() - t_start;

  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Invoke failed!");
    return;
  }

  float out_scale = output_tensor->params.scale;
  int32_t out_zp = output_tensor->params.zero_point;

  int pred_idx = 0;
  float max_p = -1e9;
  for (int c = 0; c < NUM_CLASSES; c++) {
    int8_t q_val = output_tensor->data.int8[c];
    float p = (q_val - out_zp) * out_scale;
    if (p > max_p) {
      max_p = p;
      pred_idx = c;
    }
  }

  Serial.printf("Prediction: %s (%.1f%% confidence)\n", CLASS_NAMES[pred_idx], max_p * 100.0f);
  Serial.printf("Execution Latency: %lu us (%.2f ms)\n", duration, duration / 1000.0f);
  Serial.print("Class Breakdown: [ ");
  for (int c = 0; c < NUM_CLASSES; c++) {
    int8_t q_val = output_tensor->data.int8[c];
    Serial.printf("%s: %.1f%% ", CLASS_NAMES[c], (q_val - out_zp) * out_scale * 100.0f);
  }
  Serial.println("]\n");
#else
  Serial.printf("Telemetry preview: PPG true class %s\n\n", CLASS_NAMES[TEST_LABELS[sample_index]]);
#endif
}

void loop() {
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    run_ppg_sample(i);
    delay(1000);
  }
  delay(4000);
}
