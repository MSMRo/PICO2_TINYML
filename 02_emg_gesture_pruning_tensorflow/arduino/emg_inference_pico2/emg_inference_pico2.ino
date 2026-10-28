/*
 * ==============================================================================
 * TinyML Biosignals on Raspberry Pi Pico 2 (RP2350 - Cortex-M33)
 * Example 02: EMG Muscle Activity Detection with Pruning + INT8 Quantization
 *
 * Board: Raspberry Pi Pico 2 / Pico 2 W (RP2350)
 * Core:  Raspberry Pi Pico/RP2040/RP2350 by Earle F. Philhower, III
 * ==============================================================================
 */

#include "model_data.h"
#include "test_samples.h"

#if __has_include(<TensorFlowLite.h>)
  #include <TensorFlowLite.h>
  #include <tensorflow/lite/micro/all_ops_resolver.h>
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/schema/schema_generated.h>
  #define HAS_TFLM 1
#elif __has_include(<tensorflow/lite/micro/micro_interpreter.h>)
  #include <tensorflow/lite/micro/all_ops_resolver.h>
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/schema/schema_generated.h>
  #define HAS_TFLM 1
#else
  #define HAS_TFLM 0
#endif

// On Pico 2, standard onboard LED is pin LED_BUILTIN (GPIO 25 or CYW43 on Pico 2 W)
#ifndef LED_BUILTIN
  #define LED_BUILTIN 25
#endif

constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

#if HAS_TFLM
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
tflite::AllOpsResolver resolver;
#endif

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(115200);
  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 4000)) {
    delay(10);
  }

  Serial.println("\n========================================================");
  Serial.println("  TinyML Biosignals - Raspberry Pi Pico 2 (RP2350)");
  Serial.println("  Example 02: EMG Muscle Contraction (Pruned + INT8)");
  Serial.println("========================================================");
  Serial.printf("RP2350 Clock: %lu MHz\n", rp2040.f_cpu() / 1000000);
  Serial.printf("Model Array Size in Flash: %u bytes\n", g_model_len);

#if HAS_TFLM
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Schema mismatch: %ld vs %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] Tensor allocation failed!");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("Arena Used: %u / %d bytes\n", interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.printf("Input Scale: %.6f | Zero-Point: %ld\n", input_tensor->params.scale, input_tensor->params.zero_point);
#endif
  Serial.println("========================================================\n");
}

void run_emg_sample(int sample_index) {
  if (sample_index < 0 || sample_index >= NUM_TEST_SAMPLES) return;

  Serial.printf(">>> Testing EMG Sample #%d [Ground Truth: %s] <<<\n",
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

  uint32_t t0 = micros();
  TfLiteStatus status = interpreter->Invoke();
  uint32_t latency = micros() - t0;

  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Invoke failed!");
    return;
  }

  float out_scale = output_tensor->params.scale;
  int32_t out_zp = output_tensor->params.zero_point;

  int best_class = 0;
  float max_prob = -1e9;
  for (int c = 0; c < NUM_CLASSES; c++) {
    int8_t q_val = output_tensor->data.int8[c];
    float prob = (q_val - out_zp) * out_scale;
    if (prob > max_prob) {
      max_prob = prob;
      best_class = c;
    }
  }

  Serial.printf("Detection: %s (Confidence: %.1f%%)\n", CLASS_NAMES[best_class], max_prob * 100.0f);
  Serial.printf("Latency: %lu us (%.2f ms)\n", latency, latency / 1000.0f);

  // Visual LED Feedback: High contraction = Solid ON, Low = Blink, Rest = OFF
  if (best_class == 2) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else if (best_class == 1) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(50);
    digitalWrite(LED_BUILTIN, LOW);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
#else
  Serial.printf("Simulated run: Ground truth %s\n", CLASS_NAMES[TEST_LABELS[sample_index]]);
#endif
  Serial.println();
}

void loop() {
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    run_emg_sample(i);
    delay(1200);
  }
  delay(4000);
}
