/*
 * ==============================================================================
 * TinyML Biosignals on Raspberry Pi Pico 2 (RP2350 - Cortex-M33)
 * Example 03: EEG Brainwave Rhythm Classification (Knowledge Distillation)
 *
 * Trained in PyTorch (Teacher-Student Distillation) -> Exported to TFLM INT8
 * Board: Raspberry Pi Pico 2 / Pico 2 W (RP2350)
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

constexpr int kTensorArenaSize = 36 * 1024;
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
  Serial.println("  Example 03: EEG Brainwave State (Distilled PyTorch)");
  Serial.println("========================================================");
  Serial.printf("RP2350 Frequency: %lu MHz\n", rp2040.f_cpu() / 1000000);
  Serial.printf("Flash footprint of Distilled Student: %u bytes\n", g_model_len);

#if HAS_TFLM
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Model schema version %ld != %d\n", model->version(), TFLITE_SCHEMA_VERSION);
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

  Serial.printf("Input Shape: [%d, %d], Type: %d\n",
                input_tensor->dims->data[1], input_tensor->dims->data[2], input_tensor->type);
  Serial.printf("Input Scale: %.6f | Zero-Point: %ld\n",
                input_tensor->params.scale, input_tensor->params.zero_point);
  Serial.printf("Actual Arena Used: %u bytes\n", interpreter->arena_used_bytes());
#endif
  Serial.println("========================================================\n");
}

void run_eeg_sample(int sample_index) {
  if (sample_index < 0 || sample_index >= NUM_TEST_SAMPLES) return;

  Serial.printf("--- EEG Epoch #%d [True State: %s] ---\n",
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
  uint32_t t_duration = micros() - t_start;

  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Model execution failed!");
    return;
  }

  float out_scale = output_tensor->params.scale;
  int32_t out_zp = output_tensor->params.zero_point;

  int best_c = 0;
  float max_prob = -1e9;
  for (int c = 0; c < NUM_CLASSES; c++) {
    int8_t q_val = output_tensor->data.int8[c];
    float prob = (q_val - out_zp) * out_scale;
    if (prob > max_prob) {
      max_prob = prob;
      best_c = c;
    }
  }

  Serial.printf("Classification: %s (Confidence: %.1f%%)\n", CLASS_NAMES[best_c], max_prob * 100.0f);
  Serial.printf("Inference Time: %lu us (%.2f ms)\n", t_duration, t_duration / 1000.0f);
  Serial.print("Band Probabilities: ");
  for (int c = 0; c < NUM_CLASSES; c++) {
    int8_t q_val = output_tensor->data.int8[c];
    float p = (q_val - out_zp) * out_scale;
    Serial.printf("[%s: %.1f%%] ", CLASS_NAMES[c], p * 100.0f);
  }
  Serial.println("\n");
#else
  Serial.printf("Telemetry preview: EEG class %s\n\n", CLASS_NAMES[TEST_LABELS[sample_index]]);
#endif
}

void loop() {
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    run_eeg_sample(i);
    delay(1000);
  }
  delay(5000);
}
