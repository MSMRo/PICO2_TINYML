/*
 * ==============================================================================
 * TinyML Biosignals on Raspberry Pi Pico 2 (RP2350 - Cortex-M33)
 * Example 01: ECG Arrhythmia Inference with Post-Training Quantization (INT8)
 *
 * Board: Raspberry Pi Pico 2 / Pico 2 W (RP2350)
 * Core:  Raspberry Pi Pico/RP2040/RP2350 by Earle F. Philhower, III
 * Library: TensorFlowLite_ESP32 or EloquentTinyML or Arduino_TensorFlowLite
 * ==============================================================================
 */

#include "model_data.h"
#include "test_samples.h"

// Check if TensorFlow Lite Micro headers are available in Arduino include path
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

// Tensor Arena Size: RP2350 has 520 KB SRAM, so 32 KB arena is extremely safe and lightweight!
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
  Serial.begin(115200);
  while(!Serial);
  // Wait up to 4 seconds for Serial connection on Pico 2 USB
  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 4000)) {
    delay(10);
  }

  Serial.println("\n========================================================");
  Serial.println("  TinyML Biosignals - Raspberry Pi Pico 2 (RP2350)");
  Serial.println("  Example 01: ECG Arrhythmia Classification (INT8 PTQ)");
  Serial.println("========================================================");
  Serial.printf("CPU Frequency: %lu MHz\n", rp2040.f_cpu() / 1000000);
  Serial.printf("Model Size in Flash: %u bytes\n", g_model_len);
  Serial.printf("Allocated Tensor Arena: %d bytes\n", kTensorArenaSize);

#if HAS_TFLM
  // 1. Map Model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Model schema %ld != runtime %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 2. Build Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 3. Allocate Tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors() failed! Increase kTensorArenaSize.");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("\n[Model Architecture Information]");
  Serial.printf("Input Shape: [%d, %d, %d], Type: %d\n",
                input_tensor->dims->data[0],
                input_tensor->dims->data[1],
                input_tensor->dims->data[2],
                input_tensor->type);
  Serial.printf("Input Quantization: Scale = %.6f, Zero-Point = %ld\n",
                input_tensor->params.scale,
                input_tensor->params.zero_point);
  Serial.printf("Output Shape: [%d, %d], Type: %d\n",
                output_tensor->dims->data[0],
                output_tensor->dims->data[1],
                output_tensor->type);
  Serial.printf("Output Quantization: Scale = %.6f, Zero-Point = %ld\n",
                output_tensor->params.scale,
                output_tensor->params.zero_point);
  Serial.printf("Actual Arena Used: %u bytes / %d bytes (%.1f%%)\n",
                interpreter->arena_used_bytes(),
                kTensorArenaSize,
                (float)interpreter->arena_used_bytes() * 100.0f / kTensorArenaSize);
  Serial.println("========================================================\n");
#else
  Serial.println("\n[NOTICE] TensorFlowLite library not detected in compiler search path.");
  Serial.println("Install 'EloquentTinyML' or 'TensorFlowLite_ESP32' via Arduino Library Manager,");
  Serial.println("or copy the TFLM headers into the sketch folder to run live hardware inference.");
  Serial.println("Running in Standby / Telemetry Test Mode...\n");
#endif
}

void run_single_inference(int sample_index) {
  if (sample_index < 0 || sample_index >= NUM_TEST_SAMPLES) return;

  Serial.printf("--- Evaluating Sample %d [True Label: %s (%d)] ---\n",
                sample_index,
                CLASS_NAMES[TEST_LABELS[sample_index]],
                TEST_LABELS[sample_index]);

#if HAS_TFLM
  // 1. Quantize Input: float32 -> int8: q = round(val / scale) + zero_point
  float in_scale = input_tensor->params.scale;
  int32_t in_zp = input_tensor->params.zero_point;

  if (input_tensor->type == kTfLiteInt8) {
    int8_t* in_data = input_tensor->data.int8;
    for (int i = 0; i < SAMPLE_LENGTH; i++) {
      float val = TEST_SAMPLES[sample_index][i];
      int32_t q = (int32_t)roundf(val / in_scale) + in_zp;
      in_data[i] = (int8_t)constrain(q, -128, 127);
    }
  } else if (input_tensor->type == kTfLiteFloat32) {
    float* in_data = input_tensor->data.f;
    for (int i = 0; i < SAMPLE_LENGTH; i++) {
      in_data[i] = TEST_SAMPLES[sample_index][i];
    }
  }

  // 2. Invoke Model & Benchmark Latency with high-precision timer
  uint32_t t_start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t t_latency = micros() - t_start;

  if (invoke_status != kTfLiteOk) {
    Serial.println("[ERROR] Invoke failed!");
    return;
  }

  // 3. Dequantize Output: int8 -> float32: f = (q - zero_point) * scale
  float out_scale = output_tensor->params.scale;
  int32_t out_zp = output_tensor->params.zero_point;
  
  float probabilities[NUM_CLASSES];
  int predicted_class = 0;
  float max_score = -1e9;

  for (int c = 0; c < NUM_CLASSES; c++) {
    if (output_tensor->type == kTfLiteInt8) {
      int8_t q_val = output_tensor->data.int8[c];
      probabilities[c] = (q_val - out_zp) * out_scale;
    } else {
      probabilities[c] = output_tensor->data.f[c];
    }

    if (probabilities[c] > max_score) {
      max_score = probabilities[c];
      predicted_class = c;
    }
  }

  // 4. Print Results
  Serial.printf("Predicted: %s (%d) with Confidence: %.2f%%\n",
                CLASS_NAMES[predicted_class],
                predicted_class,
                max_score * 100.0f);
  Serial.printf("Inference Latency: %lu us (%.2f ms) | Throughput: ~%.1f inferences/sec\n",
                t_latency, t_latency / 1000.0f, 1000000.0f / (float)t_latency);

  // Softmax Probabilities Breakdown
  Serial.print("Class Probabilities: [ ");
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.printf("%s: %.1f%% ", CLASS_NAMES[c], probabilities[c] * 100.0f);
  }
  Serial.println("]");

  if (predicted_class == TEST_LABELS[sample_index]) {
    Serial.println(" Result: SUCCESS (MATCH)\n");
  } else {
    Serial.println(" Result: MISMATCH\n");
  }
#else
  Serial.printf("Sample data loaded: %d points. True class: %s.\n\n",
                SAMPLE_LENGTH, CLASS_NAMES[TEST_LABELS[sample_index]]);
#endif
}

void loop() {
  Serial.println(">>> Starting In-System Verification Loop on Pico 2 <<<\n");
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    run_single_inference(i);
    delay(1000);
  }
  Serial.println(">>> All test samples evaluated. Pausing for 5 seconds...\n");
  delay(5000);
}
