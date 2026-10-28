/*
 * ECG Float32 inference from an external 500 Hz analog waveform.
 *
 * Input: Cayetino R4 DAC A0 -> Pico GP26 / ADC0, plus shared GND.
 * The Cayetino waveform CSV must be played at 500 Hz.
 * The classifier was trained with 187 samples at 250 Hz, so this sketch
 * captures 374 ADC ticks and keeps every other sample before inference.
 *
 * Board: Raspberry Pi Pico W (RP2040) or Pico 2 W (RP2350)
 * Core:  Raspberry Pi Pico/RP2040/RP2350 by Earle F. Philhower, III
 * Runtime: ArduTFLite (requires its Chirale_TensorFlowLite dependency)
 *
 * Never drive a Pico ADC input above 3.3 V. Verify the Cayetino A0 voltage
 * before connecting; share ground between the two boards.
 */

#include <Arduino.h>
#include <hardware/adc.h>
#include <hardware/sync.h>
#include <pico/time.h>

#include <ArduTFLite.h>

#include <math.h>
#include <stdint.h>

#include "model_data.h"

namespace {

constexpr uint8_t kAdcGpio = 26;
constexpr uint8_t kAdcInput = 0;
constexpr uint32_t kAdcSampleRateHz = 500;
constexpr uint32_t kModelSampleRateHz = 250;
constexpr uint32_t kModelWindowSamples = 187;
constexpr uint32_t kRawWindowSamples =
    kModelWindowSamples * (kAdcSampleRateHz / kModelSampleRateHz);
constexpr uint32_t kSamplePeriodUs = 1000000 / kAdcSampleRateHz;
constexpr uint32_t kRingBufferSize = 512;
constexpr uint32_t kRingBufferMask = kRingBufferSize - 1;
constexpr size_t kTensorArenaSize = 64 * 1024;
constexpr uint8_t kClassCount = 3;

static_assert((kRingBufferSize & kRingBufferMask) == 0,
              "ADC ring buffer size must be a power of two.");
static_assert(kAdcSampleRateHz % kModelSampleRateHz == 0,
              "ADC rate must be an integer multiple of model rate.");
static_assert(kRawWindowSamples < kRingBufferSize,
              "Ring buffer must hold at least one full inference window.");

const char* const kClassNames[kClassCount] = {
    "Normal", "Tachycardia", "Bradycardia"};

alignas(16) byte tensorArena[kTensorArenaSize];
volatile uint16_t adcRing[kRingBufferSize];
volatile uint32_t adcSampleHead = 0;
repeating_timer_t adcTimer;

uint32_t lastInferenceHead = 0;

bool adcSampleCallback(repeating_timer_t*) {
  const uint32_t writeIndex = adcSampleHead;
  adcRing[writeIndex & kRingBufferMask] = adc_read();
  adcSampleHead = writeIndex + 1;
  return true;
}

bool takeModelWindow(uint16_t* modelSamples, uint32_t* frameHead) {
  const uint32_t interruptState = save_and_disable_interrupts();
  const uint32_t head = adcSampleHead;
  restore_interrupts(interruptState);

  if (head - lastInferenceHead < kRawWindowSamples) {
    return false;
  }

  const uint32_t firstSample = head - kRawWindowSamples;
  for (uint32_t i = 0; i < kModelWindowSamples; ++i) {
    const uint32_t ringIndex =
        (firstSample + i * (kAdcSampleRateHz / kModelSampleRateHz)) &
        kRingBufferMask;
    modelSamples[i] = adcRing[ringIndex];
  }

  lastInferenceHead = head;
  *frameHead = head;
  return true;
}

bool normalizeAndSetInputs(const uint16_t* samples) {
  float mean = 0.0f;
  for (uint32_t i = 0; i < kModelWindowSamples; ++i) {
    mean += static_cast<float>(samples[i]);
  }
  mean /= static_cast<float>(kModelWindowSamples);

  float variance = 0.0f;
  for (uint32_t i = 0; i < kModelWindowSamples; ++i) {
    const float centered = static_cast<float>(samples[i]) - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(kModelWindowSamples);
  const float standardDeviation = sqrtf(variance);
  if (standardDeviation < 1e-6f) {
    Serial.println("[ERROR] ADC window has no measurable signal.");
    return false;
  }

  for (uint32_t i = 0; i < kModelWindowSamples; ++i) {
    const float normalized =
        (static_cast<float>(samples[i]) - mean) / standardDeviation;
    if (!modelSetInput(normalized, static_cast<int>(i))) {
      Serial.println("[ERROR] ArduTFLite rejected an input sample.");
      return false;
    }
  }
  return true;
}

bool initializeInterpreter() {
  if (!modelInit(g_model, tensorArena, static_cast<int>(kTensorArenaSize))) {
    Serial.println("[ERROR] ArduTFLite could not initialize the model.");
    return false;
  }
  Serial.println("[INFO] ArduTFLite model initialized.");
  return true;
}

void runInference(const uint16_t* samples, uint32_t frameHead) {
  if (!normalizeAndSetInputs(samples)) {
    return;
  }

  const uint32_t startUs = micros();
  if (!modelRunInference()) {
    Serial.println("[ERROR] Model invocation failed.");
    return;
  }
  const uint32_t elapsedUs = micros() - startUs;

  int predictedClass = 0;
  float highestScore = modelGetOutput(0);
  for (uint8_t i = 1; i < kClassCount; ++i) {
    const float score = modelGetOutput(i);
    if (score > highestScore) {
      highestScore = score;
      predictedClass = i;
    }
  }

  Serial.print("[ECG] ADC samples=");
  Serial.print(frameHead);
  Serial.print(" | prediction=");
  Serial.print(kClassNames[predictedClass]);
  Serial.print(" | score=");
  Serial.print(highestScore * 100.0f, 1);
  Serial.print("% | inference=");
  Serial.print(elapsedUs);
  Serial.println(" us");
}

}  // namespace

void setup() {
  Serial.begin(115200);
  const uint32_t serialWaitStart = millis();
  while (!Serial && millis() - serialWaitStart < 3000) {
    delay(10);
  }

  Serial.println("\nECG live inference: ADC 500 Hz -> model 250 Hz");
  Serial.println("Connect Cayetino A0 to GP26 and connect GND to GND.");
  Serial.println("Confirm the ADC signal never exceeds 3.3 V.");

  if (!initializeInterpreter()) {
    while (true) {
      delay(1000);
    }
  }

  adc_init();
  adc_gpio_init(kAdcGpio);
  adc_select_input(kAdcInput);
  if (!add_repeating_timer_us(
          -static_cast<int64_t>(kSamplePeriodUs),
          adcSampleCallback, nullptr, &adcTimer)) {
    Serial.println("[ERROR] Could not start the 500 Hz ADC timer.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("[INFO] Sampling started. First prediction follows one window.");
}

void loop() {
  static uint16_t modelSamples[kModelWindowSamples];
  uint32_t frameHead = 0;
  if (takeModelWindow(modelSamples, &frameHead)) {
    runInference(modelSamples, frameHead);
  }
  delay(1);
}
