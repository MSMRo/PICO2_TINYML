// Generated Test Samples for Raspberry Pi Pico 2 Verification
#ifndef TEST_SAMPLES_H_
#define TEST_SAMPLES_H_

#include <cstdint>
#include <cstddef>

const int NUM_TEST_SAMPLES = 8;
const int SAMPLE_LENGTH = 4;

const char* const CLASS_NAMES[] = { "Resting", "Active", "High_Stress", "Fatigue" };
const int NUM_CLASSES = 4;

const int TEST_LABELS[NUM_TEST_SAMPLES] = { 0, 0, 1, 1, 2, 2, 3, 3 };

const float TEST_SAMPLES[NUM_TEST_SAMPLES][SAMPLE_LENGTH] = {
  // Sample 0 (True Target/Class: 0)
  { -0.361068f, -0.468032f, -0.913231f, -0.855592f },
  // Sample 1 (True Target/Class: 0)
  { -0.395097f, -0.760575f, -0.511639f, -0.820527f },
  // Sample 2 (True Target/Class: 1)
  { 0.030413f, -0.737218f, 0.354695f, 0.140941f },
  // Sample 3 (True Target/Class: 1)
  { 0.345305f, -1.123086f, 0.034131f, -1.328431f },
  // Sample 4 (True Target/Class: 2)
  { -0.416846f, 0.572504f, 1.056701f, -1.147163f },
  // Sample 5 (True Target/Class: 2)
  { -0.425450f, 0.999531f, 0.843384f, 0.470457f },
  // Sample 6 (True Target/Class: 3)
  { 3.139808f, 1.235965f, -1.700883f, 1.959511f },
  // Sample 7 (True Target/Class: 3)
  { 1.941374f, 1.083295f, 0.051682f, 1.632405f }
};

#endif // TEST_SAMPLES_H_
