// Generated Test Samples for Raspberry Pi Pico 2 Verification
#ifndef TEST_SAMPLES_H_
#define TEST_SAMPLES_H_

#include <cstdint>
#include <cstddef>

const int NUM_TEST_SAMPLES = 6;
const int SAMPLE_LENGTH = 2;

const char* const CLASS_NAMES[] = { "Region_A", "Region_B" };
const int NUM_CLASSES = 2;

const int TEST_LABELS[NUM_TEST_SAMPLES] = { 0, 0, 0, 1, 1, 1 };

const float TEST_SAMPLES[NUM_TEST_SAMPLES][SAMPLE_LENGTH] = {
  // Sample 0 (True Target/Class: 0)
  { -1.155042f, 1.445669f },
  // Sample 1 (True Target/Class: 0)
  { 0.386480f, -0.268608f },
  // Sample 2 (True Target/Class: 0)
  { 0.110625f, 1.133949f },
  // Sample 3 (True Target/Class: 1)
  { 0.775278f, -1.636276f },
  // Sample 4 (True Target/Class: 1)
  { 0.851272f, -1.280348f },
  // Sample 5 (True Target/Class: 1)
  { 1.334414f, -0.840076f }
};

#endif // TEST_SAMPLES_H_
