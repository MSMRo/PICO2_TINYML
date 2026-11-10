// Generated Test Samples for Raspberry Pi Pico 2 Verification
#ifndef TEST_SAMPLES_H_
#define TEST_SAMPLES_H_

#include <cstdint>
#include <cstddef>

const int NUM_TEST_SAMPLES = 8;
const int SAMPLE_LENGTH = 1;

const float TEST_TARGETS[NUM_TEST_SAMPLES] = { -0.277956f, 0.040329f, 0.216360f, -0.243671f, 0.915283f, -0.256967f, -0.428029f, -0.082501f };

const float TEST_SAMPLES[NUM_TEST_SAMPLES][SAMPLE_LENGTH] = {
  // Sample 0 (True Target/Class: -0.2780)
  { 2.124441f },
  // Sample 1 (True Target/Class: 0.0403)
  { 0.772144f },
  // Sample 2 (True Target/Class: 0.2164)
  { 3.010305f },
  // Sample 3 (True Target/Class: -0.2437)
  { -0.260608f },
  // Sample 4 (True Target/Class: 0.9153)
  { -1.419653f },
  // Sample 5 (True Target/Class: -0.2570)
  { -2.541327f },
  // Sample 6 (True Target/Class: -0.4280)
  { 1.072743f },
  // Sample 7 (True Target/Class: -0.0825)
  { 0.857035f }
};

#endif // TEST_SAMPLES_H_
