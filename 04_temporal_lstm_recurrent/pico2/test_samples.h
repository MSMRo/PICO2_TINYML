// Generated Test Samples for Raspberry Pi Pico 2 Verification
#ifndef TEST_SAMPLES_H_
#define TEST_SAMPLES_H_

#include <cstdint>
#include <cstddef>

const int NUM_TEST_SAMPLES = 4;
const int SAMPLE_LENGTH = 10;

const char* const CLASS_NAMES[] = { "Sinusoidal_Harmonic", "Transient_Anomaly" };
const int NUM_CLASSES = 2;

const int TEST_LABELS[NUM_TEST_SAMPLES] = { 0, 0, 1, 1 };

const float TEST_SAMPLES[NUM_TEST_SAMPLES][SAMPLE_LENGTH] = {
  // Sample 0 (True Target/Class: 0)
  { 0.568794f, 1.111918f, 0.914986f, 0.371499f, -0.243973f, -0.888459f, -0.804300f, -0.777224f, -0.240396f, 0.450464f },
  // Sample 1 (True Target/Class: 0)
  { 1.127981f, 1.030428f, 0.264993f, -0.524903f, -0.901499f, -0.912520f, -0.395319f, 0.555957f, 1.045955f, 0.910096f },
  // Sample 2 (True Target/Class: 1)
  { 0.091856f, 0.134751f, 1.482543f, 1.509418f, -0.120708f, -0.033151f, 0.040835f, 0.028389f, -0.040613f, -0.021939f },
  // Sample 3 (True Target/Class: 1)
  { 0.067482f, -0.098034f, -0.005913f, 0.052844f, 0.059858f, 1.611876f, 1.458376f, -0.003083f, -0.060401f, 0.106780f }
};

#endif // TEST_SAMPLES_H_
