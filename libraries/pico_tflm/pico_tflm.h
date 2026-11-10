/*
 * ==============================================================================
 * pico_tflm.h - TensorFlow Lite Micro Convenience Header for Raspberry Pi Pico 2
 * Core: ARM Cortex-M33 (RP2350) / Cortex-M0+ (RP2040)
 * ==============================================================================
 */

#ifndef PICO_TFLM_H_
#define PICO_TFLM_H_

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "pico/stdlib.h"
#include "hardware/timer.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace pico_tflm {

/**
 * @brief Quantizes a float32 value into an int8 value:
 *        q = clip(round(val / scale) + zero_point, -128, 127)
 */
inline int8_t QuantizeFloatToInt8(float val, float scale, int32_t zero_point) {
    int32_t q = static_cast<int32_t>(std::round(val / scale)) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return static_cast<int8_t>(q);
}

/**
 * @brief Dequantizes an int8 value into a float32 value:
 *        f = (q - zero_point) * scale
 */
inline float DequantizeInt8ToFloat(int8_t q_val, float scale, int32_t zero_point) {
    return (static_cast<float>(q_val) - static_cast<float>(zero_point)) * scale;
}

/**
 * @brief Simple execution timer helper for microsecond latency measurement
 */
class ScopedTimer {
public:
    ScopedTimer(uint64_t* out_duration_us) : out_duration_(out_duration_us), start_time_(time_us_64()) {}
    ~ScopedTimer() {
        if (out_duration_) {
            *out_duration_ = time_us_64() - start_time_;
        }
    }
private:
    uint64_t* out_duration_;
    uint64_t start_time_;
};

} // namespace pico_tflm

#endif // PICO_TFLM_H_
