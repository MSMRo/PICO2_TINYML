/*
 * ==============================================================================
 * TensorFlow Lite for Microcontrollers (TFLM) - Raspberry Pi Pico 2 / Pico Port
 * Hardware timer, stdio DebugLog, and peripheral utility implementation
 * ==============================================================================
 */

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include "pico/stdlib.h"
#include "hardware/timer.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "peripherals/utility.h"

namespace peripherals {

void DelayMicroseconds(uint32_t delay) {
    sleep_us(delay);
}

void DelayMilliseconds(uint32_t amount) {
    sleep_ms(amount);
}

uint32_t MicrosecondsCounter() {
    return time_us_32();
}

uint32_t MillisecondsCounter() {
    return to_ms_since_boot(get_absolute_time());
}

void DebugOutput(const char* s) {
    printf("%s\n", s);
}

TimestampBuffer::TimestampBuffer()
    : insert_index_{0}, show_index_{0}, entries_{} {}

TimestampBuffer& TimestampBuffer::Instance() {
    static TimestampBuffer TimestampBuffer_static;
    return TimestampBuffer_static;
}

void TimestampBuffer::Insert(const char c) {
    size_t next_index = (insert_index_ + 1) % kNumEntries;
    if (next_index == show_index_) {
        if (entries_[insert_index_].c_ != '\0') {
            entries_[insert_index_].c_ = '\0';
            entries_[insert_index_].timestamp_us_ = time_us_32();
        }
    } else {
        entries_[insert_index_].c_ = c;
        entries_[insert_index_].timestamp_us_ = time_us_32();
        insert_index_ = next_index;
    }
}

void TimestampBuffer::Show() {
    while (show_index_ != insert_index_) {
        if (entries_[show_index_].c_ == '\0') {
            printf("%lu: *** TimestampBuffer Overflow ***\n", (unsigned long)entries_[show_index_].timestamp_us_);
            break;
        } else {
            printf("%lu: %c\n", (unsigned long)entries_[show_index_].timestamp_us_, entries_[show_index_].c_);
            show_index_ = (show_index_ + 1) % kNumEntries;
        }
    }
    show_index_ = insert_index_;
}

} // namespace peripherals

namespace tflite {
void InitializeTarget() {
    // stdio_init_all() is initialized by main
}
} // namespace tflite

extern "C" void DebugLog(const char* s) {
    printf("%s", s);
}
