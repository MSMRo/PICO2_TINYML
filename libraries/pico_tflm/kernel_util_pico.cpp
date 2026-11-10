/*
 * ==============================================================================
 * kernel_util_pico.cpp - Lightweight Kernel Utilities for TFLM on Raspberry Pi Pico
 * Core: ARM Cortex-M33 (RP2350) / Cortex-M0+ (RP2040)
 * ==============================================================================
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {

int TfLiteTypeGetSize(TfLiteType type) {
    switch (type) {
        case kTfLiteNoType: return 0;
        case kTfLiteFloat32: return sizeof(float);
        case kTfLiteInt32: return sizeof(int32_t);
        case kTfLiteUInt8: return sizeof(uint8_t);
        case kTfLiteInt8: return sizeof(int8_t);
        case kTfLiteInt64: return sizeof(int64_t);
        case kTfLiteBool: return sizeof(bool);
        case kTfLiteInt16: return sizeof(int16_t);
        case kTfLiteFloat16: return sizeof(int16_t);
        case kTfLiteFloat64: return sizeof(double);
        default: return 0;
    }
}

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
    return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

namespace {
inline TfLiteStatus QuantizeHelper(TfLiteContext* context, float scale,
                                  int32_t zero_point, float f, int32_t& q) {
    const float tmp = TfLiteRound(f / scale);
    const float max_int32 = static_cast<float>(std::numeric_limits<int32_t>::max());
    const float min_int32 = static_cast<float>(std::numeric_limits<int32_t>::min());
    TF_LITE_ENSURE(context, tmp <= max_int32);
    TF_LITE_ENSURE(context, tmp >= min_int32);
    q = zero_point + static_cast<int32_t>(tmp);
    return kTfLiteOk;
}

TfLiteStatus CalculateActivationRangeQuantizedImpl(
    TfLiteContext* context, TfLiteFusedActivation activation, int32_t qmin,
    int32_t qmax, TfLiteTensor* output, int32_t* act_min, int32_t* act_max) {
    const auto scale = output->params.scale;
    const auto zero_point = output->params.zero_point;

    int32_t tmp_q;
    if (activation == kTfLiteActRelu) {
        TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 0.0f, tmp_q));
        *act_min = std::max(qmin, tmp_q);
        *act_max = qmax;
    } else if (activation == kTfLiteActRelu6) {
        TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 0.0f, tmp_q));
        *act_min = std::max(qmin, tmp_q);
        TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 6.0f, tmp_q));
        *act_max = std::min(qmax, tmp_q);
    } else if (activation == kTfLiteActReluN1To1) {
        TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, -1.0f, tmp_q));
        *act_min = std::max(qmin, tmp_q);
        TF_LITE_ENSURE_OK(context, QuantizeHelper(context, scale, zero_point, 1.0f, tmp_q));
        *act_max = std::min(qmax, tmp_q);
    } else {
        *act_min = qmin;
        *act_max = qmax;
    }
    return kTfLiteOk;
}
} // namespace

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
    int32_t qmin = 0;
    int32_t qmax = 0;
    if (output->type == kTfLiteUInt8) {
        qmin = std::numeric_limits<uint8_t>::min();
        qmax = std::numeric_limits<uint8_t>::max();
    } else if (output->type == kTfLiteInt8) {
        qmin = std::numeric_limits<int8_t>::min();
        qmax = std::numeric_limits<int8_t>::max();
    } else if (output->type == kTfLiteInt16) {
        qmin = std::numeric_limits<int16_t>::min();
        qmax = std::numeric_limits<int16_t>::max();
    } else {
        TF_LITE_ENSURE(context, false);
    }

    return CalculateActivationRangeQuantizedImpl(context, activation, qmin, qmax,
                                                 output, act_min, act_max);
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier) {
    const double input_product_scale =
        static_cast<double>(input->params.scale * filter->params.scale);
    TF_LITE_ENSURE(context, input_product_scale >= 0);
    *multiplier = input_product_scale / static_cast<double>(output->params.scale);
    return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
    return GetQuantizedConvolutionMultipler(context, input, filter, output, multiplier);
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels) {
    (void)per_channel_multiplier;
    (void)per_channel_shift;
    (void)num_channels;

    if (multiplier && shift) {
        double real_multiplier = 0.0;
        TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
            context, input, filter, bias, output, &real_multiplier));
        int exponent;
        QuantizeMultiplier(real_multiplier, multiplier, &exponent);
        *shift = -exponent;
    }
    if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
        input->type == kTfLiteInt16) {
        TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
            context, activation, output, output_activation_min,
            output_activation_max));
    }
    return kTfLiteOk;
}

} // namespace tflite
