# ==============================================================================
# TensorFlow Lite for Microcontrollers (TFLM) - Raspberry Pi Pico 2 / Pico Target
# ==============================================================================

cmake_minimum_required(VERSION 3.13)

if (TARGET pico_tflm)
    return()
endif()

add_library(pico_tflm STATIC)

set(TFLM_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../TFLM_Pico/src")

target_sources(pico_tflm PRIVATE
    # Pico platform bindings & Kernel Utilities
    ${CMAKE_CURRENT_LIST_DIR}/utility_pico.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_util_pico.cpp

    # TensorFlow Lite Micro Core
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/all_ops_resolver.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/debug_log.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/fake_micro_context.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/flatbuffer_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/memory_helpers.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_allocation_info.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_allocator.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_context.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_graph.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_interpreter.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_interpreter_context.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_interpreter_graph.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_log.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_op_resolver.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_profiler.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_resource_variable.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_string.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_time.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/micro_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/mock_micro_graph.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/recording_micro_allocator.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/system_setup.cpp

    # Allocators and Planners
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/memory_planner/linear_memory_planner.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cpp

    # TFLite Bridge
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/tflite_bridge/op_resolver_bridge.cpp

    # Schema & API
    ${TFLM_BASE_DIR}/tensorflow/lite/schema/schema_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/core/api/error_reporter.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/core/api/flatbuffer_conversions.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/core/api/op_resolver.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/core/api/tensor_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/core/c/common.cpp

    # Kernels Internal
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/portable_tensor_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/quantization_util.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/runtime_shape.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/tensor_ctypes.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/tensor_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/reference/comparisons.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cpp

    # CMSIS-NN Optimized Kernels for ARM Cortex-M
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/add.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/conv.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/mul.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/pooling.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/softmax.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cmsis_nn/svdf.cpp

    # Standard Micro Reference Kernels
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/activations.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/activations_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/add_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/add_n.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/arg_min_max.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/assign_variable.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/batch_matmul.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/batch_to_space_nd.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/broadcast_args.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/broadcast_to.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/call_once.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cast.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/ceil.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/circular_buffer.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/circular_buffer_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/comparisons.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/concatenation.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/conv_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/cumsum.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/depth_to_space.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/depthwise_conv_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/dequantize.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/dequantize_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/detection_postprocess.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/div.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/elementwise.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/elu.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/embedding_lookup.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/ethosu.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/exp.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/expand_dims.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/fill.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/floor.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/floor_div.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/floor_mod.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/fully_connected_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/gather.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/gather_nd.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/hard_swish.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/hard_swish_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/if.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/kernel_runner.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/kernel_util.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/l2norm.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/l2_pool_2d.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/leaky_relu.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/leaky_relu_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/logical.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/logical_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/logistic.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/logistic_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/log_softmax.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/lstm_eval.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/lstm_eval_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/maximum_minimum.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/micro_tensor_utils.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/mirror_pad.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/mul_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/neg.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/pack.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/pad.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/pooling_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/prelu.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/prelu_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/quantize.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/quantize_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/read_variable.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/reduce.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/reduce_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/reshape.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/reshape_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/resize_bilinear.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/round.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/select.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/shape.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/slice.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/softmax_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/space_to_batch_nd.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/space_to_depth.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/split.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/split_v.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/squared_difference.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/squeeze.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/strided_slice.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/strided_slice_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/sub.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/sub_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/svdf_common.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/tanh.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/transpose.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/transpose_conv.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/unpack.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/var_handle.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/while.cpp
    ${TFLM_BASE_DIR}/tensorflow/lite/micro/kernels/zeros_like.cpp

    # CMSIS-NN Library Source Functions
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ActivationFunctions/arm_relu6_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ActivationFunctions/arm_relu_q15.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ActivationFunctions/arm_relu_q7.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/BasicMathFunctions/arm_elementwise_add_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/BasicMathFunctions/arm_elementwise_add_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/BasicMathFunctions/arm_elementwise_mul_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/BasicMathFunctions/arm_elementwise_mul_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConcatenationFunctions/arm_concatenation_s8_w.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConcatenationFunctions/arm_concatenation_s8_x.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConcatenationFunctions/arm_concatenation_s8_y.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConcatenationFunctions/arm_concatenation_s8_z.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_fast_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nntables.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mul_kernel_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/PoolingFunctions/arm_avgpool_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/PoolingFunctions/arm_avgpool_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/PoolingFunctions/arm_max_pool_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/PoolingFunctions/arm_max_pool_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/ReshapeFunctions/arm_reshape_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_s8_s16.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_u8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SVDFunctions/arm_svdf_s8.c
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Source/SVDFunctions/arm_svdf_state_s16_s8.c
)

target_include_directories(pico_tflm PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${TFLM_BASE_DIR}
    ${TFLM_BASE_DIR}/third_party/flatbuffers/include
    ${TFLM_BASE_DIR}/third_party/gemmlowp
    ${TFLM_BASE_DIR}/third_party/kissfft
    ${TFLM_BASE_DIR}/third_party/ruy
    ${TFLM_BASE_DIR}/third_party/cmsis_nn/Include
    ${TFLM_BASE_DIR}/third_party/cmsis/CMSIS/Core/Include
)

target_compile_definitions(pico_tflm PUBLIC
    ARDUINO=1
    ARDUINO_EXCLUDE_CODE=1
    TF_LITE_STATIC_MEMORY=1
    TF_LITE_DISABLE_X86_NEON=1
    CMSIS_NN=1
)

target_compile_options(pico_tflm PRIVATE
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -Wno-type-limits
)

target_link_libraries(pico_tflm PUBLIC
    pico_stdlib
    hardware_timer
)
