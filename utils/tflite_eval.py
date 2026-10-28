"""
Evaluation utilities for TensorFlow Lite models (Float32 vs Quantized INT8).
"""
import time
import numpy as np
import tensorflow as tf


def evaluate_tflite_model(tflite_model_path_or_bytes, X_test, y_test):
    """
    Runs inference on X_test using the TFLite Interpreter and computes accuracy,
    quantization parameters, and average latency per inference.
    """
    if isinstance(tflite_model_path_or_bytes, (str, bytes, bytearray)):
        if isinstance(tflite_model_path_or_bytes, str):
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path_or_bytes)
        else:
            interpreter = tf.lite.Interpreter(model_content=tflite_model_path_or_bytes)
    else:
        raise ValueError("Invalid tflite model reference")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))
    is_int8 = input_details["dtype"] == np.int8

    correct = 0
    total_time_us = 0.0

    predictions = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1]

        if is_int8:
            sample_prepared = np.round(sample / input_scale + input_zero_point).clip(-128, 127).astype(np.int8)
        else:
            sample_prepared = sample.astype(np.float32)

        interpreter.set_tensor(input_details["index"], sample_prepared)

        start = time.perf_counter()
        interpreter.invoke()
        dur_us = (time.perf_counter() - start) * 1e6
        total_time_us += dur_us

        output = interpreter.get_tensor(output_details["index"])

        if is_int8 and output_scale > 0:
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        pred_class = np.argmax(output)
        predictions.append(pred_class)
        if pred_class == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    avg_latency_us = total_time_us / len(X_test)

    return {
        "accuracy": accuracy,
        "is_int8": is_int8,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "avg_latency_us": avg_latency_us,
        "predictions": np.array(predictions)
    }
