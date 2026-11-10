"""
03_sine_regression: Non-Linear Function Approximation & Sensor Calibration
Target: Continuous Function Regression y = sin(x) * cos(2x)
Target Microcontroller: Raspberry Pi Pico 2 (RP2350 - Cortex-M33)
Quantization: Full INT8 PTQ with Scalar Continuous Dequantization
"""

import os
import sys
from pathlib import Path
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.exporter import tflite_to_c_header, save_test_samples_header

np.random.seed(42)
tf.random.set_seed(42)

def generate_nonlinear_curve(n_samples=2500):
    """Generates continuous regression points with realistic sensor noise."""
    x = np.random.uniform(-np.pi, np.pi, size=(n_samples, 1)).astype(np.float32)
    noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
    y = np.sin(x) * np.cos(2.0 * x) + noise
    return x, y

def build_regression_model():
    """Builds a continuous value predictor network."""
    model = keras.Sequential([
        keras.layers.Input(shape=(1,), name="input_x"),
        keras.layers.Dense(32, activation="relu", name="hidden_layer_1"),
        keras.layers.Dense(32, activation="relu", name="hidden_layer_2"),
        keras.layers.Dense(1, activation=None, name="output_y_continuous")
    ], name="nonlinear_regressor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mae"]
    )
    return model

def quantize_to_int8(model, representative_data):
    """Full INT8 PTQ for regression model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for sample in representative_data[:200]:
            yield [np.expand_dims(sample, axis=0)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()

def main():
    print("=" * 70)
    print("  03 - Non-Linear Regression Neural Network for Pico 2")
    print("=" * 70)

    # 1. Dataset Generation
    X, y = generate_nonlinear_curve()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Generated {len(X)} curve samples. Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. Build & Train
    model = build_regression_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=35,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Float32 Baseline] Test MSE: {test_mse:.5f} | Test MAE: {test_mae:.5f}")

    # 3. Quantize to INT8
    print("\nQuantizing to Full INT8 for ARM Cortex-M33 (RP2350)...")
    tflite_int8 = quantize_to_int8(model, X_train)
    print(f"Quantized TFLite Model Size: {len(tflite_int8)} bytes ({len(tflite_int8)/1024:.2f} KB)")

    # 4. Export C++ Headers
    output_dir = Path(__file__).resolve().parent.parent / "pico2"
    output_dir.mkdir(parents=True, exist_ok=True)

    header_path = output_dir / "model_data.h"
    tflite_to_c_header(tflite_int8, str(header_path), array_name="g_model")

    samples_header_path = output_dir / "test_samples.h"
    save_test_samples_header(
        samples=X_test,
        labels=y_test,
        output_path=str(samples_header_path),
        samples_per_class=2
    )

    tflite_raw_path = output_dir / "model_regression_int8.tflite"
    with open(tflite_raw_path, "wb") as f:
        f.write(tflite_int8)

    print("\n" + "=" * 70)
    print("SUCCESS: Regression model exported for Raspberry Pi Pico 2!")
    print(f"Header: {header_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
