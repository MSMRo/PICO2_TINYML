"""
01_basic_3layer_dense: Training and Export of a Minimal 3-Layer Neural Network
Architecture: Input(2) -> Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(2, Softmax)
Quantization: Full INT8 Post-Training Quantization (PTQ) for ARM Cortex-M33 (RP2350)
Target: Raspberry Pi Pico 2 C++ SDK
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure Keras 2 / tf_keras compatibility with TFLite
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.exporter import tflite_to_c_header, save_test_samples_header

# Fix seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_data(n_samples=2000):
    """Generates a non-linear 2D dataset (two interleaving half circles)."""
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    # Normalize features to [-1.0, 1.0]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float32), y.astype(np.int32)

def build_3layer_model():
    """Builds a classic 3-layer neural network."""
    model = keras.Sequential([
        keras.layers.Input(shape=(2,), name="input_coords"),
        keras.layers.Dense(16, activation="relu", name="hidden_layer_1"),
        keras.layers.Dense(8, activation="relu", name="hidden_layer_2"),
        keras.layers.Dense(2, activation="softmax", name="output_probabilities")
    ], name="minimal_3layer_dense")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def quantize_to_int8(model, representative_data):
    """Performs full integer (INT8) Post-Training Quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset_gen():
        for sample in representative_data[:200]:
            yield [np.expand_dims(sample, axis=0)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant = converter.convert()
    return tflite_quant

def main():
    print("=" * 70)
    print("  01 - Minimal 3-Layer Neural Network for Raspberry Pi Pico 2")
    print("=" * 70)

    # 1. Generate Dataset
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Generated {len(X)} samples. Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. Build & Train Model
    model = build_3layer_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Float32 Baseline] Test Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}")

    # 3. Quantize to Full INT8
    print("\nQuantizing to Full INT8 for ARM Cortex-M33 (RP2350)...")
    tflite_int8 = quantize_to_int8(model, X_train)
    print(f"Quantized TFLite Model Size: {len(tflite_int8)} bytes ({len(tflite_int8)/1024:.2f} KB)")

    # 4. Export C++ Headers for Pico 2
    output_dir = Path(__file__).resolve().parent.parent / "pico2"
    output_dir.mkdir(parents=True, exist_ok=True)

    header_path = output_dir / "model_data.h"
    tflite_to_c_header(tflite_int8, str(header_path), array_name="g_model")

    samples_header_path = output_dir / "test_samples.h"
    save_test_samples_header(
        samples=X_test,
        labels=y_test,
        output_path=str(samples_header_path),
        class_names=["Region_A", "Region_B"],
        samples_per_class=3
    )

    # Also save raw .tflite for inspection
    tflite_raw_path = output_dir / "model_3layer_dense_int8.tflite"
    with open(tflite_raw_path, "wb") as f:
        f.write(tflite_int8)

    print("\n" + "=" * 70)
    print("SUCCESS: 3-Layer Dense model & test samples generated for Pico 2!")
    print(f"Headers: {header_path} and {samples_header_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
