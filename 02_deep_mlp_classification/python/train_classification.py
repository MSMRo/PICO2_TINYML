"""
02_deep_mlp_classification: Multi-Layer Neural Network for Multi-Class Classification
Architecture: Input(4) -> Dense(32, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(4, Softmax)
Target: Wearable / Multi-sensor State Classification on Raspberry Pi Pico 2 (RP2350)
Quantization: Full INT8 PTQ for ARM Cortex-M33
"""

import os
import sys
from pathlib import Path
import numpy as np

# Use tf_keras for full TFLite & TFMOT compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.exporter import tflite_to_c_header, save_test_samples_header

# Fix seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ["Resting", "Active", "High_Stress", "Fatigue"]

def generate_multiclass_data(n_samples=2400):
    """Generates synthetic 4-feature 4-class sensor data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.4,
        random_state=42
    )
    # Standardize to zero mean and unit variance
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-7)
    return X.astype(np.float32), y.astype(np.int32)

def build_deep_mlp_model():
    """Builds a deeper 5-layer classification network."""
    model = keras.Sequential([
        keras.layers.Input(shape=(4,), name="sensor_features"),
        keras.layers.Dense(32, activation="relu", name="dense_layer_1"),
        keras.layers.Dense(32, activation="relu", name="dense_layer_2"),
        keras.layers.Dense(16, activation="relu", name="dense_layer_3"),
        keras.layers.Dense(4, activation="softmax", name="class_probabilities")
    ], name="deep_mlp_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
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

    return converter.convert()

def main():
    print("=" * 70)
    print("  02 - Deep Multi-Layer Perceptron (MLP) Classifier for Pico 2")
    print("=" * 70)

    # 1. Dataset Generation
    X, y = generate_multiclass_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Dataset generated: {len(X)} samples across {len(CLASS_NAMES)} classes.")

    # 2. Build & Train
    model = build_deep_mlp_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Float32 Baseline] Test Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}")

    # 3. Quantize to INT8
    print("\nQuantizing Deep MLP to Full INT8 for ARM Cortex-M33 (RP2350)...")
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
        class_names=CLASS_NAMES,
        samples_per_class=2
    )

    tflite_raw_path = output_dir / "model_deep_mlp_int8.tflite"
    with open(tflite_raw_path, "wb") as f:
        f.write(tflite_int8)

    print("\n" + "=" * 70)
    print("SUCCESS: Deep MLP Classifier exported for Raspberry Pi Pico 2!")
    print(f"Header: {header_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
