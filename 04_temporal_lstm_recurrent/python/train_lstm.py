"""
04_temporal_lstm_recurrent: Recurrent Sequence Neural Network with LSTM
Architecture: Input(10, 1) -> LSTM(16) -> Dense(8, ReLU) -> Dense(2, Softmax)
Target: Time-Series / Sequence Anomaly Detection on Raspberry Pi Pico 2 (RP2350)
Quantization: Full INT8 PTQ for TFLM UnidirectionalSequenceLSTM Kernel
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

CLASS_NAMES = ["Sinusoidal_Harmonic", "Transient_Anomaly"]
SEQUENCE_LENGTH = 10
NUM_FEATURES = 1

def generate_synthetic_sequences(n_samples=2000):
    """
    Generates synthetic temporal sequences:
    Class 0: Smooth sinusoidal harmonic wave + noise
    Class 1: Sharp square/sawtooth step transient + noise
    """
    half = n_samples // 2
    t = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH, dtype=np.float32)

    # Class 0: Sinusoids with varying phase and frequency
    X0 = []
    for _ in range(half):
        phase = np.random.uniform(0, np.pi)
        freq = np.random.uniform(0.8, 1.2)
        noise = np.random.normal(0, 0.08, size=SEQUENCE_LENGTH)
        sig = np.sin(freq * t + phase) + noise
        X0.append(sig)

    # Class 1: Sharp pulse/step transients
    X1 = []
    for _ in range(half):
        pulse_pos = np.random.randint(2, SEQUENCE_LENGTH - 2)
        sig = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
        sig[pulse_pos : pulse_pos + 2] = np.random.uniform(1.2, 1.8)
        noise = np.random.normal(0, 0.08, size=SEQUENCE_LENGTH)
        sig = sig + noise
        X1.append(sig)

    X0 = np.array(X0, dtype=np.float32)
    X1 = np.array(X1, dtype=np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * half + [1] * half, dtype=np.int32)

    # Reshape to (N, SEQUENCE_LENGTH, 1)
    X = np.expand_dims(X, axis=-1)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def build_lstm_model():
    """Builds an unrolled LSTM recurrent network compatible with TFLM."""
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name="sequence_input"),
        keras.layers.LSTM(16, unroll=True, return_sequences=False, name="lstm_layer"),
        keras.layers.Dense(8, activation="relu", name="dense_hidden"),
        keras.layers.Dense(2, activation="softmax", name="output_probabilities")
    ], name="temporal_lstm_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def quantize_to_int8(model, representative_data):
    """Full INT8 PTQ for LSTM model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for sample in representative_data[:200]:
            yield [np.expand_dims(sample, axis=0)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()

def main():
    print("=" * 70)
    print("  04 - Recurrent Neural Network (LSTM) for Pico 2")
    print("=" * 70)

    # 1. Dataset Generation
    X, y = generate_synthetic_sequences()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Generated {len(X)} sequences of shape {X.shape[1:]}. Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. Build & Train
    model = build_lstm_model()
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

    # 3. Quantize to INT8
    print("\nQuantizing LSTM to Full INT8 for ARM Cortex-M33 (RP2350)...")
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

    tflite_raw_path = output_dir / "model_lstm_int8.tflite"
    with open(tflite_raw_path, "wb") as f:
        f.write(tflite_int8)

    print("\n" + "=" * 70)
    print("SUCCESS: Temporal LSTM Sequence model exported for Pico 2!")
    print(f"Header: {header_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
