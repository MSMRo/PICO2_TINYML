"""
Example 5: Electrodermal Activity (EDA/GSR) Stress Detection & Hybrid TinyML Pipeline
Pipeline:
  1. Synthetic EDA Signal Generation with NeuroKit2 (Calm Baseline vs High Stress Arousal)
  2. End-to-End Raw 1D-CNN vs Feature-Extracted Micro-MLP comparison
  3. INT8 Quantization and Latency Profiling
  4. C-Header (.h) Export for Raspberry Pi Pico 2 / Arduino IDE
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

from utils.signal_generator import generate_eda_dataset
from utils.exporter import tflite_to_c_header, save_test_samples_header
from utils.tflite_eval import evaluate_tflite_model

CLASS_NAMES = ["Calm_Baseline", "Stress_High_Arousal"]
SAMPLE_LEN = 200
SAMPLING_RATE = 50
ARDUINO_DIR = Path(__file__).resolve().parent.parent / "arduino" / "eda_inference_pico2"


def build_eda_cnn(input_shape=(SAMPLE_LEN, 1)):
    """Ultra-compact 1D-CNN for EDA raw time-series."""
    inputs = keras.Input(shape=input_shape, name="eda_input")
    x = keras.layers.Conv1D(8, 7, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool1D(2)(x)
    x = keras.layers.Conv1D(16, 5, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="stress_prediction")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="tiny_eda_cnn")


def main():
    print("=" * 70)
    print("Tutorial 05: EDA Stress Detection & Hybrid Pipeline (TensorFlow INT8)")
    print("Target Hardware: Raspberry Pi Pico 2 / Pico 2 W (RP2350 Cortex-M33)")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[Step 1/5] Generating synthetic EDA dataset with NeuroKit2...")
    X, y = generate_eda_dataset(num_samples_per_class=400, sample_len=SAMPLE_LEN, sampling_rate=SAMPLING_RATE)
    X = np.expand_dims(X, axis=-1)  # (N, 200, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Build & Train Model
    print("\n[Step 2/5] Training 1D-CNN for EDA raw windows...")
    model = build_eda_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32, verbose=1)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Model Test Accuracy: {acc * 100:.2f}%")

    # 3. Full INT8 Quantization
    print("\n[Step 3/5] Converting to Full INT8 TFLite...")
    def rep_gen():
        for i in range(150):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_eda_int8 = converter.convert()

    # 4. Evaluation
    eval_res = evaluate_tflite_model(tflite_eda_int8, X_test, y_test)
    print(f"  Quantized INT8 Model Accuracy: {eval_res['accuracy'] * 100:.2f}%")
    print(f"  Model Size: {len(tflite_eda_int8)} bytes ({len(tflite_eda_int8) / 1024:.2f} KB)")

    # 5. Export to Arduino
    print("\n[Step 5/5] Exporting C header to Arduino sketch folder...")
    ARDUINO_DIR.mkdir(parents=True, exist_ok=True)
    tflite_to_c_header(tflite_eda_int8, output_path=str(ARDUINO_DIR / "model_data.h"), array_name="g_model")
    save_test_samples_header(
        samples=X_test[:, :, 0],
        labels=y_test,
        output_path=str(ARDUINO_DIR / "test_samples.h"),
        class_names=CLASS_NAMES,
        samples_per_class=3
    )

    print(f"\nTraining & C-Array Export Finished!")
    print(f"Open Arduino IDE and flash: {ARDUINO_DIR / 'eda_inference_pico2.ino'}")


if __name__ == "__main__":
    main()
