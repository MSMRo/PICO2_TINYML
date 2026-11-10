"""
Example 1: ECG Arrhythmia Classification & Post-Training Quantization (PTQ)
Pipeline:
  1. Synthetic ECG Generation with NeuroKit2 (Normal, Tachycardia, Bradycardia)
  2. 1D Convolutional Neural Network (1D-CNN) in TensorFlow/Keras
  3. Post-Training Quantization (Float32 -> Dynamic Range -> Full INT8)
  4. Accuracy & Size Benchmarking
  5. C-Header (.h) Export for Raspberry Pi Pico 2 / Arduino IDE
"""
import os
import sys
from pathlib import Path
import numpy as np

# Use tf_keras for full TFLite & TFMOT compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.signal_generator import generate_ecg_dataset
from utils.exporter import tflite_to_c_header, save_test_samples_header
from utils.tflite_eval import evaluate_tflite_model

CLASS_NAMES = ["Normal", "Tachycardia", "Bradycardia"]
SAMPLE_LEN = 187
SAMPLING_RATE = 250
ARDUINO_DIR = Path(__file__).resolve().parent.parent / "arduino" / "ecg_inference_pico2"


def build_ecg_model(input_shape=(SAMPLE_LEN, 1), num_classes=3):
    """
    Lightweight 1D-CNN optimized for ARM Cortex-M33 / RP2350 microcontrollers.
    Uses Separable Convolutions to reduce MACs and Flash usage.
    """
    inputs = keras.Input(shape=input_shape, name="ecg_input")
    
    # Block 1
    x = keras.layers.Conv1D(filters=8, kernel_size=7, strides=2, padding="same", activation="relu")(inputs)
    
    # Block 2: Separable Conv1D
    x = keras.layers.SeparableConv1D(filters=16, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    
    # Block 3: Separable Conv1D
    x = keras.layers.SeparableConv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    
    # Global Pooling + Dense Classifier
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="tiny_ecg_cnn")
    return model


def main():
    print("=" * 70)
    print("Tutorial 01: ECG Arrhythmia Classification & PTQ (TensorFlow)")
    print("Target Hardware: Raspberry Pi Pico 2 / Pico 2 W (RP2350 Cortex-M33)")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[Step 1/5] Generating synthetic ECG dataset with NeuroKit2...")
    X, y = generate_ecg_dataset(num_samples_per_class=400, sample_len=SAMPLE_LEN, sampling_rate=SAMPLING_RATE)
    X = np.expand_dims(X, axis=-1)  # (N, 187, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Build & Train Model
    print("\n[Step 2/5] Training 1D-CNN baseline model...")
    model = build_ecg_model(input_shape=(SAMPLE_LEN, 1), num_classes=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Baseline Model Test Accuracy: {acc * 100:.2f}%")

    # Save Keras Model
    keras_model_path = ARDUINO_DIR / "model_ecg.h5"
    model.save(str(keras_model_path))

    # 3. Post-Training Quantization (PTQ)
    print("\n[Step 3/5] Applying Post-Training Quantization (PTQ)...")
    
    # 3.1 Float32 TFLite (Baseline)
    converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter_float.convert()
    float_size = len(tflite_float)

    # 3.2 Full INT8 Quantization with Representative Dataset Calibration
    def representative_dataset_gen():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset_gen
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    tflite_int8 = converter_int8.convert()
    int8_size = len(tflite_int8)

    print(f"  Float32 Model Size: {float_size} bytes ({float_size / 1024:.2f} KB)")
    print(f"  INT8 Model Size:    {int8_size} bytes ({int8_size / 1024:.2f} KB)")
    print(f"  Compression Ratio:  {float_size / int8_size:.2f}x reduction!")

    # 4. Evaluation of Quantized Model
    print("\n[Step 4/5] Evaluating Quantized INT8 model accuracy...")
    eval_res = evaluate_tflite_model(tflite_int8, X_test, y_test)
    print(f"  Quantized INT8 Accuracy: {eval_res['accuracy'] * 100:.2f}%")
    print(f"  Input Scale: {eval_res['input_scale']:.6f} | Zero-Point: {eval_res['input_zero_point']}")

    # 5. Export for Pico 2 Arduino Sketch
    print("\n[Step 5/5] Exporting C header files to Arduino sketch folder...")
    ARDUINO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save .tflite binary
    with open(ARDUINO_DIR / "model_ecg_int8.tflite", "wb") as f:
        f.write(tflite_int8)

    # Save model_data.h
    tflite_to_c_header(
        tflite_int8,
        output_path=str(ARDUINO_DIR / "model_data.h"),
        array_name="g_model"
    )

    # Save test_samples.h for real-time verification in Serial Monitor
    save_test_samples_header(
        samples=X_test[:, :, 0],
        labels=y_test,
        output_path=str(ARDUINO_DIR / "test_samples.h"),
        class_names=CLASS_NAMES,
        samples_per_class=3
    )

    print("\n Training & Export Complete!")
    print(f"Open Arduino IDE and flash: {ARDUINO_DIR / 'ecg_inference_pico2.ino'}")


if __name__ == "__main__":
    main()
