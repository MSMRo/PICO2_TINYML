"""
Example 4: PPG Pulse Contour & Quantization-Aware Training (QAT)
Pipeline:
  1. Synthetic PPG Waveform Generation with NeuroKit2 (Normal, Ectopic, Motion Artifact)
  2. Baseline Keras 1D-CNN
  3. Quantization-Aware Training (QAT) with tfmot
  4. Fake-quantization simulation and Straight-Through Estimator (STE)
  5. Full INT8 TFLite Conversion and C-Header (.h) Export
  6. Arduino IDE Sketch for Raspberry Pi Pico 2 / RP2350
"""
import os
import sys
from pathlib import Path
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split

try:
    import tensorflow_model_optimization as tfmot
    HAS_TFMOT = True
except ImportError:
    HAS_TFMOT = False

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.signal_generator import generate_ppg_dataset
from utils.exporter import tflite_to_c_header, save_test_samples_header
from utils.tflite_eval import evaluate_tflite_model

CLASS_NAMES = ["Normal_Pulse", "Ectopic_Arrhythmic", "Motion_Artifact"]
SAMPLE_LEN = 150
SAMPLING_RATE = 100
ARDUINO_DIR = Path(__file__).resolve().parent.parent / "arduino" / "ppg_inference_pico2"


def build_ppg_model(input_shape=(SAMPLE_LEN, 1, 1), num_classes=3):
    """Clean 2D-as-1D CNN for optical Photoplethysmogram waveforms with full TFMOT QAT support."""
    inputs = keras.Input(shape=input_shape, name="ppg_input")
    x = keras.layers.Conv2D(filters=16, kernel_size=(7, 1), strides=(2, 1), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool2D(pool_size=(2, 1))(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(5, 1), strides=(2, 1), padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(24, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="ppg_classes")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="tiny_ppg_cnn")


def main():
    print("=" * 70)
    print("Tutorial 04: PPG Pulse Analysis & Quantization-Aware Training (QAT)")
    print("Target Hardware: Raspberry Pi Pico 2 / Pico 2 W (RP2350 Cortex-M33)")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[Step 1/5] Generating synthetic PPG pulse dataset with NeuroKit2...")
    X, y = generate_ppg_dataset(num_samples_per_class=400, sample_len=SAMPLE_LEN, sampling_rate=SAMPLING_RATE)
    X = X[:, :, None, None]  # (N, 150, 1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Train Float32 Baseline
    print("\n[Step 2/5] Training Baseline Float32 PPG Model...")
    base_model = build_ppg_model()
    base_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    base_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32, verbose=1)
    _, base_acc = base_model.evaluate(X_test, y_test, verbose=0)
    print(f"  Baseline Accuracy: {base_acc * 100:.2f}%")

    # 3. Quantization-Aware Training (QAT)
    print("\n[Step 3/5] Applying Quantization-Aware Training (QAT)...")
    if HAS_TFMOT:
        quant_aware_model = tfmot.quantization.keras.quantize_model(base_model)
        quant_aware_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        quant_aware_model.summary()
        
        # Fine-tune with fake-quantization operators inserted
        quant_aware_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        _, qat_acc = quant_aware_model.evaluate(X_test, y_test, verbose=0)
        print(f"  QAT Fine-tuned Accuracy: {qat_acc * 100:.2f}%")
        model_to_convert = quant_aware_model
    else:
        print("[NOTICE] tfmot not available. Using PTQ fallback...")
        model_to_convert = base_model

    # 4. Convert QAT Model to INT8 TFLite
    print("\n[Step 4/5] Converting QAT Model to Full INT8 TFLite FlatBuffer...")
    def rep_gen():
        for i in range(150):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_qat_int8 = converter.convert()

    # 5. Export to Arduino
    print("\n[Step 5/5] Exporting C Header to Arduino sketch directory...")
    ARDUINO_DIR.mkdir(parents=True, exist_ok=True)
    tflite_to_c_header(tflite_qat_int8, output_path=str(ARDUINO_DIR / "model_data.h"), array_name="g_model")
    save_test_samples_header(
        samples=X_test[:, :, 0, 0],
        labels=y_test,
        output_path=str(ARDUINO_DIR / "test_samples.h"),
        class_names=CLASS_NAMES,
        samples_per_class=3
    )

    print(f"\nFinal QAT INT8 Model Size: {len(tflite_qat_int8)} bytes ({len(tflite_qat_int8) / 1024:.2f} KB)")
    print(f"Open Arduino IDE and flash: {ARDUINO_DIR / 'ppg_inference_pico2.ino'}")


if __name__ == "__main__":
    main()
