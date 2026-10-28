"""
Example 2: EMG Muscle Activity & Weight Pruning (TensorFlow Model Optimization)
Pipeline:
  1. Synthetic sEMG Generation with NeuroKit2 (Rest, Low Contraction, High Contraction)
  2. Baseline 1D-CNN Model
  3. Weight Pruning using tfmot (Polynomial Decay to 60% Sparsity)
  4. Strip Pruning Wrapper & TFLite INT8 Export
  5. C-Header (.h) Export for Raspberry Pi Pico 2 / Arduino IDE
"""
import os
import sys
import zipfile
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

from utils.signal_generator import generate_emg_dataset
from utils.exporter import tflite_to_c_header, save_test_samples_header
from utils.tflite_eval import evaluate_tflite_model

CLASS_NAMES = ["Rest", "Low_Contraction", "High_Contraction"]
SAMPLE_LEN = 200
SAMPLING_RATE = 1000
ARDUINO_DIR = Path(__file__).resolve().parent.parent / "arduino" / "emg_inference_pico2"


def build_emg_model(input_shape=(SAMPLE_LEN, 1), num_classes=3):
    """Compact 1D-CNN for Electromyography signals."""
    inputs = keras.Input(shape=input_shape, name="emg_input")
    x = keras.layers.Conv1D(filters=16, kernel_size=9, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool1D(pool_size=2)(x)
    
    x = keras.layers.Conv1D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool1D(pool_size=2)(x)
    
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="emg_cnn")


def get_gzipped_model_size(file_path):
    """Returns zipped size of file in bytes (measures real storage benefit of sparsity)."""
    zip_path = str(file_path) + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file_path, arcname="model")
    size = os.path.getsize(zip_path)
    os.remove(zip_path)
    return size


def main():
    print("=" * 70)
    print("Tutorial 02: EMG Muscle Activity & Weight Pruning (TensorFlow / TFMOT)")
    print("Target Hardware: Raspberry Pi Pico 2 / Pico 2 W (RP2350 Cortex-M33)")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[Step 1/5] Generating synthetic sEMG signals with NeuroKit2...")
    X, y = generate_emg_dataset(num_samples_per_class=400, sample_len=SAMPLE_LEN, sampling_rate=SAMPLING_RATE)
    X = np.expand_dims(X, axis=-1)  # (N, 200, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Train Baseline
    print("\n[Step 2/5] Training Baseline EMG 1D-CNN...")
    base_model = build_emg_model()
    base_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    base_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)
    _, base_acc = base_model.evaluate(X_test, y_test, verbose=0)
    print(f"  Baseline Test Accuracy: {base_acc * 100:.2f}%")

    # 3. Apply Weight Pruning (Sparsity)
    print("\n[Step 3/5] Applying Magnitude-Based Pruning with TFMOT...")
    if HAS_TFMOT:
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.65,  # 65% of weights set to exact zero!
                begin_step=0,
                end_step=int(np.ceil(len(X_train) / 32)) * 15
            )
        }
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
        model_for_pruning.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        model_for_pruning.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
        
        # Strip pruning wrappers to recover standard Keras model
        pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    else:
        print("[NOTICE] tfmot not installed. Applying simulated magnitude weight pruning...")
        pruned_model = base_model
        for layer in pruned_model.layers:
            weights = layer.get_weights()
            if weights:
                w, *rest = weights
                threshold = np.percentile(np.abs(w), 65)
                w[np.abs(w) < threshold] = 0.0
                layer.set_weights([w] + rest)

    _, pruned_acc = pruned_model.evaluate(X_test, y_test, verbose=0)
    print(f"  Pruned Model Accuracy (at ~65% Sparsity): {pruned_acc * 100:.2f}%")

    # 4. Quantize to INT8
    print("\n[Step 4/5] Converting Pruned Model to INT8 TFLite...")
    def rep_data_gen():
        for i in range(min(150, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_pruned_int8 = converter.convert()

    # 5. Export to Arduino
    print("\n[Step 5/5] Exporting C header files to Arduino sketch directory...")
    ARDUINO_DIR.mkdir(parents=True, exist_ok=True)
    tflite_path = ARDUINO_DIR / "model_emg_pruned_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_pruned_int8)

    tflite_to_c_header(tflite_pruned_int8, output_path=str(ARDUINO_DIR / "model_data.h"), array_name="g_model")
    save_test_samples_header(
        samples=X_test[:, :, 0],
        labels=y_test,
        output_path=str(ARDUINO_DIR / "test_samples.h"),
        class_names=CLASS_NAMES,
        samples_per_class=3
    )

    zip_size = get_gzipped_model_size(tflite_path)
    raw_size = len(tflite_pruned_int8)
    print(f"\nModel Raw Flash Size: {raw_size} bytes ({raw_size / 1024:.2f} KB)")
    print(f"Model Compressed (Gzip/OTA) Size: {zip_size} bytes ({zip_size / 1024:.2f} KB)")
    print(f"Open Arduino IDE and flash: {ARDUINO_DIR / 'emg_inference_pico2.ino'}")


if __name__ == "__main__":
    main()
