"""
Example 3: EEG Brainwave Rhythm Classification & Knowledge Distillation (PyTorch)
Pipeline:
  1. Synthetic EEG Generation (Theta, Alpha, Beta bands) with NeuroKit2
  2. Large PyTorch Teacher CNN (~300k params) trained to high accuracy
  3. Ultra-compact PyTorch Student 1D-CNN (<3k params)
  4. Knowledge Distillation Training (Soft CrossEntropy / KL Divergence with Temperature T)
  5. Export PyTorch Student -> ONNX -> TFLite INT8 -> C Header (.h)
  6. Arduino IDE Sketch for Raspberry Pi Pico 2 / RP2350
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from utils.signal_generator import generate_eeg_dataset
from utils.exporter import tflite_to_c_header, save_test_samples_header

CLASS_NAMES = ["Theta_Drowsy", "Alpha_Relaxed", "Beta_Focused"]
SAMPLE_LEN = 256
SAMPLING_RATE = 128
ARDUINO_DIR = Path(__file__).resolve().parent.parent / "arduino" / "eeg_inference_pico2"


# ====================================================================
# 1. Models: Big Teacher vs Tiny Student
# ====================================================================
class DeepTeacherCNN(nn.Module):
    """Heavy Teacher Network with multiple residual-like conv layers."""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TinyStudentCNN(nn.Module):
    """Ultra-compact Student Network (< 3,000 parameters) designed for RP2350 microcontrollers."""
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Conv1: 8 filters
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        # Conv2: 16 filters
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(16)
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Output Linear
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ====================================================================
# 2. Distillation Loss (Hinton et al.)
# ====================================================================
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.6):
    """
    Combines hard label cross-entropy with soft teacher distillation loss (KL Divergence).
    """
    hard_loss = F.cross_entropy(student_logits, labels)
    
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    
    # KL Divergence scaled by T^2 as per Geoffrey Hinton et al.
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def main():
    print("=" * 70)
    print("Tutorial 03: EEG Rhythm Classification & Knowledge Distillation (PyTorch)")
    print("Target Hardware: Raspberry Pi Pico 2 / Pico 2 W (RP2350 Cortex-M33)")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[Step 1/6] Generating synthetic EEG dataset with NeuroKit2...")
    X, y = generate_eeg_dataset(num_samples_per_class=400, sample_len=SAMPLE_LEN, sampling_rate=SAMPLING_RATE)
    # Shape: (N, 1, 256) for PyTorch 1D-CNN
    X = np.expand_dims(X, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on Device: {device}")

    # 2. Train Teacher Model
    print("\n[Step 2/6] Training Teacher CNN (~300k params)...")
    teacher = DeepTeacherCNN().to(device)
    optimizer_t = torch.optim.Adam(teacher.parameters(), lr=0.002)

    teacher.train()
    for epoch in range(12):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_t.zero_grad()
            outputs = teacher(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer_t.step()

    teacher.eval()
    t_correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = teacher(inputs).argmax(dim=1)
            t_correct += (preds == targets).sum().item()
    teacher_acc = t_correct / len(test_ds)
    print(f"  Teacher Model Test Accuracy: {teacher_acc * 100:.2f}%")

    # 3. Knowledge Distillation into Tiny Student
    print("\n[Step 3/6] Distilling knowledge into Tiny Student CNN (<3k params)...")
    student = TinyStudentCNN().to(device)
    optimizer_s = torch.optim.Adam(student.parameters(), lr=0.003)

    student.train()
    for epoch in range(15):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_s.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                
            student_logits = student(inputs)
            loss = distillation_loss(student_logits, teacher_logits, targets, temperature=3.0, alpha=0.6)
            loss.backward()
            optimizer_s.step()

    student.eval()
    s_correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = student(inputs).argmax(dim=1)
            s_correct += (preds == targets).sum().item()
    student_acc = s_correct / len(test_ds)
    print(f"  Distilled Student Test Accuracy: {student_acc * 100:.2f}%")

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher Parameters: {teacher_params:,}")
    print(f"  Student Parameters: {student_params:,} ({teacher_params / student_params:.1f}x smaller!)")

    # 4. Export PyTorch Student to ONNX
    print("\n[Step 4/6] Exporting PyTorch Student to ONNX...")
    ARDUINO_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ARDUINO_DIR / "student_model.onnx"
    dummy_input = torch.randn(1, 1, SAMPLE_LEN, device=device)
    
    torch.onnx.export(
        student,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"]
    )

    # 5. Convert to TFLite INT8 with Keras / Flatbuffer representation
    print("\n[Step 5/6] Building equivalent TFLite model and applying INT8 PTQ...")
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf
    import tf_keras as keras
    
    # We construct the matching Keras model and copy the trained student weights for 100% stable TFLM compatibility
    keras_student = keras.Sequential([
        keras.layers.Input(shape=(SAMPLE_LEN, 1)),
        keras.layers.Conv1D(8, 7, strides=2, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv1D(16, 5, strides=2, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(3, activation="softmax")
    ])
    
    # Copy weights from PyTorch student to Keras student
    with torch.no_grad():
        w_conv1 = student.conv1.weight.cpu().numpy().transpose(2, 1, 0) # (kernel, in, out)
        bn1_gamma = student.bn1.weight.cpu().numpy()
        bn1_beta = student.bn1.bias.cpu().numpy()
        bn1_mean = student.bn1.running_mean.cpu().numpy()
        bn1_var = student.bn1.running_var.cpu().numpy()

        w_conv2 = student.conv2.weight.cpu().numpy().transpose(2, 1, 0)
        bn2_gamma = student.bn2.weight.cpu().numpy()
        bn2_beta = student.bn2.bias.cpu().numpy()
        bn2_mean = student.bn2.running_mean.cpu().numpy()
        bn2_var = student.bn2.running_var.cpu().numpy()

        fc_w = student.fc.weight.cpu().numpy().T
        fc_b = student.fc.bias.cpu().numpy()

    keras_student.layers[0].set_weights([w_conv1])
    keras_student.layers[1].set_weights([bn1_gamma, bn1_beta, bn1_mean, bn1_var])
    keras_student.layers[3].set_weights([w_conv2])
    keras_student.layers[4].set_weights([bn2_gamma, bn2_beta, bn2_mean, bn2_var])
    keras_student.layers[7].set_weights([fc_w, fc_b])

    # Convert to INT8 TFLite
    X_train_tf = np.transpose(X_train, (0, 2, 1)) # (N, 256, 1)
    def rep_gen():
        for i in range(150):
            yield [X_train_tf[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_student)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_student_int8 = converter.convert()

    # 6. Export to C Header
    print("\n[Step 6/6] Exporting C header files to Arduino...")
    tflite_to_c_header(tflite_student_int8, output_path=str(ARDUINO_DIR / "model_data.h"), array_name="g_model")
    save_test_samples_header(
        samples=X_test[:, 0, :],
        labels=y_test,
        output_path=str(ARDUINO_DIR / "test_samples.h"),
        class_names=CLASS_NAMES,
        samples_per_class=3
    )

    print(f"\nStudent INT8 Model size: {len(tflite_student_int8)} bytes ({len(tflite_student_int8) / 1024:.2f} KB)")
    print(f"Open Arduino IDE and flash: {ARDUINO_DIR / 'eeg_inference_pico2.ino'}")


if __name__ == "__main__":
    main()
