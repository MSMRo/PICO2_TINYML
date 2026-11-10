"""
C Header Exporter for TensorFlow Lite Micro models and Test Samples.
Produces clean, standard C++17 compile-ready .h files for Raspberry Pi Pico 2 / RP2350 (Cortex-M33) SDK.
"""
from pathlib import Path
from typing import List, Optional
import numpy as np


def tflite_to_c_header(
    tflite_content_or_path,
    output_path: str,
    array_name: str = "g_model",
    bytes_per_line: int = 12
) -> int:
    """
    Converts a .tflite byte array or file path into a C++ header file (.h)
    aligned to 16 bytes for ARM Cortex-M33 (RP2350) / CMSIS-NN SIMD acceleration.
    """
    if isinstance(tflite_content_or_path, (str, Path)):
        with open(tflite_content_or_path, "rb") as f:
            data = f.read()
    elif isinstance(tflite_content_or_path, (bytes, bytearray)):
        data = tflite_content_or_path
    else:
        raise ValueError("Expected bytes or path to .tflite file")

    length = len(data)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    header_guard = f"{out_file.stem.upper()}_H_"

    lines = []
    lines.append(f"// Generated for Raspberry Pi Pico 2 C++ SDK (RP2350 - ARM Cortex-M33)")
    lines.append(f"// Model size: {length} bytes ({length / 1024.0:.2f} KB)")
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}\n")
    lines.append("#include <cstdint>")
    lines.append("#include <cstddef>\n")
    lines.append("// Aligned to 16 bytes for ARM Cortex-M33 / CMSIS-NN SIMD vectorization")
    lines.append(f"alignas(16) const unsigned char {array_name}[] = {{")

    for i in range(0, length, bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + bytes_per_line < length:
            hex_bytes += ","
        lines.append(f"  {hex_bytes}")

    lines.append("};\n")
    lines.append(f"const unsigned int {array_name}_len = {length};\n")
    lines.append(f"#endif // {header_guard}\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Exporter] Successfully exported C Header: {out_file} ({length} bytes)")
    return length


def save_test_samples_header(
    samples: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None,
    samples_per_class: int = 2
) -> None:
    """
    Selects balanced representative test samples and exports them to test_samples.h
    for in-system verification on Raspberry Pi Pico 2 via USB/UART stdio.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    samples = np.array(samples)
    labels = np.array(labels)

    if labels.ndim > 1 and labels.shape[1] == 1:
        labels = labels.ravel()

    # Determine unique classes if discrete, or sample evenly if continuous (regression)
    if np.issubdtype(labels.dtype, np.floating):
        # Continuous target (regression)
        step = max(1, len(samples) // (samples_per_class * 4))
        chosen_indices = np.arange(0, len(samples), step)[:samples_per_class * 4]
    else:
        unique_classes = np.unique(labels)
        chosen_indices = []
        for c in unique_classes:
            c_idxs = np.where(labels == c)[0]
            chosen = c_idxs[:samples_per_class]
            chosen_indices.extend(chosen)
        chosen_indices = np.array(chosen_indices)

    selected_samples = samples[chosen_indices]
    selected_labels = labels[chosen_indices]

    num_samples = selected_samples.shape[0]
    sample_flat_len = int(np.prod(selected_samples.shape[1:]))
    header_guard = f"{out_file.stem.upper()}_H_"

    lines = []
    lines.append(f"// Generated Test Samples for Raspberry Pi Pico 2 Verification")
    lines.append(f"#ifndef {header_guard}")
    lines.append(f"#define {header_guard}\n")
    lines.append("#include <cstdint>")
    lines.append("#include <cstddef>\n")
    lines.append(f"const int NUM_TEST_SAMPLES = {num_samples};")
    lines.append(f"const int SAMPLE_LENGTH = {sample_flat_len};\n")

    # Class names if available
    if class_names is not None:
        names_str = ", ".join(f'"{name}"' for name in class_names)
        lines.append(f"const char* const CLASS_NAMES[] = {{ {names_str} }};")
        lines.append(f"const int NUM_CLASSES = {len(class_names)};\n")

    # Ground truth labels/targets
    if np.issubdtype(selected_labels.dtype, np.floating):
        labels_str = ", ".join(f"{float(lbl):.6f}f" for lbl in selected_labels)
        lines.append(f"const float TEST_TARGETS[NUM_TEST_SAMPLES] = {{ {labels_str} }};\n")
    else:
        labels_str = ", ".join(str(int(lbl)) for lbl in selected_labels)
        lines.append(f"const int TEST_LABELS[NUM_TEST_SAMPLES] = {{ {labels_str} }};\n")

    # Data array
    lines.append(f"const float TEST_SAMPLES[NUM_TEST_SAMPLES][SAMPLE_LENGTH] = {{")
    for s_idx, sample in enumerate(selected_samples):
        floats = ", ".join(f"{float(val):.6f}f" for val in sample.flatten())
        comma = "," if s_idx < num_samples - 1 else ""
        lbl_info = f"{selected_labels[s_idx]:.4f}" if np.issubdtype(selected_labels.dtype, np.floating) else f"{int(selected_labels[s_idx])}"
        lines.append(f"  // Sample {s_idx} (True Target/Class: {lbl_info})")
        lines.append(f"  {{ {floats} }}{comma}")
    lines.append("};\n")
    lines.append(f"#endif // {header_guard}\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Exporter] Saved {num_samples} test samples to {out_file}")


def quantize_input_for_c(sample: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """
    Quantizes a floating point input to int8 using the model's input quantization parameters:
    q = round(x / scale) + zero_point, clipped to [-128, 127]
    """
    q = np.round(sample / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)
