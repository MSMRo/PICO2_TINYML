"""
PyTorch to TensorFlow Lite conversion utility.
Converts a trained PyTorch model -> ONNX -> TensorFlow/TFLite with INT8 quantization support.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def export_pytorch_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    input_names: list = ["input"],
    output_names: list = ["output"]
):
    """Exports a PyTorch model to ONNX format."""
    model.eval()
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None  # Fixed batch size 1 for microcontrollers
        )
    print(f"[PyTorch2ONNX] Model exported to {onnx_path}")


def convert_onnx_to_tflite(
    onnx_path: str,
    output_tflite_path: str,
    representative_data: np.ndarray = None,
    quantize_int8: bool = True
):
    """
    Converts ONNX to TFLite using onnx2tf or onnx_tf, then applies TFLite Converter.
    """
    import tensorflow as tf
    import onnx
    from onnx2tf import convert

    output_dir = Path(output_tflite_path).parent / "onnx2tf_saved_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ONNX2TF] Converting ONNX model to TensorFlow SavedModel...")
    convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=str(output_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(output_dir))

    if quantize_int8 and representative_data is not None:
        def representative_dataset_gen():
            for i in range(min(150, len(representative_data))):
                sample = representative_data[i:i+1].astype(np.float32)
                yield [sample]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("[TFLite] Applying Full INT8 Quantization...")
    elif quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[TFLite] Applying Dynamic Range Quantization...")

    tflite_model = converter.convert()
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"[TFLite] Saved model to {output_tflite_path} ({len(tflite_model)} bytes)")
    return tflite_model
