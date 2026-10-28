"""
Standalone Command-Line Tool: Convert any .tflite model to C Header (.h)
Usage:
    uv run 00_environment_setup/export_tflite_to_header.py <model.tflite> <output_header.h> [array_name]
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.exporter import tflite_to_c_header

def main():
    if len(sys.argv) < 3:
        print("Usage: uv run export_tflite_to_header.py <input.tflite> <output_header.h> [array_name]")
        sys.exit(1)

    tflite_file = sys.argv[1]
    output_h = sys.argv[2]
    array_name = sys.argv[3] if len(sys.argv) > 3 else "g_model"

    if not Path(tflite_file).exists():
        print(f"Error: Input file '{tflite_file}' does not exist.")
        sys.exit(1)

    bytes_written = tflite_to_c_header(tflite_file, output_h, array_name=array_name)
    print(f"Success! {bytes_written} bytes exported to {output_h}")

if __name__ == "__main__":
    main()
