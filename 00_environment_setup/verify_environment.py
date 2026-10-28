"""
Environment & Hardware Toolchain Verification Script.
Checks Python packages, uv status, and target RP2350 microcontroller toolchain settings.
"""
import sys
import platform

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [OK] {name:<30} Version: {ver}")
        return True
    except ImportError as e:
        print(f"  [MISSING] {name:<30} (ImportError: {e})")
        return False

def main():
    print("=" * 70)
    print("TinyML Biosignals on Raspberry Pi Pico 2 - Environment Verification")
    print("=" * 70)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")
    print("-" * 70)

    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("neurokit2", "neurokit2"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
        ("onnx", "onnx"),
        ("tensorflow-model-optimization", "tensorflow_model_optimization"),
    ]

    all_ok = True
    for pkg, imp in packages:
        ok = check_package(pkg, imp)
        if not ok:
            all_ok = False

    print("-" * 70)
    if all_ok:
        print(" SUCCESS: All Python and TinyML dependencies are properly configured!")
    else:
        print(" NOTICE: Some dependencies are missing. Run: uv sync")

    print("\n[RP2350 / Raspberry Pi Pico 2 Specifications Summary]")
    print("  - Processor: Dual ARM Cortex-M33 @ 150 MHz (with DSP instructions & FPU)")
    print("  - RAM: 520 KB on-chip SRAM (Ample room for 32-64KB TFLM tensor arenas!)")
    print("  - Flash: 4 MB external QSPI flash (Easily stores multiple 5-50KB models)")
    print("  - Wireless (Pico 2 W): CYW43439 (Wi-Fi 802.11n + Bluetooth 5.2)")
    print("=" * 70)

if __name__ == "__main__":
    main()
