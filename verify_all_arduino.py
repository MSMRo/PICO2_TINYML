"""
Automated Verification Script for Arduino TinyML Sketches on Raspberry Pi Pico / Pico 2
Compiles each of the 5 biosignal inference sketches using arduino-cli.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CLI_PATH = ROOT_DIR / "bin" / "arduino-cli.exe"
LIBRARIES_DIR = ROOT_DIR / "libraries"

SKETCHES = [
    (
        "01_ecg_arrhythmia_ptq_tensorflow",
        "01_ecg_arrhythmia_ptq_tensorflow/arduino/ecg_inference_pico2",
    ),
    (
        "02_emg_gesture_pruning_tensorflow",
        "02_emg_gesture_pruning_tensorflow/arduino/emg_inference_pico2",
    ),
    (
        "03_eeg_distillation_pytorch",
        "03_eeg_distillation_pytorch/arduino/eeg_inference_pico2",
    ),
    (
        "04_ppg_heart_rate_qat_tensorflow",
        "04_ppg_heart_rate_qat_tensorflow/arduino/ppg_inference_pico2",
    ),
    (
        "05_eda_stress_hybrid_pipeline",
        "05_eda_stress_hybrid_pipeline/arduino/eda_inference_pico2",
    ),
]

BOARDS = [
    ("Raspberry Pi Pico 2 (RP2350 - Cortex-M33)", "rp2040:rp2040:rpipico2"),
    ("Raspberry Pi Pico (RP2040 - Cortex-M0+)", "rp2040:rp2040:rpipico"),
]


def test_compile(sketch_dir: Path, fqbn: str) -> tuple[bool, str, float]:
    cmd = [
        str(CLI_PATH),
        "compile",
        "-j",
        "0",
        "--warnings",
        "none",
        "--fqbn",
        fqbn,
        "--libraries",
        str(LIBRARIES_DIR),
        str(sketch_dir),
    ]
    t0 = time.time()
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT_DIR),
    )
    elapsed = time.time() - t0
    success = res.returncode == 0
    lines = [
        ln.strip()
        for ln in (res.stdout + res.stderr).splitlines()
        if "Sketch usa" in ln
        or "Sketch uses" in ln
        or "Global variables" in ln
        or "variables Globales" in ln
    ]
    summary = " | ".join(lines) if lines else ("PASS" if success else res.stderr[-300:])
    return success, summary, elapsed


def main():
    if not CLI_PATH.exists():
        print(f"[ERROR] arduino-cli not found at {CLI_PATH}")
        sys.exit(1)

    print("=" * 80)
    print("  TINYML BIOSIGNALS - ARDUINO SKETCH AUTOMATED VERIFICATION SUITE")
    print(f"  Tool: {CLI_PATH}")
    print(f"  Libraries: {LIBRARIES_DIR}")
    print("=" * 80)

    results = []
    all_passed = True

    for board_name, fqbn in BOARDS:
        print(f"\n[TARGET] {board_name}")
        print("-" * 80)
        for name, rel_path in SKETCHES:
            sketch_path = ROOT_DIR / rel_path
            print(f"  -> Testing {name} ... ", end="", flush=True)
            ok, msg, dur = test_compile(sketch_path, fqbn)
            if ok:
                print(f"OK [PASS] ({dur:.1f}s)")
                if msg:
                    print(f"     {msg}")
            else:
                print(f"FAILED [ERROR] ({dur:.1f}s)")
                print(f"     {msg}")
                all_passed = False
            results.append((board_name, name, ok, dur, msg))

    print("\n" + "=" * 80)
    print("  SUMMARY MATRIX")
    print("=" * 80)
    print(f"{'Target Board':<42} | {'Example':<36} | {'Status':<6} | {'Time'}")
    print("-" * 95)
    for b_name, ex_name, ok, dur, _ in results:
        status_str = "PASS" if ok else "FAIL"
        print(f"{b_name:<42} | {ex_name:<36} | {status_str:<6} | {dur:.1f}s")
    print("=" * 80)

    if all_passed:
        print(">>> 10/10 TARGETS COMPILED AND VALIDATED WITH ZERO ERRORS! <<<")
    else:
        print(">>> SOME TARGETS FAILED COMPILATION <<<")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
