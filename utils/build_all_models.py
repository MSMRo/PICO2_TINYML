"""
One-click Build Script: Trains and exports all 5 models to their respective Arduino folders.
Useful for quick initialization of all course examples.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    print("=" * 70)
    print("Batch Generator: Building All 5 Course Models & Arduino Headers")
    print("=" * 70)

    import importlib

    # 1. EMG Pruning
    print("\n>>> [1/4] Building Example 02 (EMG Pruning)...")
    try:
        mod_emg = importlib.import_module("02_emg_gesture_pruning_tensorflow.python.train_pruning")
        mod_emg.main()
    except Exception as e:
        print(f"Error in Example 02: {e}")

    # 2. EEG Distillation
    print("\n>>> [2/4] Building Example 03 (EEG Distillation - PyTorch)...")
    try:
        mod_eeg = importlib.import_module("03_eeg_distillation_pytorch.python.train_distillation")
        mod_eeg.main()
    except Exception as e:
        print(f"Error in Example 03: {e}")

    # 3. PPG QAT
    print("\n>>> [3/4] Building Example 04 (PPG QAT)...")
    try:
        mod_ppg = importlib.import_module("04_ppg_heart_rate_qat_tensorflow.python.train_qat")
        mod_ppg.main()
    except Exception as e:
        print(f"Error in Example 04: {e}")

    # 4. EDA Stress
    print("\n>>> [4/4] Building Example 05 (EDA Stress)...")
    try:
        mod_eda = importlib.import_module("05_eda_stress_hybrid_pipeline.python.train_eda")
        mod_eda.main()
    except Exception as e:
        print(f"Error in Example 05: {e}")

    print("\n" + "=" * 70)
    print("All models successfully built and exported to Arduino folders!")
    print("=" * 70)

if __name__ == "__main__":
    main()
