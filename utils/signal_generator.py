"""
Synthetic Biosignal Generator using NeuroKit2 and NumPy/SciPy.
Generates realistic ECG, EMG, EEG, PPG, and EDA signals for TinyML tasks.
"""
import numpy as np

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False


def add_gaussian_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Adds white Gaussian noise to match a desired Signal-to-Noise Ratio (SNR)."""
    p_signal = np.mean(signal ** 2)
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(max(p_noise, 1e-12)), size=signal.shape)
    return signal + noise


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Standardizes signal to zero mean and unit variance or [-1, 1]."""
    std = np.std(signal)
    if std < 1e-8:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


# ====================================================================
# 1. ECG (Electrocardiogram) Arrhythmia Dataset
# Classes: 0: Normal Sinus, 1: Tachycardia (High HR), 2: Bradycardia / Irregular (Low HR / Ectopic)
# ====================================================================
def generate_ecg_dataset(
    num_samples_per_class: int = 300,
    sample_len: int = 187,
    sampling_rate: int = 250,
    random_state: int = 42
):
    """
    Generates synthetic ECG heartbeat windows (sample_len points each).
    Default 187 points at 250 Hz (standard MIT-BIH style windowing).
    """
    np.random.seed(random_state)
    X, y = [], []

    classes = [
        ("Normal", 70, 0.05),       # HR ~ 70 bpm, standard variability
        ("Tachycardia", 125, 0.08),  # HR ~ 125 bpm
        ("Bradycardia", 45, 0.12)    # HR ~ 45 bpm, irregular
    ]

    duration_sec = int(np.ceil((sample_len / sampling_rate) * 2.5)) + 2

    for class_idx, (name, heart_rate, noise_level) in enumerate(classes):
        for _ in range(num_samples_per_class):
            jittered_hr = int(heart_rate + np.random.uniform(-4, 4))
            if NEUROKIT_AVAILABLE:
                ecg = nk.ecg_simulate(
                    duration=duration_sec,
                    sampling_rate=sampling_rate,
                    heart_rate=jittered_hr,
                    random_state=int(np.random.randint(0, 100000))
                )
            else:
                # Fallback synthetic sine-wave cardiac approximation
                t = np.linspace(0, duration_sec, duration_sec * sampling_rate)
                f0 = jittered_hr / 60.0
                ecg = (
                    1.0 * np.sin(2 * np.pi * f0 * t) ** 11
                    + 0.2 * np.sin(2 * np.pi * f0 * t - 0.4)
                    + 0.3 * np.sin(2 * np.pi * f0 * t + 0.5)
                )

            # Add random baseline drift (respiratory wander 0.1 - 0.3 Hz)
            t_axis = np.linspace(0, duration_sec, len(ecg))
            drift = 0.15 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.3) * t_axis)
            ecg = ecg + drift

            # Add high frequency muscle artifact / sensor noise
            ecg = add_gaussian_noise(ecg, snr_db=np.random.uniform(22, 35))

            # Crop random single heartbeat / centered R-peak window of length sample_len
            start_max = len(ecg) - sample_len
            if start_max > 0:
                start = np.random.randint(0, start_max)
                window = ecg[start : start + sample_len]
            else:
                window = np.pad(ecg, (0, max(0, sample_len - len(ecg))))[:sample_len]

            window = normalize_signal(window)
            X.append(window)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


# ====================================================================
# 2. EMG (Electromyogram) Muscle Activity Dataset
# Classes: 0: Rest (Basal), 1: Moderate Contraction, 2: Strong / Burst Contraction
# ====================================================================
def generate_emg_dataset(
    num_samples_per_class: int = 300,
    sample_len: int = 200,
    sampling_rate: int = 1000,
    random_state: int = 42
):
    """
    Generates synthetic surface EMG (sEMG) patterns for gesture/contraction recognition.
    """
    np.random.seed(random_state)
    X, y = [], []

    # Burst frequencies / burst density per class
    profiles = [
        ("Rest", 0.05, 0.02),
        ("Low_Contraction", 0.35, 0.25),
        ("High_Contraction", 0.90, 0.70)
    ]

    duration = 3.0
    for class_idx, (name, burst_amp, burst_freq) in enumerate(profiles):
        for _ in range(num_samples_per_class):
            if NEUROKIT_AVAILABLE:
                # Ensure burst_number * burst_duration < duration
                num_bursts = int(np.random.randint(1, 3)) if class_idx > 0 else 0
                burst_len = min(burst_freq, 0.4)
                emg = nk.emg_simulate(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    burst_number=num_bursts,
                    burst_duration=burst_len,
                    random_state=int(np.random.randint(0, 100000))
                ) * burst_amp
            else:
                t = np.linspace(0, duration, int(duration * sampling_rate))
                carrier = np.random.normal(0, burst_amp, size=len(t))
                envelope = 0.5 * (1 + np.sin(2 * np.pi * np.random.uniform(1.5, 4.0) * t))
                emg = carrier * envelope

            # Add baseline noise
            emg = emg + np.random.normal(0, 0.02, size=len(emg))

            start_max = len(emg) - sample_len
            start = np.random.randint(0, start_max) if start_max > 0 else 0
            window = emg[start : start + sample_len]

            # In EMG, keeping energy level is important, so we scale by fixed factor
            window = np.clip(window, -3.0, 3.0).astype(np.float32)
            X.append(window)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


# ====================================================================
# 3. EEG (Electroencephalogram) Rhythm Dataset
# Classes: 0: Theta band (4-8 Hz, Drowsiness), 1: Alpha band (8-12 Hz, Calm/Relaxed), 2: Beta band (13-30 Hz, Active/Alert)
# ====================================================================
def generate_eeg_dataset(
    num_samples_per_class: int = 300,
    sample_len: int = 256,
    sampling_rate: int = 128,
    random_state: int = 42
):
    """
    Generates synthetic EEG epochs dominated by distinct oscillatory rhythm bands.
    """
    np.random.seed(random_state)
    X, y = [], []

    # Bands definition: (dominant_low, dominant_high, class_id)
    bands = [
        ("Theta", 4.0, 7.5),
        ("Alpha", 8.0, 12.0),
        ("Beta", 14.0, 28.0)
    ]

    duration = sample_len / sampling_rate
    t = np.linspace(0, duration, sample_len, endpoint=False)

    for class_idx, (name, f_low, f_high) in enumerate(bands):
        for _ in range(num_samples_per_class):
            # Background pink-like 1/f noise
            white = np.random.normal(0, 1.0, size=sample_len)
            # Simple cumulative/leaky integrator for 1/f slope
            pink = np.convolve(white, np.exp(-np.linspace(0, 2, 15)), mode='same')

            # Target rhythm frequency
            target_f = np.random.uniform(f_low, f_high)
            phase = np.random.uniform(0, 2 * np.pi)
            oscillation = 2.2 * np.sin(2 * np.pi * target_f * t + phase)

            # Add minor harmonics or non-dominant frequencies
            leak_f = np.random.uniform(1.0, 40.0)
            oscillation += 0.4 * np.sin(2 * np.pi * leak_f * t)

            raw_eeg = oscillation + 0.8 * pink
            window = normalize_signal(raw_eeg)
            X.append(window)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


# ====================================================================
# 4. PPG (Photoplethysmogram) Pulse Contour Dataset
# Classes: 0: Clean/Normal Pulse, 1: Premature/Irregular Beat, 2: High Motion Artifact
# ====================================================================
def generate_ppg_dataset(
    num_samples_per_class: int = 300,
    sample_len: int = 150,
    sampling_rate: int = 100,
    random_state: int = 42
):
    """
    Generates synthetic PPG pulse waves with dicrotic notch, arrhythmia or motion artifact.
    """
    np.random.seed(random_state)
    X, y = [], []

    duration = 5
    for class_idx in range(3):
        for _ in range(num_samples_per_class):
            hr = int(np.random.uniform(60, 85))
            if NEUROKIT_AVAILABLE:
                ppg = nk.ppg_simulate(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    heart_rate=hr,
                    random_state=int(np.random.randint(0, 100000))
                )
            else:
                t = np.linspace(0, duration, duration * sampling_rate)
                f = hr / 60.0
                # Double peak waveform (systolic peak + diastolic notch)
                ppg = 0.8 * np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t - 1.0)

            if class_idx == 1:
                # Ectopic pulse: sudden amplitude and frequency shift
                t_arr = np.linspace(0, 1, len(ppg))
                ppg = ppg * (1.0 + 0.5 * np.sin(2 * np.pi * 3.5 * t_arr))
            elif class_idx == 2:
                # Strong Motion Artifact
                motion = 1.2 * np.sin(2 * np.pi * np.random.uniform(0.5, 2.0) * np.linspace(0, duration, len(ppg)))
                ppg = ppg + motion + np.random.normal(0, 0.2, size=len(ppg))

            start_max = len(ppg) - sample_len
            start = np.random.randint(0, start_max) if start_max > 0 else 0
            window = ppg[start : start + sample_len]
            window = normalize_signal(window)
            X.append(window)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


# ====================================================================
# 5. EDA (Electrodermal Activity / Galvanic Skin Response)
# Classes: 0: Low Arousal (Rest / Baseline), 1: High Arousal (Sympathetic Stress / High SCRs)
# ====================================================================
def generate_eda_dataset(
    num_samples_per_class: int = 300,
    sample_len: int = 200,
    sampling_rate: int = 50,
    random_state: int = 42
):
    """
    Generates synthetic EDA signals with tonic skin conductance level (SCL)
    and phasic skin conductance responses (SCR).
    """
    np.random.seed(random_state)
    X, y = [], []

    duration = 10
    configs = [
        ("Calm_Low_Arousal", 1),   # 1 SCR peak per duration
        ("Stress_High_Arousal", 8)  # 8 SCR peaks per duration
    ]

    for class_idx, (name, num_peaks) in enumerate(configs):
        for _ in range(num_samples_per_class):
            if NEUROKIT_AVAILABLE:
                eda = nk.eda_simulate(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    scr_number=num_peaks,
                    drift=0.01 if class_idx == 0 else 0.05,
                    random_state=int(np.random.randint(0, 100000))
                )
            else:
                t = np.linspace(0, duration, duration * sampling_rate)
                tonic = 2.0 + 0.1 * t
                phasic = np.zeros_like(t)
                for _p in range(num_peaks):
                    p_loc = np.random.uniform(0.5, duration - 1.0)
                    phasic += 0.5 * np.exp(-((t - p_loc) ** 2) / 0.4) * (t >= p_loc)
                eda = tonic + phasic + np.random.normal(0, 0.02, size=len(t))

            start_max = len(eda) - sample_len
            start = np.random.randint(0, start_max) if start_max > 0 else 0
            window = eda[start : start + sample_len]
            window = normalize_signal(window)
            X.append(window)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]
