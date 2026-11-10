# TinyML con Bioseñales y Deep Learning en Raspberry Pi Pico 2 (RP2350)
### Arquitectura Nativa C++ con Raspberry Pi Pico SDK (`pico-sdk`) y TensorFlow Lite Micro (TFLM + CMSIS-NN)

Bienvenido al repositorio profesional **TinyML en Raspberry Pi Pico 2 / Pico 2 W**. Este proyecto implementa un flujo de trabajo integral (*End-to-End*) para diseñar, entrenar, optimizar y desplegar redes neuronales profundas (Deep Learning) aplicadas tanto a modelos canónicos de aprendizaje automático como al procesamiento de bioseñales fisiológicas complejas (ECG, EMG, PPG, EEG, EDA) sobre microcontroladores de ultra-bajo consumo.

A diferencia de implementaciones convencionales basadas en Arduino IDE, **este repositorio utiliza exclusivamente el SDK nativo de C++ de Raspberry Pi (`pico-sdk` v2.3+)**, aprovechando al máximo la microarquitectura **RP2350 (ARM Cortex-M33)**, aceleración SIMD con instrucciones DSP CMSIS-NN, temporizadores de hardware de 64 bits para benchmarks de latencia y gestión estricta de memoria en compilación estática.

---

## Tabla de Contenidos
1. [Hardware: Raspberry Pi Pico 2 y Pico 2 W (RP2350)](#1-hardware-raspberry-pi-pico-2-y-pico-2-w-rp2350)
2. [Hoja de Ruta Progresiva de Proyectos (01 al 09)](#2-hoja-de-ruta-progresiva-de-proyectos-01-al-09)
3. [Técnicas de Optimización TinyML Abordadas](#3-técnicas-de-optimización-tinyml-abordadas)
4. [Estructura del Repositorio](#4-estructura-del-repositorio)
5. [Guía del Entorno de Python con `uv`](#5-guía-del-entorno-de-python-con-uv)
6. [Toolchain de C++ y Raspberry Pi Pico SDK](#6-toolchain-de-c-y-raspberry-pi-pico-sdk)
7. [Instrucciones de Compilación](#7-instrucciones-de-compilación)
8. [Flasheo de Binarios (.uf2) y Monitor Serie](#8-flasheo-de-binarios-uf2-y-monitor-serie)

---

## 1. Hardware: Raspberry Pi Pico 2 y Pico 2 W (RP2350)

La **Raspberry Pi Pico 2** está equipada con el microcontrolador **RP2350**, un avance revolucionario para TinyML respecto al RP2040:

| Parámetro | Pico 1 (RP2040) | Pico 2 / Pico 2 W (RP2350) | Impacto en Inferencia TinyML |
| :--- | :--- | :--- | :--- |
| **Núcleos de CPU** | 2x ARM Cortex-M0+ | **2x ARM Cortex-M33 con TrustZone** | Ejecución de instrucciones ARMv8-M de ciclo único |
| **FPU (Punto Flotante)** | No (emulado por software) | **FPU de precisión simple por hardware** | Acelera cuantización y operaciones float32 x15 - x20 |
| **Instrucciones DSP** | No disponible | **SMLAD, SMUAD, SIMD de 32/16/8 bits** | Kernels CMSIS-NN acelerados para convoluciones y Dense INT8 |
| **Frecuencia de Reloj** | 133 MHz | **150 MHz por defecto** (overclocking a 200–250 MHz) | Cientos de inferencias por segundo con sub-milisegundo de latencia |
| **Memoria SRAM** | 264 KB | **520 KB en chip (SRAM contigua)** | Permite *Tensor Arenas* de 32 KB a 128 KB sin fragmentación |
| **Memoria Flash** | 2 MB | **4 MB QSPI Flash** | Almacena múltiples modelos cuantizados y datos de calibración |
| **Conectividad (Pico 2 W)** | Solo USB | **Wi-Fi 4 (802.11n) + BLE 5.2 (CYW43439)** | Telemetría biomédica y alarmas en tiempo real |

---

## 2. Hoja de Ruta Progresiva de Proyectos (01 al 09)

El repositorio está estructurado pedagógicamente en una secuencia que va desde los fundamentos absolutos de TensorFlow Lite Micro hasta modelos recurrentes temporales y clasificadores biomédicos multicapa:

```
[01 Fundamentos 3-Capas] ──> [02 Deep MLP Clasificación] ──> [03 Regresión Seno] ──> [04 LSTM Temporal]
                                                                                               │
                                                                                               ▼
[09 EDA Estrés 1D-CNN] <── [08 EEG PyTorch KD] <── [07 PPG QAT] <── [06 EMG Pruning] <── [05 ECG PTQ]
```

### Proyectos Fundamentales
1. **[01_basic_3layer_dense](file:///e:/PROJECTS/PICO2_TINYML/01_basic_3layer_dense/pico2)**: Red Neuronal Densa de 3 capas.
   - *Objetivo*: Comprender la anatomía básica de TFLM: inicialización de `tflite::MicroInterpreter`, resolución de operadores con `tflite::MicroMutableOpResolver`, asignación del *Tensor Arena* con alineación de 16 bytes y mapeo de tensores INT8.
2. **[02_deep_mlp_classification](file:///e:/PROJECTS/PICO2_TINYML/02_deep_mlp_classification/pico2)**: Perceptrón Multicapa Profundo (*Deep MLP*).
   - *Objetivo*: Clasificación multiclase no lineal con 4 capas densas intermedias, función de activación Softmax y selección de clase argmax con reporte de probabilidades des-cuantizadas.
3. **[03_sine_regression](file:///e:/PROJECTS/PICO2_TINYML/03_sine_regression/pico2)**: Regresión No Lineal Continua ($\sin(x)$).
   - *Objetivo*: Modelado continuo de funciones analíticas con TFLM INT8. Cuantización de la entrada $x \in [0, 2\pi]$, inferencia del modelo y de-cuantización analítica de la salida flotante con cálculo de error absoluto residual.
4. **[04_temporal_lstm_recurrent](file:///e:/PROJECTS/PICO2_TINYML/04_temporal_lstm_recurrent/pico2)**: Red Neuronal Recurrente con Celdas LSTM Desenrolladas (*Unrolled LSTM*).
   - *Objetivo*: Modelado de series temporales dinámicas y señales periódicas sintéticas. Implementación compatible con microcontroladores mediante desenrollado de grafo (`unroll=True`) sin dependencias de `TensorListReserve`.

### Clasificadores de Bioseñales Reales / Sintéticas
5. **[05_ecg_arrhythmia_ptq_tensorflow](file:///e:/PROJECTS/PICO2_TINYML/05_ecg_arrhythmia_ptq_tensorflow/pico2)**: Clasificador de Arritmias Cardíacas ECG (1D-CNN) con Cuantización Post-Entrenamiento (PTQ).
   - *Bioseñal*: Ventanas de 187 muestras generadas con `neurokit2` (Ritmo Sinusal, Taquicardia, Latido Ectópico).
   - *Optimización*: Calibración INT8 con *Representative Dataset*.
6. **[06_emg_gesture_pruning_tensorflow](file:///e:/PROJECTS/PICO2_TINYML/06_emg_gesture_pruning_tensorflow/pico2)**: Reconocimiento de Gestos Musculares sEMG con Poda de Pesos (*Weight Pruning*).
   - *Bioseñal*: Ráfagas de electromiografía a 1000 Hz (Reposo, Contracción Ligera, Contracción Fuerte).
   - *Optimización*: Poda de magnitud al 65% de dispersión (*sparsity*) con TensorFlow Model Optimization (`tfmot`).
7. **[07_ppg_heart_rate_qat_tensorflow](file:///e:/PROJECTS/PICO2_TINYML/07_ppg_heart_rate_qat_tensorflow/pico2)**: Análisis Morfológico y Calidad de Pulso Óptico PPG con QAT.
   - *Bioseñal*: Pulso fotopletismográfico arterial a 100 Hz (Pulso Limpio, Pulso Ectópico, Artefacto de Movimiento).
   - *Optimización*: *Quantization-Aware Training* (QAT) con estimador directo (*Straight-Through Estimator*).
8. **[08_eeg_distillation_pytorch](file:///e:/PROJECTS/PICO2_TINYML/08_eeg_distillation_pytorch/pico2)**: Clasificador de Ritmos Cerebrales EEG con Destilación de Conocimiento (PyTorch).
   - *Bioseñal*: Ondas cerebrales sintéticas en bandas Theta (4-8 Hz), Alpha (8-12 Hz) y Beta (14-28 Hz).
   - *Optimización*: Transferencia de conocimiento Profesor-Alumno (*Teacher-Student*) mediante divergencia KL con temperatura $T=3.0$, exportado vía ONNX -> TFLite INT8.
9. **[09_eda_stress_hybrid_pipeline](file:///e:/PROJECTS/PICO2_TINYML/09_eda_stress_hybrid_pipeline/pico2)**: Detección de Estrés Autónomo con Respuesta Galvánica de la Piel (EDA / GSR).
   - *Bioseñal*: Conductancia de la piel descomponiendo componentes tónica (SCL) y fásica (SCR).
   - *Arquitectura*: 1D-CNN compacta cuantizada a INT8 para dispositivos portátiles *wearables*.

---

## 3. Técnicas de Optimización TinyML Abordadas

### 3.1 Cuantización Post-Entrenamiento (PTQ INT8)
Convierte tensores float32 a enteros de 8 bits con signo (`int8`, rango $[-128, 127]$):
$$q = \text{clip}\left(\text{round}\left(\frac{x}{S}\right) + Z, -128, 127\right)$$
- **Escala ($S$)**: Factor de resolución flotante.
- **Punto Cero ($Z$)**: Entero que representa el origen flotante 0.0.
- **Representative Dataset**: Conjunto de bioseñales de calibración utilizado por `TFLiteConverter` para fijar los rangos dinámicos de cada capa sin requerir reentrenamiento.

### 3.2 Poda de Pesos (Weight Pruning / Sparsity)
Fuerza a cero las conexiones sinápticas con magnitudes $|w_i| < \epsilon$ mediante decaimiento polinomial progresivo durante el entrenamiento:
- Logra un **65%–75% de dispersión (*sparsity*)**, permitiendo ratios de compresión superiores al 60% en algoritmos como Gzip/Zlib y optimizando las actualizaciones de firmware por aire (OTA).

### 3.3 Destilación de Conocimiento (Knowledge Distillation)
Entrena una red microscópica (*Student*, < 3,000 parámetros) supervisada por una red profunda (*Teacher*, ~300,000 parámetros) minimizando la divergencia de Kullback-Leibler sobre distribuciones de probabilidad suavizadas con un factor de temperatura $T$:
$$\mathcal{L}_{\text{KD}} = \alpha T^2 D_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right) \parallel \sigma\left(\frac{z_t}{T}\right)\right) + (1 - \alpha) \mathcal{L}_{\text{CE}}(z_s, y)$$

### 3.4 Entrenamiento Consciente de Cuantización (QAT)
Inserta operadores de cuantización simulada (*fake quantization*) durante el pase hacia adelante (*forward pass*) para modelar el ruido de truncamiento de 8 bits directamente en la retropropagación con el optimizador Adam. Previene la pérdida de exactitud en contornos biomédicos sutiles.

---

## 4. Estructura del Repositorio

```text
PICO2_TINYML/
├── CMakeLists.txt                         <- CMake maestro para compilar la suite completa
├── pico_sdk_import.cmake                  <- Vinculación y detección automática de Pico SDK
├── build_all_pico2.ps1                    <- Script PowerShell para compilar los 9 binarios .uf2
├── pyproject.toml                         <- Configuración del entorno Python gestionado con uv
├── README.md                              <- Documentación general del repositorio
│
├── libraries/
│   ├── pico_tflm/                         <- Librería estática TFLM nativa para Pico SDK (C++)
│   │   ├── CMakeLists.txt                 <- Compila TFLM con optimizaciones para ARM Cortex-M33
│   │   ├── pico_tflm.h                    <- Encabezado maestro de TensorFlow Lite Micro
│   │   ├── utility_pico.cpp               <- Implementación de debug log para UART/USB stdio
│   │   └── kernel_util_pico.cpp           <- Adaptadores de runtime
│   └── TFLM_Pico/src/                     <- Código fuente C++ de TFLM y CMSIS-NN
│
├── utils/                                 <- Módulos Python reutilizables
│   ├── signal_generator.py                <- Generador de bioseñales sintéticas con NeuroKit2
│   ├── exporter.py                        <- Exportador de modelos TFLite a model_data.h y test_samples.h
│   ├── tflite_eval.py                     <- Evaluador de precisión y métricas FP32 vs INT8
│   └── pytorch_to_tflite.py               <- Puente de conversión PyTorch -> ONNX -> TFLite INT8
│
├── 00_environment_setup/                  <- Guías y verificación de herramientas
│   ├── pico2_sdk_setup_guide.md           <- Guía paso a paso del SDK de C++, CMake y Ninja
│   ├── verify_environment.py              <- Verificador del entorno de Python
│   └── export_tflite_to_header.py         <- Utilidad CLI para generar arreglos C alignas(16)
│
├── 01_basic_3layer_dense/                 <- PROYECTO 01: Fundamentos TFLM 3-Capas
│   ├── python/train_basic_dense.py        <- Entrenamiento en Keras y exportación a INT8
│   └── pico2/                             <- Proyecto C++ Pico SDK (CMakeLists.txt, main.cpp, .h)
│
├── 02_deep_mlp_classification/            <- PROYECTO 02: Deep MLP Clasificación Multiclase
│   ├── python/train_deep_mlp.py           <- Arquitectura profunda con capas intermedias
│   └── pico2/                             <- Inferencia C++ y cálculo argmax de clases
│
├── 03_sine_regression/                    <- PROYECTO 03: Regresión Continua Seno
│   ├── python/train_regression.py         <- Generación de dataset y ajuste de función continua
│   └── pico2/                             <- De-cuantización y benchmark de error absoluto
│
├── 04_temporal_lstm_recurrent/            <- PROYECTO 04: LSTM Temporal para Series de Tiempo
│   ├── python/train_lstm.py               <- LSTM con grafo desenrollado (unroll=True)
│   └── pico2/                             <- Inferencia secuencial en microcontrolador
│
├── 05_ecg_arrhythmia_ptq_tensorflow/      <- PROYECTO 05: ECG + Cuantización PTQ (1D-CNN)
│   ├── python/train_ptq.py                <- Calibración con Representative Dataset
│   └── pico2/                             <- Clasificador de latidos y taquicardia
│
├── 06_emg_gesture_pruning_tensorflow/     <- PROYECTO 06: EMG + Poda de Pesos (65% Sparsity)
│   ├── python/train_pruning.py            <- Poda polinomial con TensorFlow Model Optimization
│   └── pico2/                             <- Detección de contracción muscular en tiempo real
│
├── 07_ppg_heart_rate_qat_tensorflow/      <- PROYECTO 07: PPG + QAT (Quantization-Aware Training)
│   ├── python/train_qat.py                <- Simulación de cuantización en forward pass
│   └── pico2/                             <- Clasificación de calidad de pulso arterial
│
├── 08_eeg_distillation_pytorch/           <- PROYECTO 08: EEG + Destilación de Conocimiento (PyTorch)
│   ├── python/train_distillation.py       <- Entrenamiento Teacher-Student y exportación ONNX
│   └── pico2/                             <- Clasificador de bandas cerebrales (Theta, Alpha, Beta)
│
└── 09_eda_stress_hybrid_pipeline/         <- PROYECTO 09: EDA / GSR + Detección de Estrés
    ├── python/train_eda.py                <- 1D-CNN para conductancia galvánica de la piel
    └── pico2/                             <- Inferencia de estrés psicofisiológico
```

---

## 5. Guía del Entorno de Python con `uv`

Este proyecto gestiona sus dependencias de Python con **`uv`**, el gestor ultra-rápido desarrollado en Rust por Astral.

### 5.1 Instalación de `uv`
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 5.2 Sincronización del Entorno Virtual
Para crear y poblar el entorno virtual `.venv` con TensorFlow, PyTorch, NeuroKit2, ONNX y utilidades:
```powershell
uv sync
```

### 5.3 Entrenamiento y Re-generación de Modelos
Puedes ejecutar los scripts de Python sin activar manualmente el entorno:
```powershell
# Verificar estado del entorno:
uv run 00_environment_setup/verify_environment.py

# Entrenar cualquiera de los 9 proyectos:
uv run 01_basic_3layer_dense/python/train_basic_dense.py
uv run 02_deep_mlp_classification/python/train_deep_mlp.py
uv run 03_sine_regression/python/train_regression.py
uv run 04_temporal_lstm_recurrent/python/train_lstm.py
uv run 05_ecg_arrhythmia_ptq_tensorflow/python/train_ptq.py
uv run 06_emg_gesture_pruning_tensorflow/python/train_pruning.py
uv run 07_ppg_heart_rate_qat_tensorflow/python/train_qat.py
uv run 08_eeg_distillation_pytorch/python/train_distillation.py
uv run 09_eda_stress_hybrid_pipeline/python/train_eda.py
```

---

## 6. Toolchain de C++ y Raspberry Pi Pico SDK

Para compilar el código de C++ nativo se requiere:
1. **Raspberry Pi Pico SDK v2.3+**: Ubicado típicamente en `~/.pico-sdk/sdk/2.3.1` o mediante la variable de entorno `PICO_SDK_PATH`.
2. **Toolchain GNU ARM Embedded**: `arm-none-eabi-gcc`, `arm-none-eabi-g++` (versión 12 o superior con soporte Cortex-M33).
3. **Generador de Construcción**: `CMake` (v3.13+) y `Ninja`.

> Si instalaste el paquete oficial de Raspberry Pi para Windows (*Raspberry Pi Pico SDK Installer*), las herramientas se encuentran instaladas en `C:\Users\<Usuario>\.pico-sdk\`.

---

## 7. Instrucciones de Compilación

### Opción A: Compilación Unificada de los 9 Subproyectos (Recomendado)
El proyecto incluye un script en PowerShell que detecta automáticamente el toolchain y compila todos los proyectos en un solo paso:

```powershell
.\build_all_pico2.ps1
```

O utilizando CMake directamente:
```powershell
# 1. Configurar directorio de construcción global
cmake -B build_all -S . -G "Ninja" -DPICO_BOARD=pico2

# 2. Compilar la suite completa
cmake --build build_all
```

Al terminar, se habrán generado los 9 binarios `.uf2` listos para flashear:
- `build_all/01_basic_dense_build/basic_3layer_dense.uf2`
- `build_all/02_deep_mlp_build/deep_mlp_classification.uf2`
- `build_all/03_regression_build/sine_regression.uf2`
- `build_all/04_lstm_build/temporal_lstm_recurrent.uf2`
- `build_all/05_ecg_build/ecg_arrhythmia_ptq.uf2`
- `build_all/06_emg_build/emg_gesture_pruning.uf2`
- `build_all/07_ppg_build/ppg_heart_rate_qat.uf2`
- `build_all/08_eeg_build/eeg_distillation_pytorch.uf2`
- `build_all/09_eda_build/eda_stress_hybrid_pipeline.uf2`

---

### Opción B: Compilación Individual de un Subproyecto Específico
Cada carpeta `pico2/` es completamente autónoma. Para compilar únicamente un ejemplo:

```powershell
cd 01_basic_3layer_dense/pico2
cmake -B build -S . -G "Ninja" -DPICO_BOARD=pico2
cmake --build build
```
El binario resultante se creará en `01_basic_3layer_dense/pico2/build/basic_3layer_dense.uf2`.

---

## 8. Flasheo de Binarios (.uf2) y Monitor Serie

### 8.1 Carga del Firmware (.uf2) en la Raspberry Pi Pico 2
1. Mantén presionado el botón blanco **BOOTSEL** en la placa Raspberry Pi Pico 2.
2. Conecta el cable Micro-USB / USB-C a tu PC.
3. Suelta el botón **BOOTSEL**. La placa aparecerá como una unidad de disco extraíble llamada `RP2350`.
4. Arrastra y suelta el archivo `.uf2` deseado dentro de la unidad `RP2350`.
5. La Pico 2 se reiniciará automáticamente y comenzará a ejecutar la red neuronal de inmediato.

### 8.2 Lectura del Monitor Serie
Todos los ejemplos tienen habilitada la salida estándar mediante el puerto USB nativo (`stdio_usb_init()`).

Abre cualquier terminal serial (PuTTY, Tera Term, VS Code Serial Monitor o PowerShell) configurado a:
- **Baudrate**: 115200 bps
- **Data bits**: 8, Stop bits: 1, Paridad: Ninguna

#### Ejemplo de Salida en Terminal (Proyecto 05 - ECG Arrhythmia):
```text
==================================================
  Raspberry Pi Pico 2 (RP2350) - ECG Arrhythmia PTQ
  TensorFlow Lite Micro INT8 Inference
==================================================
[INFO] Inicializando TensorFlow Lite Micro...
[INFO] Modelo cargado exitosamente. Version flatbuffer: 3
[INFO] Tensor Arena configurado: 49152 bytes (Alineado a 16 bytes).
[INFO] Input tensor: type=INT8, dims=[1, 187, 1], scale=0.00392157, zero_point=0
[INFO] Output tensor: type=INT8, dims=[1, 3], scale=0.00390625, zero_point=-128

--- Muestra de Test 1/3 (Etiqueta esperada: 0 - Ritmo Normal) ---
[BENCHMARK] Inferencia completada en 814 microsegundos (0.814 ms)
[PREDICCION] Clase: 0 (Ritmo Normal) | Confianza: 98.4%
[ESTADO] Clasificacion CORRECTA!
```

---

## Licencia
Este proyecto y todo su código fuente están licenciados bajo la licencia **MIT**, permitiendo su uso y adaptación académica, personal y comercial libremente.
