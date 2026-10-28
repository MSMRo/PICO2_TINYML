# TinyML con Bioseñales en Raspberry Pi Pico 2 / Pico 2 W (RP2350)
### Curso Práctico Completo: Deep Learning, Optimización en el Borde y Despliegue en Arduino IDE

Bienvenido al repositorio del curso **TinyML con Bioseñales en Raspberry Pi Pico 2 / Pico 2 W**. Este proyecto está diseñado para enseñarte el flujo de trabajo profesional de extremo a extremo (*End-to-End*) para diseñar, entrenar, optimizar y desplegar redes neuronales profundas (Deep Learning) aplicadas a señales biomédicas en sistemas embebidos de ultra-bajo consumo.

---

## Tabla de Contenidos
1. [Visión General del Curso](#1-visión-general-del-curso)
2. [Hardware: Raspberry Pi Pico 2 y Pico 2 W (RP2350)](#2-hardware-raspberry-pi-pico-2-y-pico-2-w-rp2350)
3. [Técnicas de Optimización TinyML Abordadas](#3-técnicas-de-optimización-tinyml-abordadas)
4. [Estructura del Proyecto y Contenido de las Carpetas](#4-estructura-del-proyecto-y-contenido-de-las-carpetas)
5. [Guía Completa de Uso de `uv`](#5-guía-completa-de-uso-de-uv)
6. [Configuración de Arduino IDE para Raspberry Pi Pico 2](#6-configuración-de-arduino-ide-para-raspberry-pi-pico-2)
7. [Guía Rápida de Ejecución de los Ejemplos](#7-guía-rápida-de-ejecución-de-los-ejemplos)

---

## 1. Visión General del Curso

El procesamiento de bioseñales (ECG, EMG, EEG, PPG, EDA) tradicionalmente requería equipos de escritorio o servidores en la nube debido a la complejidad de las operaciones matemáticas de filtrado y redes neuronales. Con **TinyML** y microcontroladores modernos como el **RP2350** de la Raspberry Pi Pico 2, ahora es posible ejecutar modelos de Deep Learning directamente en el cuerpo del paciente (dispositivos *Wearables* e implantes) con:
- **Latencia determinista**: Decisiones críticas en menos de 3 milisegundos sin depender de conexión a internet.
- **Privacidad total**: Los datos biomédicos sensibles nunca salen del microcontrolador.
- **Consumo ultra-bajo**: Baterías que duran semanas o meses.
- **Cero costo de infraestructura en la nube**: Inferencia local autónoma.

Para garantizar datos fisiológicos realistas sin requerir sensores físicos desde el primer día, el curso utiliza la biblioteca científica **`neurokit2`** para generar bioseñales sintéticas basadas en modelos dinámicos cardiovasculares y neurofisiológicos.

---

## 2. Hardware: Raspberry Pi Pico 2 y Pico 2 W (RP2350)

La **Raspberry Pi Pico 2** está equipada con el procesador **RP2350**, un avance monumental frente a los microcontroladores de generaciones anteriores:

| Parámetro | Pico 1 (RP2040) | Pico 2 / Pico 2 W (RP2350) | Impacto en TinyML |
| :--- | :--- | :--- | :--- |
| **Núcleos de CPU** | 2x ARM Cortex-M0+ | **2x ARM Cortex-M33** (o 2x Hazard3 RISC-V) | Conjunto de instrucciones ARMv8-M moderno |
| **FPU (Punto Flotante)** | No (emulado por software) | **FPU de precisión simple por hardware** | Acelera capas float32 x15 - x20 |
| **Instrucciones DSP** | No disponible | **SMLAD, SMUAD (SIMD de 32/16/8 bits)** | Convoluciones y capas densas INT8 a máxima velocidad |
| **Frecuencia de Reloj** | 133 MHz | **150 MHz** (overclock estable a 200-250 MHz) | Más de 500 inferencias por segundo |
| **Memoria SRAM (RAM)** | 264 KB | **520 KB en chip** | Gran capacidad para *Tensor Arenas* de 32 a 64 KB |
| **Memoria Flash** | 2 MB | **4 MB QSPI Flash** | Almacena decenas de modelos `.h` simultáneos |
| **Inalámbrico (Pico 2 W)** | Solo USB | **Wi-Fi 4 (802.11n) + Bluetooth 5.2 (CYW43439)** | Transmisión BLE/Wi-Fi de resultados o alarmas |

---

## 3. Técnicas de Optimización TinyML Abordadas

### 3.1 Cuantización Post-Entrenamiento (PTQ - Post-Training Quantization)
Convierte pesos y mapas de activación de precisión flotante IEEE 754 de 32 bits (`float32`) a enteros de 8 bits con signo (`int8`, rango $[-128, 127]$):
$$q = \text{clip}\left(\text{round}\left(\frac{x}{S}\right) + Z, -128, 127\right)$$
- **Escala ($S$)**: Factor de escala flotante que define la resolución del entero.
- **Zero-Point ($Z$)**: Entero que representa el cero real del valor flotante.
- **Representative Dataset**: Conjunto de bioseñales de calibración utilizado para registrar el rango dinámico de las activaciones internas sin reentrenar la red.

### 3.2 Poda de Pesos (Weight Pruning / Sparsity)
Elimina conexiones neuronales prescindibles forzando a cero los pesos cuya magnitud $|w_i| < \epsilon$.
- Mediante un programa polinomial de decaimiento (*Polynomial Decay*), la red se poda progresivamente durante el entrenamiento hasta alcanzar entre 50% y 75% de dispersión (*sparsity*).
- Aunque el tamaño del archivo `.tflite` plano no cambia inicialmente, las matrices dispersas se comprimen hasta un **70% más con compresión Gzip/Zlib**, ideal para actualizaciones por aire (*OTA Firmware Updates*) y almacenamiento flash.

### 3.3 Destilación de Conocimiento (Knowledge Distillation)
Transfiere la capacidad predictiva de una red neuronal grande y profunda (**Teacher / Profesor**, p. ej. 300,000 parámetros) a una red ultra-compacta (**Student / Alumno**, < 3,000 parámetros).
- Utiliza la divergencia de Kullback-Leibler sobre distribuciones de probabilidad suavizadas con un hiperparámetro de **Temperatura ($T$)**:
  $$\mathcal{L}_{\text{KD}} = \alpha T^2 D_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right) \parallel \sigma\left(\frac{z_t}{T}\right)\right) + (1 - \alpha) \mathcal{L}_{\text{CE}}(z_s, y)$$
- El Alumno aprende no solo la etiqueta correcta, sino las similitudes ocultas entre clases (*Dark Knowledge*).

### 3.4 Entrenamiento Consciente de Cuantización (QAT - Quantization-Aware Training)
Inserta operadores de cuantización simulada (*fake quantization*) durante el pase hacia adelante (*forward pass*) y utiliza el estimador directo (*Straight-Through Estimator, STE*) en la retropropagación (*backpropagation*).
- Modela el error de redondeo de 8 bits en cada iteración del optimizador Adam.
- Elimina casi por completo la pérdida de exactitud observada en señales complejas como pulsos PPG.

---

## 4. Estructura del Proyecto y Contenido de las Carpetas

Cada ejemplo está estrictamente organizado con su propia carpeta, conteniendo un subdirectorio `python/` (con scripts y notebooks de Jupyter) y un subdirectorio `arduino/` (con el código `.ino` y los encabezados `.h` listos para compilar en la Pico 2):

```text
PICO2_TINYML/
├── pyproject.toml                     <- Configuración del entorno Python gestionado con uv
├── README.md                          <- Manual del curso, tutorial de uv y guía de Arduino
├── utils/                             <- Herramientas compartidas
│   ├── signal_generator.py            <- Generadores de bioseñales con NeuroKit2 (ECG, EMG, EEG, PPG, EDA)
│   ├── exporter.py                    <- Exportador de modelos TFLite a arreglos C (model_data.h) y test_samples.h
│   ├── tflite_eval.py                 <- Evaluador de precisión, latencia y tamaño FP32 vs INT8
│   └── pytorch_to_tflite.py           <- Puente de conversión PyTorch -> ONNX -> TFLite
│
├── 00_environment_setup/              <- Verificación de entorno y guías
│   ├── verify_environment.py          <- Comprueba paquetes Python y especificaciones RP2350
│   ├── export_tflite_to_header.py     <- Utilidad de línea de comandos para exportar .tflite a .h
│   └── arduino_setup_guide.md         <- Guía detallada de instalación de placas y librerías en Arduino IDE
│
├── 01_ecg_arrhythmia_ptq_tensorflow/  <- Ejemplo 01: ECG + PTQ (Post-Training Quantization)
│   ├── python/
│   │   ├── 01_ecg_arrhythmia_ptq.ipynb <- Notebook interactivo paso a paso con gráficas
│   │   └── train_ptq.py               <- Script de entrenamiento, cuantización y exportación automática
│   └── arduino/
│       └── ecg_inference_pico2/
│           ├── ecg_inference_pico2.ino <- Sketch de Arduino para Raspberry Pi Pico 2
│           ├── model_data.h           <- Arreglo C del modelo INT8 (alignas 16 para Cortex-M33)
│           └── test_samples.h         <- Arreglos de latidos de prueba con etiquetas verdaderas
│
├── 02_emg_gesture_pruning_tensorflow/ <- Ejemplo 02: sEMG + Poda de Pesos (Weight Pruning / TFMOT)
│   ├── python/
│   │   ├── 02_emg_pruning.ipynb       <- Notebook con histogramas de dispersión de pesos y compresión
│   │   └── train_pruning.py           <- Script de poda polinomial a 65% de dispersión + INT8
│   └── arduino/
│       └── emg_inference_pico2/
│           ├── emg_inference_pico2.ino <- Sketch de Arduino con control de LED según contracción
│           ├── model_data.h
│           └── test_samples.h
│
├── 03_eeg_distillation_pytorch/       <- Ejemplo 03: EEG + Destilación de Conocimiento (PyTorch)
│   ├── python/
│   │   ├── 03_eeg_distillation_pytorch.ipynb <- Notebook con arquitectura Teacher-Student
│   │   └── train_distillation.py      <- Script de entrenamiento con pérdida de destilación y exportación
│   └── arduino/
│       └── eeg_inference_pico2/
│           ├── eeg_inference_pico2.ino <- Sketch de Arduino para inferencia de estados cerebrales
│           ├── model_data.h
│           └── test_samples.h
│
├── 04_ppg_heart_rate_qat_tensorflow/  <- Ejemplo 04: PPG + QAT (Quantization-Aware Training)
│   ├── python/
│   │   ├── 04_ppg_qat.ipynb           <- Notebook de simulación de cuantización y cálculo de STE
│   │   └── train_qat.py               <- Script de entrenamiento QAT con TFMOT
│   └── arduino/
│       └── ppg_inference_pico2/
│           ├── ppg_inference_pico2.ino <- Sketch de Arduino para análisis de ondas de pulso
│           ├── model_data.h
│           └── test_samples.h
│
└── 05_eda_stress_hybrid_pipeline/     <- Ejemplo 05: EDA / GSR + Detección de Estrés Simpático
    ├── python/
    │   ├── 05_eda_stress.ipynb        <- Notebook de análisis tónico/fásico y clasificación
    │   └── train_eda.py               <- Script de entrenamiento de 1D-CNN para conductancia de la piel
    └── arduino/
        └── eda_inference_pico2/
            ├── eda_inference_pico2.ino <- Sketch de Arduino con alarma de estrés psicofisiológico
            ├── model_data.h
            └── test_samples.h
```

---

## 5. Guía Completa de Uso de `uv`

Este proyecto utiliza **`uv`**, el gestor de paquetes y entornos de Python más rápido del mundo (escrito en Rust por Astral, creadores de Ruff). `uv` reemplaza a `pip`, `pip-tools`, `virtualenv`, `poetry` y `conda`, ofreciendo instalaciones entre 10 y 100 veces más rápidas con resolución determinista y gestión automática de versiones de Python.

### 5.1 ¿Cómo instalar `uv`?

- **En Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **En Linux o macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

Para verificar que esté instalado correctamente:
```bash
uv --version
```

### 5.2 Creación del Entorno Virtual y Sincronización Automática
Para instalar automáticamente todas las dependencias del proyecto (TensorFlow, PyTorch, NeuroKit2, JupyterLab, ONNX, etc.) en un entorno virtual aislado `.venv`:
```bash
uv sync
```
`uv` descargará e instalará todas las ruedas (*wheels*) en pocos segundos.

### 5.3 ¿Cómo ejecutar scripts de Python con `uv`?
**No necesitas activar manualmente el entorno virtual**. `uv run` detecta automáticamente el `.venv` y lo ejecuta en el contexto correcto:

```bash
# 1. Verificar el entorno y dependencias:
uv run 00_environment_setup/verify_environment.py

# 2. Entrenar y exportar el Ejemplo 01 (ECG):
uv run 01_ecg_arrhythmia_ptq_tensorflow/python/train_ptq.py

# 3. Entrenar y exportar el Ejemplo 02 (EMG Pruning):
uv run 02_emg_gesture_pruning_tensorflow/python/train_pruning.py

# 4. Entrenar y exportar el Ejemplo 03 (EEG Distillation con PyTorch):
uv run 03_eeg_distillation_pytorch/python/train_distillation.py

# 5. Entrenar y exportar el Ejemplo 04 (PPG QAT):
uv run 04_ppg_heart_rate_qat_tensorflow/python/train_qat.py

# 6. Entrenar y exportar el Ejemplo 05 (EDA Stress):
uv run 05_eda_stress_hybrid_pipeline/python/train_eda.py
```

### 5.4 ¿Cómo abrir los Notebooks de Jupyter con `uv`?
Para lanzar **JupyterLab** o el servidor de **Jupyter Notebook**:
```bash
uv run jupyter lab
```
o bien:
```bash
uv run jupyter notebook
```
Esto abrirá tu navegador web predeterminado. Podrás explorar y ejecutar celda por celda cada uno de los notebooks interactivos `.ipynb` de cada carpeta.

### 5.5 ¿Cómo añadir nuevos paquetes al proyecto?
Si deseas instalar una librería adicional (por ejemplo `seaborn`):
```bash
uv add seaborn
```
`uv` actualizará automáticamente el archivo `pyproject.toml` y bloqueará las versiones exactas.

---

## 6. Configuración de Arduino IDE para Raspberry Pi Pico 2

Para compilar los sketches `.ino` en la Raspberry Pi Pico 2 / Pico 2 W:

1. **Instalar el Core RP2350 de Earle Philhower**:
   - En Arduino IDE: **File -> Preferences**.
   - En *Additional Boards Manager URLs*, agrega:
     ```text
     https://github.com/earlephilhower/arduino-pico/releases/download/global/package_rp2040_index.json
     ```
   - Abre **Tools -> Board -> Boards Manager**, busca `Raspberry Pi Pico/RP2040/RP2350` e instálalo.
2. **Seleccionar la Placa**:
   - **Tools -> Board -> Raspberry Pi RP2040/RP2350 Boards -> Raspberry Pi Pico 2** (o **Raspberry Pi Pico 2 W**).
   - **Tools -> CPU Speed -> 150 MHz** (o 200 MHz para máximo desempeño).
3. **Instalar la Biblioteca TensorFlow Lite Micro**:
   - En **Tools -> Manage Libraries...**, busca e instala `EloquentTinyML` o `TensorFlowLite_ESP32`.
4. **Cargar el Código en la Placa**:
   - Conecta la Raspberry Pi Pico 2 mediante USB.
   - Abre cualquiera de los archivos `.ino` (por ejemplo `01_ecg_arrhythmia_ptq_tensorflow/arduino/ecg_inference_pico2/ecg_inference_pico2.ino`).
   - Haz clic en **Upload (Subir)**.
   - Abre el **Serial Monitor** configurado a **115200 baudios**.

---

## 7. Guía Rápida de Ejecución de los Ejemplos

### Ejemplo 01: Detección de Arritmias ECG con Cuantización INT8 (PTQ)
- **Bioseñal**: Ventanas de 187 muestras de electrocardiograma a 250 Hz generadas con `neurokit2.ecg_simulate`.
- **Clases**: Ritmo Sinusal Normal (0), Taquicardia (1), Bradicardia / Latido Ectópico (2).
- **Modelo**: 1D-CNN con convoluciones separables en profundidad.
- **Optimización**: Cuantización post-entrenamiento completa a INT8 con calibración mediante *Representative Dataset*.
- **Despliegue**: El sketch en Arduino lee las señales empaquetadas en `test_samples.h`, aplica la cuantización de entrada $q = \text{round}(x / S) + Z$, ejecuta la inferencia con TFLM y calcula la latencia en microsegundos.

### Ejemplo 02: Reconocimiento de Gestos/Contracción sEMG con Poda de Pesos (Pruning)
- **Bioseñal**: Electromiografía superficial a 1000 Hz simulando ráfagas de activación muscular.
- **Clases**: Reposo (0), Contracción Ligera (1), Contracción Fuerte (2).
- **Optimización**: Poda basada en magnitud con `tfmot` alcanzando 65% de dispersión (*weights = 0.0*), seguida de conversión a INT8.
- **Despliegue**: La Pico 2 detecta la fuerza muscular y enciende el LED integrado en tiempo real según el nivel de contracción.

### Ejemplo 03: Clasificación de Ritmos Cerebrales EEG con Destilación de Conocimiento (PyTorch)
- **Bioseñal**: Ritmos electroencefalográficos (Theta: 4-8 Hz, Alpha: 8-12 Hz, Beta: 14-28 Hz).
- **Framework**: PyTorch.
- **Optimización**: Se entrena un modelo *Teacher* profundo (~300,000 parámetros) y se transfiere su conocimiento a un modelo *Student* microscópico (< 3,000 parámetros) mediante pérdida KL-Divergence con Temperatura $T=3.0$. Se exporta vía ONNX -> TFLite INT8 -> `model_data.h`.

### Ejemplo 04: Calidad y Detección de Ondas de Pulso PPG con QAT
- **Bioseñal**: Señales de fotopletismografía óptica a 100 Hz (Pulso Limpio, Pulso Ectópico, Artefacto de Movimiento Severo).
- **Optimización**: *Quantization-Aware Training* (QAT) con `tfmot`, emulando el redondeo de 8 bits durante el entrenamiento para garantizar cero degradación de exactitud en los picos sistólicos y diastólicos.

### Ejemplo 05: Detección de Estrés Autónomo con Actividad Electrodérmica (EDA/GSR)
- **Bioseñal**: Respuesta Galvánica de la Piel a 50 Hz simulando respuestas de conductancia fásica (SCR) y nivel tónico (SCL).
- **Clases**: Calma / Reposo (0) vs Alta Activación / Estrés Simpático (1).
- **Despliegue**: Inferencia en menos de 1.5 milisegundos en la RP2350, ideal para dispositivos portátiles de monitoreo de estrés continuo.

---

## Licencia
Este curso y su código están bajo la licencia **MIT**, permitiendo su uso académico, personal e industrial.
