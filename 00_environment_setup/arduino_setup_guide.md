# Guía Completa de Configuración de Arduino IDE para Raspberry Pi Pico 2 / Pico 2 W (RP2350)

Esta guía explica detalladamente cómo configurar el entorno de **Arduino IDE** para compilar y ejecutar modelos **TensorFlow Lite for Microcontrollers (TFLM)** en la **Raspberry Pi Pico 2** y **Raspberry Pi Pico 2 W** (basadas en el microcontrolador RP2350).

---

## 1. Características de Hardware de la Raspberry Pi Pico 2 (RP2350)

A diferencia de la Pico 1 (RP2040, Cortex-M0+ a 133 MHz sin FPU ni DSP), la **Pico 2 (RP2350)** es una de las placas más potentes para TinyML:

| Característica | Raspberry Pi Pico 1 (RP2040) | Raspberry Pi Pico 2 / Pico 2 W (RP2350) | Beneficio para TinyML |
| :--- | :--- | :--- | :--- |
| **Arquitectura de CPU** | Dual ARM Cortex-M0+ | **Dual ARM Cortex-M33** (o Dual Hazard3 RISC-V) | Instrucciones SIMD / DSP dedicadas |
| **Unidad de Punto Flotante (FPU)** | No (Emulada por software) | **FPU por hardware de precisión simple** | Acelera capas float32 x10 - x20 |
| **Instrucciones DSP** | No | **ARMv8-M DSP (SMLAD, etc.)** | Acelera convoluciones y densas INT8 |
| **Frecuencia de Reloj** | 133 MHz | **150 MHz** (estable hasta 250+ MHz overclock) | Mayor tasa de inferencias por segundo |
| **Memoria SRAM (RAM)** | 264 KB | **520 KB en chip** | Permite Tensor Arenas grandes (64 KB+) |
| **Memoria Flash Externa** | 2 MB | **4 MB QSPI Flash** | Almacena decenas de modelos cuantizados |
| **Conectividad (Pico 2 W)** | Solo USB | **Wi-Fi 4 (802.11n) + Bluetooth 5.2 (CYW43439)** | Telemetría inalámbrica en tiempo real |

---

## 2. Instalación del Core RP2350 en Arduino IDE

Utilizamos el core oficial mantenido por la comunidad de Earle F. Philhower, III (`arduino-pico`), el cual cuenta con soporte completo y optimizado para RP2350.

### Paso 2.1: Agregar URL del Gestor de Placas
1. Abre **Arduino IDE** (versión 2.0 o superior recomendada).
2. Ve a **File (Archivo) -> Preferences (Preferencias)**.
3. En el campo **Additional Boards Manager URLs (URLs adicionales del gestor de tarjetas)**, agrega la siguiente línea:
   ```text
   https://github.com/earlephilhower/arduino-pico/releases/download/global/package_rp2040_index.json
   ```
4. Haz clic en **OK**.

### Paso 2.2: Instalar el paquete de tarjetas
1. Ve a **Tools (Herramientas) -> Board (Placa) -> Boards Manager (Gestor de tarjetas)**.
2. Busca `Raspberry Pi Pico/RP2040/RP2350`.
3. Selecciona la versión más reciente del autor **Earle F. Philhower, III** y haz clic en **Install**.

### Paso 2.3: Seleccionar la placa correcta
1. Ve a **Tools -> Board -> Raspberry Pi RP2040/RP2350 Boards**.
2. Selecciona:
   - Para placa estándar: **Raspberry Pi Pico 2**
   - Para placa inalámbrica: **Raspberry Pi Pico 2 W**
3. En **Tools -> CPU Speed**, selecciona **150 MHz** (o 200 MHz para máxima aceleración DSP).
4. En **Tools -> Flash Size**, selecciona **4MB (Sketch: 2MB, FS: 2MB)** o la partición predeterminada.

---

## 3. Instalación de la Biblioteca TensorFlow Lite Micro

Para compilar modelos `.tflite` embebidos en arreglos C (`model_data.h`), necesitamos el motor de inferencia de TensorFlow Lite Micro. Existen dos opciones recomendadas:

### Opción A (Recomendada): Biblioteca `EloquentTinyML`
1. En Arduino IDE, ve a **Tools -> Manage Libraries... (Gestor de Bibliotecas)**.
2. Busca `EloquentTinyML` de Simone Salerno.
3. Haz clic en **Install**.
4. Esta biblioteca encapsula el runtime de TFLM con una interfaz limpia en C++ (`Eloquent::TinyML::TfLite<...>`).

### Opción B: Biblioteca `TensorFlowLite_ESP32` / `Arduino_TensorFlowLite`
1. En el Gestor de Bibliotecas, busca `TensorFlowLite_ESP32` o `Arduino_TensorFlowLite`.
2. Estas bibliotecas incluyen los encabezados estándar de Google:
   - `<tensorflow/lite/micro/all_ops_resolver.h>`
   - `<tensorflow/lite/micro/micro_interpreter.h>`
   - `<tensorflow/lite/schema/schema_generated.h>`
3. Los sketches incluidos en este curso detectan automáticamente si estos encabezados están presentes mediante `#if __has_include(...)`.

---

## 4. Estructura de un Proyecto en Arduino IDE

Cada ejemplo del curso incluye una carpeta en `arduino/` con la siguiente estructura:

```text
arduino/ecg_inference_pico2/
├── ecg_inference_pico2.ino   <- Sketch principal de Arduino
├── model_data.h              <- Modelo cuantizado INT8 exportado como byte array (alignas 16)
└── test_samples.h            <- Señales de prueba de bioseñales con etiquetas verdaderas
```

### ¿Cómo abrir y compilar un ejemplo?
1. En el Arduino IDE, ve a **File -> Open (Abrir)**.
2. Navega a cualquiera de las carpetas de ejemplo, por ejemplo:
   `PICO2_TINYML/01_ecg_arrhythmia_ptq_tensorflow/arduino/ecg_inference_pico2/ecg_inference_pico2.ino`
3. Conecta tu **Raspberry Pi Pico 2** mediante cable USB (mantén pulsado el botón **BOOTSEL** la primera vez si el puerto COM no aparece, o simplemente selecciona el puerto COM detectado).
4. Haz clic en el botón **Upload (Subir)** (flecha hacia la derecha).
5. Abre el **Serial Monitor (Monitor Serie)** a **115200 baudios**.

---

## 5. Salida Típica en el Monitor Serie

Al iniciar, el microcontrolador RP2350 inicializa el intérprete TFLM, mapea los tensores en la memoria `tensor_arena` de 32 KB y procesa las bioseñales de prueba:

```text
========================================================
  TinyML Biosignals - Raspberry Pi Pico 2 (RP2350)
  Example 01: ECG Arrhythmia Classification (INT8 PTQ)
========================================================
CPU Frequency: 150 MHz
Model Size in Flash: 8432 bytes
Allocated Tensor Arena: 32768 bytes

[Model Architecture Information]
Input Shape: [1, 187, 1], Type: 9 (INT8)
Input Quantization: Scale = 0.023412, Zero-Point = -4
Output Shape: [1, 3], Type: 9 (INT8)
Output Quantization: Scale = 0.003906, Zero-Point = -128
Actual Arena Used: 12480 bytes / 32768 bytes (38.1%)
========================================================

--- Evaluating Sample 0 [True Label: Normal (0)] ---
Predicted: Normal (0) with Confidence: 98.42%
Inference Latency: 1840 us (1.84 ms) | Throughput: ~543 inferences/sec
Class Probabilities: [ Normal: 98.4% Tachycardia: 0.8% Bradycardia: 0.8% ]
 Result: SUCCESS (MATCH)

--- Evaluating Sample 1 [True Label: Tachycardia (1)] ---
Predicted: Tachycardia (1) with Confidence: 96.15%
Inference Latency: 1835 us (1.83 ms) | Throughput: ~545 inferences/sec
Class Probabilities: [ Normal: 1.2% Tachycardia: 96.2% Bradycardia: 2.6% ]
 Result: SUCCESS (MATCH)
```

---

## 6. Monitoreo en Tiempo Real y Visualizador Serial

Puedes utilizar el **Serial Plotter (Trazador Serie)** de Arduino IDE (**Tools -> Serial Plotter**) para graficar en tiempo real las probabilidades y las salidas de bioseñales capturadas a través de pines analógicos (ADC0/GPIO26, ADC1/GPIO27, ADC2/GPIO28) conectadas a sensores como AD8232 (ECG), MyoWare (EMG), MAX30102 (PPG) o Grove GSR (EDA).
