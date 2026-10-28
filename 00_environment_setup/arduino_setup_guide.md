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

### ⚠️ El problema con `Arduino_TensorFlowLite` estándar
Si intentas instalar la biblioteca tradicional `Arduino_TensorFlowLite` desde el gestor de bibliotecas de Arduino, te toparás con errores fatales de compilación en Raspberry Pi Pico y Pico 2:
- `#error "unsupported board"`
- `hal/nrf_i2s.h: No such file or directory`
- `architecture mismatch`

**¿Por qué ocurre esto?**
Esa biblioteca oficial fue empaquetada hace años exclusivamente para la placa **Arduino Nano 33 BLE Sense** (chip Nordic nRF52840, Cortex-M4F). Contiene binarios estáticos precompilados (`.a`) que carecen de código para Cortex-M0+ (RP2040) y Cortex-M33 (RP2350), además de incluir forzosamente periféricos del SDK de Nordic Semiconductor (`peripherals/peripherals.h`).

---

### Solución Probada y Definitiva: `libraries/TFLM_Pico`

Este repositorio incluye una distribución universal y optimizada de **TensorFlow Lite Micro** directamente en la carpeta:
```text
PICO2_TINYML/
└── libraries/
    └── TFLM_Pico/      <- Runtime TFLM completo compilable desde código fuente
```

Esta versión resuelve todas las incompatibilidades:
1. **Compilación 100% desde código fuente**: Compatible con cualquier microcontrolador ARM Cortex-M (Cortex-M0+, Cortex-M33, Cortex-M4F, Cortex-M7).
2. **Compatibilidad total con GCC 14**: Corrige advertencias de FlatBuffers y tipos enteros en `<cstdint>` y `<cstddef>`.
3. **Buffer circular universal**: Implementa buffers ligeros independientes de la API de Arduino AVR/SAMD.

#### Opción 1: Uso directo con Arduino IDE
1. Copia la carpeta `libraries/TFLM_Pico` a tu directorio de bibliotecas de Arduino:
   - **Windows**: `C:\Users\<TuUsuario>\Documents\Arduino\libraries\TFLM_Pico`
   - **Linux**: `~/Arduino/libraries/TFLM_Pico`
   - **macOS**: `~/Documents/Arduino/libraries/TFLM_Pico`
2. Reinicia Arduino IDE.
3. ¡Listo! Cualquier sketch que incluya `#include <tflm_cortexm.h>` o `<tensorflow/lite/micro/...>` compilará sin ningún error.

#### Opción 2: Compilación y Verificación con `arduino-cli`
El proyecto incluye el binario portátil de `arduino-cli` en `bin/arduino-cli.exe` y un script automatizado para verificar todos los ejemplos:

```powershell
# Compilar un sketch individual para Pico 2 (RP2350):
.\bin\arduino-cli.exe compile --fqbn rp2040:rp2040:rpipico2 --libraries libraries 01_ecg_arrhythmia_ptq_tensorflow/arduino/ecg_inference_pico2

# Compilar para Pico 1 (RP2040):
.\bin\arduino-cli.exe compile --fqbn rp2040:rp2040:rpipico --libraries libraries 01_ecg_arrhythmia_ptq_tensorflow/arduino/ecg_inference_pico2

# Ejecutar la suite de verificación completa (todos los ejemplos en ambas tarjetas):
powershell -ExecutionPolicy Bypass -File .\verify_all_arduino.ps1
```

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
