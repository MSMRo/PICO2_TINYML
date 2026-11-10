# Guía Completa de Configuración: Raspberry Pi Pico 2 C++ SDK & TinyML

Esta guía detalla la configuración del entorno oficial del **Raspberry Pi Pico SDK (v2.3+)** para desarrollar, compilar y ejecutar modelos de **TensorFlow Lite Micro (TFLM)** con aceleración vectorial SIMD **CMSIS-NN** en la nueva **Raspberry Pi Pico 2 (microcontrolador RP2350, núcleos duales ARM Cortex-M33 a 150 MHz)**.

---

## 1. Arquitectura de Hardware de la Pico 2 (RP2350)

La Raspberry Pi Pico 2 introduce el procesador **RP2350**, un avance cuantitativo para TinyML:
- **Núcleos**: Dual ARM Cortex-M33 con punto flotante (FPU single-precision) y extensiones de seguridad Arm TrustZone.
- **Instrucciones DSP / SIMD**: Conjunto de instrucciones ARMv8-M Mainline con operaciones SIMD de 8 y 16 bits (`__SMLAD`, `__QADD8`, etc.), aprovechadas al máximo por los kernels de **CMSIS-NN**.
- **Memoria SRAM**: **520 KB** de SRAM dividida en 10 bancos independientes (frente a los 264 KB del RP2040), permitiendo Tensor Arenas amplias para modelos convolucionales y recurrentes (LSTM).
- **Memoria Flash**: 4 MB de memoria Flash QSPI conectada a través del controlador XIP de alto rendimiento.
- **Reloj de Sistema**: 150 MHz por defecto (overclockeable con estabilidad).

---

## 2. Requisitos e Instalación del Toolchain

Para compilar código de C++ puro con el Pico SDK en Windows:

### A. Raspberry Pi Pico SDK Toolchain (Oficial)
Ubicación estándar de instalación de Pico SDK Windows Installer (`C:\Users\<Usuario>\.pico-sdk`):
1. **Pico SDK**: Versión 2.3.1 (`$env:USERPROFILE\.pico-sdk\sdk\2.3.1`)
2. **ARM GCC Toolchain**: ARM GNU Toolchain 15.2.Rel1 (`arm-none-eabi-gcc`)
3. **CMake**: Versión 4.3+ (`cmake.exe`)
4. **Ninja**: Versión 1.13+ (`ninja.exe`)
5. **Picotool**: Versión 2.3.1 (`picotool.exe`)

### B. Configuración de Variables de Entorno (PowerShell)
Para habilitar el compilador en tu sesión actual o permanente:
```powershell
$env:PATH = "$env:USERPROFILE\.pico-sdk\cmake\v4.3.4\bin;$env:USERPROFILE\.pico-sdk\ninja\v1.13.2;$env:USERPROFILE\.pico-sdk\toolchain\15_2_Rel1\bin;" + $env:PATH
```

---

## 3. Integración de TensorFlow Lite Micro (`pico_tflm`)

La librería estática `libraries/pico_tflm` provee un runtime completo de TFLM desacoplado de Arduino:
- **CMSIS-NN Activado**: Banderas `-DARDUINO=1 -DARDUINO_EXCLUDE_CODE=1 -DTF_LITE_STATIC_MEMORY=1 -DCMSIS_NN=1` para compilar kernels optimizados de convolución, profundidad y cuantización vectorial.
- **Cronometraje de Hardware**: Uso directo de `time_us_64()` de `pico/stdlib.h` y `hardware/timer.h` para mediciones de latencia a resolución de 1 microsegundo sin overhead de interrupciones.
- **Mapeo de Tensores Estáticos**: Asignación de memoria estática con alineación `alignas(16)` requerida por los registros vectoriales del Cortex-M33.

---

## 4. Estructura de Proyectos TinyML

Cada subproyecto dentro del repositorio sigue la siguiente estructura estandarizada:
```
0X_nombre_proyecto/
├── python/
│   └── train_*.py         # Entrenamiento, validación y exportación INT8 PTQ/QAT
└── pico2/
    ├── CMakeLists.txt     # Script de construcción para Pico SDK y TFLM
    ├── main.cpp           # Código fuente C++ para inferencia en tiempo real
    ├── model_data.h       # FlatBuffer TFLite con alineación alignas(16)
    ├── test_samples.h     # Muestras de validación extraídas del conjunto de prueba
    └── README.md          # Documentación detallada del subproyecto
```

---

## 5. Compilación y Flasheo de Modelos

### Método 1: Compilación Automática de Toda la Suite
Ejecuta el script proporcionado en la raíz del repositorio:
```powershell
powershell -ExecutionPolicy Bypass -File .\build_all_pico2.ps1
```
Este script compila los 9 subproyectos y genera los 9 archivos `.uf2` dentro de la carpeta `build_all/`.

### Método 2: Compilación de un Subproyecto Individual
Por ejemplo, para compilar el clasificador de arritmias ECG (`05_ecg_arrhythmia_ptq_tensorflow`):
```powershell
# 1. Navegar a la carpeta pico2 del subproyecto
cd 05_ecg_arrhythmia_ptq_tensorflow/pico2

# 2. Configurar el build con CMake y target pico2
cmake -B build -S . -G "Ninja" -DPICO_BOARD=pico2

# 3. Compilar el archivo .uf2
cmake --build build
```

### Método 3: Flashear la Placa Pico 2
1. Conecta la Raspberry Pi Pico 2 a tu computadora manteniendo presionado el pulsador **BOOTSEL**.
2. Windows reconocerá un almacenamiento masivo USB llamado **`RP2350`**.
3. Arrastra y suelta el archivo `.uf2` generado (ej. `ecg_arrhythmia_ptq.uf2`).
4. La placa se reiniciará automáticamente y comenzará a realizar inferencias de inmediato.

---

## 6. Monitoreo Serial USB en Tiempo Real

El código inicializa `stdio_usb` a través del puerto Micro-USB/Type-C de la placa:
1. Abre tu monitor serial preferido (PuTTY, Tera Term, Serial Monitor de VS Code, o Minicom).
2. Selecciona el puerto COM asignado a la Pico 2.
3. Configura la velocidad a **115200 baudios** (8-N-1).
4. Verás los mensajes de inicialización, dimensiones del modelo, tamaño de la arena de tensores y la latencia precisa de cada inferencia en microsegundos ($\mu s$).
