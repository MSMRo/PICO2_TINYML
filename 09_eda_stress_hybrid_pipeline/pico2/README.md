# Ejemplo 09: Detector de Estrés por Actividad Electro dérmica (EDA/GSR) en Pico 2 (C++ SDK)

Este subproyecto implementa la inferencia en tiempo real de una red neuronal convolucional 1D cuantizada a INT8 para la detección automática de respuestas de estrés simpático y excitación autónoma mediante señales de **Actividad Electrodérmica (EDA / Galvanic Skin Response - GSR)** en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: Ventana continua de 200 muestras de conductancia dérmica normalizada $(\mu S)$ capturadas a 4 Hz (ventana de 50 segundos).
- **Procesamiento**: Extracción de componentes tónicos (SCL) y fásicos (SCR) mediante convoluciones 1D con aceleración SIMD CMSIS-NN.
- **Salida**: Softmax con 2 estados del sistema nervioso autónomo:
  1. `Calm_Baseline`: Estado basal de calma y relajación fisiológica.
  2. `Stress_High_Arousal`: Respuesta simpática de estrés agudo / sobrecarga cognitiva.
- **Cuantización**: Full INT8 PTQ.
- **Tamaño del Modelo en Flash**: ~7,400 bytes.

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código C++ para Pico SDK que inicializa USB stdio, procesa ventanas de conductancia dérmica, ejecuta la inferencia con CMSIS-NN y genera patrones luminosos de advertencia en caso de detección de estrés agudo. |
| `CMakeLists.txt` | Configuración CMake para Pico SDK, enlazando `pico_tflm` y generando `eda_stress_hybrid_pipeline.uf2`. |
| `model_data.h` | Arreglo de bytes alineado a 16 bytes que contiene el modelo convolucional cuantizado a INT8. |
| `test_samples.h` | 6 ventanas de prueba de EDA representativas (3 de calma, 3 de estrés agudo). |

---

## 3. Instrucciones de Compilación y Flasheo

### Paso 1: Configurar el entorno con CMake
Desde PowerShell:
```powershell
$env:PATH = "C:\Users\User\.pico-sdk\cmake\v4.3.4\bin;C:\Users\User\.pico-sdk\ninja\v1.13.2;C:\Users\User\.pico-sdk\toolchain\15_2_Rel1\bin;" + $env:PATH
cmake -B build -S . -G "Ninja" -DPICO_BOARD=pico2
```

### Paso 2: Compilar el binario UF2
```powershell
cmake --build build
```
Esto generará el archivo `build/eda_stress_hybrid_pipeline.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la placa manteniendo presionado el botón **BOOTSEL**.
2. Copia `build/eda_stress_hybrid_pipeline.uf2` en la unidad de disco `RP2350`.
3. La placa se reiniciará automáticamente.

### Paso 4: Monitoreo Serial USB
Abre un monitor serial a **115200 baudios** para inspeccionar:
- Estado clasificado (`Calm_Baseline` vs. `Stress_High_Arousal`).
- Nivel de confianza diagnóstica (%).
- Latencia de inferencia en microsegundos ($\mu s$).
