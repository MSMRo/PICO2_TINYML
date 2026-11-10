# Ejemplo 05: Clasificador de Arritmias ECG en Raspberry Pi Pico 2 (C++ SDK)

Este subproyecto implementa la inferencia en tiempo real de una red neuronal convolucional 1D (1D-CNN) cuantizada a INT8 (PTQ) para la clasificación automática de latidos cardíacos de electrocardiograma (ECG) en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: 187 muestras consecutivas de voltaje ECG normalizado $(187 \times 1)$ correspondientes a una ventana cardíaca centrada en el complejo QRS.
- **Topología**: Convolución 1D con filtros temporales, Batch Normalization, Max Pooling y capas densas con regularización Dropout.
- **Salida**: Softmax de 3 clases diagnósticas:
  1. `Normal`: Ritmo sinusal normal.
  2. `Tachycardia`: Taquicardia (frecuencia cardíaca anormalmente rápida).
  3. `Bradycardia`: Bradicardia (frecuencia cardíaca anormalmente lenta).
- **Cuantización**: Full INT8 Post-Training Quantization (PTQ) con aceleración SIMD CMSIS-NN en el Cortex-M33.
- **Tamaño del Modelo en Flash**: ~10,080 bytes (9.84 KB).

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código fuente C++ para Pico SDK que inicializa USB stdio, mapea el FlatBuffer TFLite en memoria estática, alimenta los latidos de prueba, ejecuta la inferencia con CMSIS-NN y calcula métricas de latencia y exactitud. |
| `CMakeLists.txt` | Configuración de compilación CMake para Pico SDK, enlazando `pico_tflm` y generando `ecg_arrhythmia_ptq.uf2`. |
| `model_data.h` | Arreglo de bytes alineado a 16 bytes que almacena el modelo INT8 cuantizado. |
| `test_samples.h` | 9 latidos de prueba con sus etiquetas verdaderas (Normal, Taquicardia, Bradicardia). |

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
Esto generará el archivo `build/ecg_arrhythmia_ptq.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la placa manteniendo presionado el botón **BOOTSEL**.
2. Copia `build/ecg_arrhythmia_ptq.uf2` en la unidad de disco `RP2350`.
3. La placa se reiniciará automáticamente y comenzará a clasificar latidos cardíacos.

### Paso 4: Monitoreo Serial USB
Abre un monitor serial a **115200 baudios** para ver:
- Diagnóstico clasificado por latido (`Normal`, `Tachycardia`, `Bradycardia`).
- Confianza del modelo (%).
- Latencia de inferencia en microsegundos ($\mu s$).
