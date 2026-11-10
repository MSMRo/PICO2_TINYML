# Ejemplo 06: Clasificador de Gestos y Contracción EMG en Raspberry Pi Pico 2 (C++ SDK)

Este subproyecto implementa la inferencia en tiempo real de una red neuronal convolucional 1D con **Poda Estructurada de Pesos (Weight Pruning)** y cuantización **INT8** para clasificar la actividad electromiográfica (EMG) en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: 200 muestras continuas de potencial muscular superficial normalizado $(200 \times 1)$.
- **Optimización de Modelo**: 50% de dispersión inducida mediante TensorFlow Model Optimization Toolkit (`prune_low_magnitude`), eliminando redundancia sin pérdida de exactitud.
- **Salida**: Softmax con 3 clases de activación muscular:
  1. `Rest`: Músculo en reposo.
  2. `Low_Contraction`: Contracción muscular leve/moderada.
  3. `High_Contraction`: Contracción isométrica máxima.
- **Cuantización**: Full INT8 PTQ ejecutado con CMSIS-NN SIMD.
- **Tamaño del Modelo en Flash**: ~11,472 bytes (11.2 KB).

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código C++ con Pico SDK que carga el modelo podado en Flash, mapea los tensores en la arena estática, procesa ventanas EMG de 200 puntos y genera retroalimentación visual LED según la intensidad de contracción detectada. |
| `CMakeLists.txt` | Configuración CMake para Pico SDK, enlazando `pico_tflm` y generando `emg_gesture_pruning.uf2`. |
| `model_data.h` | Arreglo de bytes alineado a 16 bytes que contiene el modelo convolucional podado a INT8. |
| `test_samples.h` | 9 muestras de validación con sus etiquetas verdaderas (3 por cada clase). |

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
Esto generará el archivo `build/emg_gesture_pruning.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la placa manteniendo presionado el botón **BOOTSEL**.
2. Copia `build/emg_gesture_pruning.uf2` en la unidad de disco `RP2350`.
3. La placa se reiniciará automáticamente.

### Paso 4: Monitoreo Serial USB
Abre una terminal serial a **115200 baudios** para inspeccionar:
- Clasificación de nivel de contracción muscular.
- Porcentaje de confianza de cada clase.
- Latencia de inferencia en microsegundos ($\mu s$).
