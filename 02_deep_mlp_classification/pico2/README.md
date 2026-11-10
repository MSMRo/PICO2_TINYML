# Ejemplo 02: Red Neuronal Profunda Multicapa (Deep MLP) en Raspberry Pi Pico 2 (C++ SDK)

Este proyecto implementa la inferencia en tiempo real de una red neuronal profunda multicapa (Deep MLP de 5 capas) para clasificación multiclase de 4 sensores en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: 4 características continuas normalizadas $(f_0, f_1, f_2, f_3)$ en el rango $[-1.0, 1.0]$.
- **Capa Oculta 1**: `Dense(32, activation='relu')`
- **Capa Oculta 2**: `Dense(24, activation='relu')`
- **Capa Oculta 3**: `Dense(16, activation='relu')`
- **Capa Oculta 4**: `Dense(8, activation='relu')`
- **Capa de Salida**: `Dense(4, activation='softmax')` (Probabilidades para 4 clases: `Class_0`, `Class_1`, `Class_2`, `Class_3`)
- **Cuantización**: INT8 Post-Training Quantization (PTQ) completa para máxima aceleración y mínimo uso de SRAM.
- **Aceleración**: SIMD nativo Cortex-M33 vía CMSIS-NN con aritmética vectorial de 8 bits.

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código fuente C++ para Pico SDK que inicializa USB stdio, carga el FlatBuffer `.h`, mapea los tensores, ejecuta la inferencia para muestras continuas de 4 sensores y mide la latencia en microsegundos con `time_us_64()`. |
| `CMakeLists.txt` | Configuración de compilación CMake para Pico SDK, enlazando `pico_tflm` y generando el ejecutable `deep_mlp_classification.uf2`. |
| `model_data.h` | Arreglo C con alineación `alignas(16)` que contiene los 5,224 bytes de pesos y topología cuantizada del modelo TFLite. |
| `test_samples.h` | Muestras de prueba de 4 dimensiones con sus etiquetas verdaderas para evaluación continua en la placa. |

---

## 3. Instrucciones de Compilación y Flasheo

### Paso 1: Configurar el entorno con CMake
Desde la terminal de PowerShell en esta subcarpeta:
```powershell
$env:PATH = "C:\Users\User\.pico-sdk\cmake\v4.3.4\bin;C:\Users\User\.pico-sdk\ninja\v1.13.2;C:\Users\User\.pico-sdk\toolchain\15_2_Rel1\bin;" + $env:PATH
cmake -B build -S . -G "Ninja" -DPICO_BOARD=pico2
```

### Paso 2: Compilar el binario UF2
```powershell
cmake --build build
```
Esto generará el archivo `build/deep_mlp_classification.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la Pico 2 manteniendo presionado el botón **BOOTSEL**.
2. Aparecerá una unidad de disco llamada `RP2350`.
3. Copia `build/deep_mlp_classification.uf2` en la unidad.
4. La placa se reiniciará automáticamente y comenzará a realizar inferencias.

### Paso 4: Monitoreo Serial USB
Abre una terminal serial a **115200 baudios** para ver los registros en tiempo real:
- Tiempos de inferencia ($\mu s$).
- Probabilidades por clase en porcentaje.
- Detección de la clase con mayor activación.
