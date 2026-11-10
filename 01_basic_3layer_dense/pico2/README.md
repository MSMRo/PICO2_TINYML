# Ejemplo 01: Red Neuronal de 3 Capas (Dense) en Raspberry Pi Pico 2 (C++ SDK)

Este proyecto implementa la inferencia en tiempo real de una red neuronal de 3 capas densas para clasificación no lineal en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: 2 características flotantes normalizadas $(x_1, x_2)$ en el rango $[-1.0, 1.0]$.
- **Capa Oculta 1**: `Dense(16, activation='relu')`
- **Capa Oculta 2**: `Dense(8, activation='relu')`
- **Capa de Salida**: `Dense(2, activation='softmax')` (Probabilidades para `Region_A` y `Region_B`)
- **Cuantización**: INT8 Post-Training Quantization (PTQ) con escalado simétrico.
- **Aceleración**: SIMD nativo Cortex-M33 vía CMSIS-NN.

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código fuente C++ para Pico SDK que inicializa USB stdio, carga el FlatBuffer `.h`, mapea los tensores y evalúa las muestras de prueba midiendo la latencia en microsegundos con el temporizador de hardware (`time_us_64()`). |
| `CMakeLists.txt` | Configuración de compilación para CMake y Pico SDK, enlazando `pico_tflm` y generando el archivo `.uf2`. |
| `model_data.h` | Arreglo C con alineación `alignas(16)` que contiene los pesos cuantizados a INT8 del modelo TFLite. |
| `test_samples.h` | Muestras de prueba de validación con sus etiquetas verdaderas para inferencia continua en el microcontrolador. |

---

## 3. Instrucciones de Compilación y Flasheo

### Paso 1: Configurar el proyecto con CMake
Desde la terminal de PowerShell en el directorio raíz o en esta subcarpeta:
```powershell
cmake -B build -S . -G "Ninja" -DPICO_BOARD=pico2
```

### Paso 2: Compilar el binario UF2
```powershell
cmake --build build
```
Esto generará el archivo `build/basic_3layer_dense.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Mantén presionado el botón **BOOTSEL** de la Raspberry Pi Pico 2 y conéctala a tu computadora mediante un cable USB.
2. La placa aparecerá como una unidad de disco externa llamada `RP2350`.
3. Arrastra y suelta el archivo `build/basic_3layer_dense.uf2` en la unidad.
4. La placa se reiniciará automáticamente y comenzará a ejecutar la red neuronal.

### Paso 4: Monitoreo Serial USB
Abre cualquier monitor serial (PuTTY, minicom, Tera Term o el Monitor Serial de VS Code) a **115200 baudios** en el puerto COM asignado a la Pico 2 para observar la latencia por inferencia (en $\mu s$), las probabilidades de cada clase y el resultado de validación.
