# Ejemplo 03: Regresión No Lineal en Raspberry Pi Pico 2 (C++ SDK)

Este subproyecto demuestra la aproximación de funciones continuas y regresión no lineal para calibración de sensores en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

El objetivo es aproximar la función trigonométrica multirrecuente:
$$y = \sin(x) \cdot \cos(2x)$$
a partir de datos ruidosos, ejecutando inferencia en punto entero INT8 en tiempo real.

---

## 1. Arquitectura de la Red

- **Entrada**: 1 valor continuo escalar $x \in [-\pi, \pi]$ en radianes.
- **Capa Oculta 1**: `Dense(32, activation='relu')`
- **Capa Oculta 2**: `Dense(32, activation='relu')`
- **Capa de Salida**: `Dense(1, activation=None)` (Salida lineal continua $y$)
- **Cuantización**: Full INT8 Post-Training Quantization (PTQ) con cuantización simétrica de pesos y asimétrica de activaciones.
- **Dequantización**: Reconstrucción flotante rápida en el firmware mediante $y_{pred} = (q_y - z_{out}) \cdot s_{out}$.

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código C++ con el Pico SDK que mapea el FlatBuffer, cuantiza la entrada escalar, invoca el runtime TFLM con aceleración CMSIS-NN, dequantiza la predicción continua y calcula el Error Absoluto Medio (MAE) y la latencia. |
| `CMakeLists.txt` | Script de compilación CMake enlazando `pico_tflm` y generando `sine_regression.uf2`. |
| `model_data.h` | Arreglo de bytes alineado a 16 bytes con el modelo entrenado y cuantizado. |
| `test_samples.h` | 8 muestras de prueba con coordenadas $x$ y etiquetas continuas de verdad de terreno $y_{true}$. |

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
Esto generará el archivo `build/sine_regression.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la Pico 2 mientras mantienes presionado el botón **BOOTSEL**.
2. Copia `build/sine_regression.uf2` en la unidad de disco `RP2350`.
3. La Pico 2 se reiniciará automáticamente.

### Paso 4: Monitoreo Serial USB
Abre tu terminal serial a **115200 baudios** para inspeccionar:
- Entradas $x$ en radianes.
- Predicciones $y_{pred}$ vs. valor real $y_{true}$.
- Error absoluto instantáneo y promedio (MAE).
- Latencia de inferencia en microsegundos ($\mu s$).
