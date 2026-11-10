# Ejemplo 07: Clasificador de Calidad y Pulso PPG en Raspberry Pi Pico 2 (C++ SDK)

Este subproyecto implementa la inferencia de una red neuronal convolucional 1D entrenada mediante **Quantization-Aware Training (QAT)** para evaluar la morfología y calidad de pulsos de fotopletismografía óptica (PPG) en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: 150 muestras normalizadas de onda de pulso óptico fotopletismográfico $(150 \times 1)$.
- **Entrenamiento**: QAT (Quantization-Aware Training) simulando pérdida de precisión INT8 durante el entrenamiento con gradiente descendente para preservar la fidelidad diagnóstica.
- **Salida**: Softmax con 3 clases de clasificación de pulso:
  1. `Normal_Pulse`: Onda de pulso limpia con incisura dícrota distinguible.
  2. `Ectopic_Arrhythmic`: Latido ectópico prematuro o pulso irregular.
  3. `Motion_Artifact`: Ruido severo por artefacto de movimiento del usuario.
- **Cuantización**: Full INT8 con SIMD CMSIS-NN en Cortex-M33.
- **Tamaño del Modelo en Flash**: ~9,120 bytes.

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código C++ para Pico SDK que inicializa USB stdio, procesa ventanas ópticas de 150 muestras, ejecuta la inferencia acelerada y provee alertas luminosas LED según la calidad del pulso. |
| `CMakeLists.txt` | Configuración CMake para Pico SDK, enlazando `pico_tflm` y generando `ppg_heart_rate_qat.uf2`. |
| `model_data.h` | Arreglo de bytes alineado a 16 bytes que contiene el modelo QAT cuantizado a INT8. |
| `test_samples.h` | 9 ventanas de prueba representativas de pulso óptico (3 normales, 3 ectópicos, 3 con artefactos de movimiento). |

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
Esto generará el archivo `build/ppg_heart_rate_qat.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la placa manteniendo presionado el botón **BOOTSEL**.
2. Copia `build/ppg_heart_rate_qat.uf2` en la unidad de disco `RP2350`.
3. La placa se reiniciará automáticamente.

### Paso 4: Monitoreo Serial USB
Abre una terminal serial a **115200 baudios** para inspeccionar:
- Clasificación de tipo de pulso óptico.
- Confianza del modelo (%) y latencia en microsegundos ($\mu s$).
