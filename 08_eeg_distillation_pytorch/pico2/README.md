# Ejemplo 08: Clasificador de Ritmos EEG en Raspberry Pi Pico 2 (C++ SDK)

Este subproyecto implementa la inferencia en tiempo real de una red neuronal ligera derivada de **Destilación de Conocimiento (Knowledge Distillation) desde PyTorch** a TensorFlow Lite Micro para clasificar ritmos de electroencefalografía (EEG) en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

---

## 1. Arquitectura de la Red

- **Entrada**: 256 muestras continuas de potencial cerebral microvoltico $(256 \times 1)$ correspondientes a 1 segundo de registro EEG a 256 Hz.
- **Entrenamiento**: Modelo estudiante entrenado en PyTorch para mimetizar la distribución de probabilidad de un ensamble profesor profundo, optimizado luego a TFLite INT8.
- **Salida**: Softmax con 3 clases de estados psicofisiológicos:
  1. `Theta_Drowsy`: Ondas theta (4–8 Hz), asociadas a somnolencia o fatiga cognitiva.
  2. `Alpha_Relaxed`: Ondas alfa (8–13 Hz), asociadas a vigilia relajada con ojos cerrados.
  3. `Beta_Focused`: Ondas beta (13–30 Hz), asociadas a alerta activa y concentración motora/mental.
- **Cuantización**: Full INT8 con aceleración CMSIS-NN SIMD.
- **Tamaño del Modelo en Flash**: ~6,000 bytes (muy compacto).

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código C++ con Pico SDK que inicializa USB stdio, recibe épocas de 256 puntos, ejecuta la inferencia de la red estudiante y activa patrones luminosos en el LED según el estado mental detectado. |
| `CMakeLists.txt` | Configuración CMake para Pico SDK, enlazando `pico_tflm` y generando `eeg_distillation_pytorch.uf2`. |
| `model_data.h` | Arreglo de bytes alineado a 16 bytes con el modelo estudiante INT8. |
| `test_samples.h` | 9 épocas de prueba de ondas cerebrales (3 por cada clase). |

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
Esto generará el archivo `build/eeg_distillation_pytorch.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la placa manteniendo presionado el botón **BOOTSEL**.
2. Copia `build/eeg_distillation_pytorch.uf2` en la unidad de disco `RP2350`.
3. La placa se reiniciará automáticamente.

### Paso 4: Monitoreo Serial USB
Abre un monitor serial a **115200 baudios** para inspeccionar:
- Ritmo cerebral detectado (`Theta_Drowsy`, `Alpha_Relaxed`, `Beta_Focused`).
- Nivel de confianza y tiempo de inferencia en microsegundos ($\mu s$).
