# Ejemplo 04: Red Neuronal Recurrente (LSTM) en Raspberry Pi Pico 2 (C++ SDK)

Este subproyecto implementa una red neuronal recurrente basada en **LSTM (Long Short-Term Memory)** para el procesamiento y clasificación de series temporales continuas en la **Raspberry Pi Pico 2** (microcontrolador **RP2350** con núcleos **ARM Cortex-M33**).

El objetivo es clasificar secuencias dinámicas de 10 pasos de tiempo en dos clases:
1. `Sinusoidal_Harmonic`: Señal armónica oscilatoria periódica continua.
2. `Transient_Anomaly`: Anomalía transitoria abrupta (pulso de impacto / pico repentino).

---

## 1. Arquitectura de la Red

- **Entrada Secuencial**: 10 pasos temporales continuos $(t_0, \dots, t_9)$ con 1 característica por paso: tensor `[1, 10, 1]`.
- **Capa Recurrente**: `LSTM(16, unroll=True)` optimizada con unrolling estático para eliminar dependencias dinámicas en microcontroladores y maximizar el rendimiento.
- **Capa Densa Oculta**: `Dense(8, activation='relu')`
- **Capa de Salida**: `Dense(2, activation='softmax')` (Probabilidades para `Sinusoidal_Harmonic` y `Transient_Anomaly`).
- **Cuantización**: Full INT8 Post-Training Quantization (PTQ) con cuantización simétrica de pesos y asimétrica de activaciones intermedias.

---

## 2. Archivos del Subproyecto

| Archivo | Descripción |
| :--- | :--- |
| `main.cpp` | Código C++ para Pico SDK que alimenta secuencias temporales en el tensor de entrada, ejecuta el grafo unrolled con TFLM y reporta latencia por inferencia y probabilidades. |
| `CMakeLists.txt` | Configuración CMake para Pico SDK y enlazado con `pico_tflm`. |
| `model_data.h` | Arreglo de 51,944 bytes con el grafo TFLite cuantizado a INT8. |
| `test_samples.h` | 4 secuencias de prueba de validación (2 armónicas, 2 transitorias). |

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
Esto generará el archivo `build/temporal_lstm_recurrent.uf2`.

### Paso 3: Flashear la Raspberry Pi Pico 2
1. Conecta la Pico 2 manteniendo presionado el botón **BOOTSEL**.
2. Arrastra `build/temporal_lstm_recurrent.uf2` a la unidad `RP2350`.
3. La placa arrancará de inmediato.

### Paso 4: Monitoreo Serial USB
Abre una terminal a **115200 baudios** para ver:
- Clasificación de secuencias en tiempo real.
- Probabilidades de pertenencia a cada clase.
- Tiempo de ejecución de la celda recurrente en microsegundos.
