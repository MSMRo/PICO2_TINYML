<#
.SYNOPSIS
    Script automatizado de configuración y compilación para la suite completa de TinyML en Raspberry Pi Pico 2.
    Compila los 9 subproyectos generando los binarios .uf2 para el microcontrolador RP2350 (ARM Cortex-M33).

.DESCRIPTION
    1. Agrega automáticamente al PATH las herramientas de Pico SDK (CMake, Ninja, ARM GCC Toolchain).
    2. Configura el proyecto raíz mediante CMake y Ninja con target PICO_BOARD=pico2.
    3. Compila la librería estática pico_tflm (acelerada con CMSIS-NN) y los 9 ejecutables .uf2:
       - 01_basic_3layer_dense.uf2
       - 02_deep_mlp_classification.uf2
       - 03_sine_regression.uf2
       - 04_temporal_lstm_recurrent.uf2
       - 05_ecg_arrhythmia_ptq.uf2
       - 06_emg_gesture_pruning.uf2
       - 07_ppg_heart_rate_qat.uf2
       - 08_eeg_distillation_pytorch.uf2
       - 09_eda_stress_hybrid_pipeline.uf2
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Raspberry Pi Pico 2 (RP2350 - ARM Cortex-M33) - TinyML Build System" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# 1. Configurar rutas de Pico SDK
$PicoSdkRoot = "$env:USERPROFILE\.pico-sdk"
$CmakeBin    = "$PicoSdkRoot\cmake\v4.3.4\bin"
$NinjaBin    = "$PicoSdkRoot\ninja\v1.13.2"
$Toolchain   = "$PicoSdkRoot\toolchain\15_2_Rel1\bin"

if (Test-Path $CmakeBin) {
    $env:PATH = "$CmakeBin;$NinjaBin;$Toolchain;$env:PATH"
    Write-Host "[INFO] Herramientas de Pico SDK agregadas al PATH." -ForegroundColor Green
} else {
    Write-Host "[WARNING] No se encontró $PicoSdkRoot, utilizando herramientas existentes en PATH." -ForegroundColor Yellow
}

$BuildDir = Join-Path $PSScriptRoot "build_all"

# 2. Configuración con CMake
Write-Host "`n[1/2] Configurando CMake para Raspberry Pi Pico 2 (RP2350)..." -ForegroundColor Yellow
cmake -B $BuildDir -S $PSScriptRoot -G "Ninja" -DPICO_BOARD=pico2

# 3. Compilación de todos los subproyectos
Write-Host "`n[2/2] Compilando suite completa de TinyML..." -ForegroundColor Yellow
cmake --build $BuildDir

Write-Host "`n========================================================================" -ForegroundColor Green
Write-Host "  COMPILACION EXITOSA: Los 9 subproyectos fueron compilados a UF2!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
Write-Host "Binarios generados en ${BuildDir}:" -ForegroundColor Cyan

Get-ChildItem -Path $BuildDir -Recurse -Filter "*.uf2" | ForEach-Object {
    $sizeKb = [math]::Round($_.Length / 1024, 1)
    Write-Host ("  -> {0,-35} ({1} KB)" -f $_.Name, $sizeKb) -ForegroundColor White
}
Write-Host ""
