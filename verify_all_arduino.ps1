# ==============================================================================
# Automated Verification Script for Arduino TinyML Sketches on Raspberry Pi Pico / Pico 2
# ==============================================================================

$CliPath = "bin\arduino-cli.exe"
$Libraries = "libraries"

$Sketches = @(
    "01_ecg_arrhythmia_ptq_tensorflow\arduino\ecg_inference_pico2",
    "02_emg_gesture_pruning_tensorflow\arduino\emg_inference_pico2",
    "03_eeg_distillation_pytorch\arduino\eeg_inference_pico2",
    "04_ppg_heart_rate_qat_tensorflow\arduino\ppg_inference_pico2",
    "05_eda_stress_hybrid_pipeline\arduino\eda_inference_pico2"
)

$Boards = @(
    @{ Name = "Raspberry Pi Pico 2 (RP2350 - Cortex-M33)"; FQBN = "rp2040:rp2040:rpipico2" },
    @{ Name = "Raspberry Pi Pico (RP2040 - Cortex-M0+)";  FQBN = "rp2040:rp2040:rpipico" }
)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  TINYML BIOSIGNALS - ARDUINO AUTOMATED VERIFICATION SUITE" -ForegroundColor Cyan
Write-Host "  CLI: $CliPath" -ForegroundColor DarkGray
Write-Host "  Libraries: $Libraries" -ForegroundColor DarkGray
Write-Host "================================================================================" -ForegroundColor Cyan

$Results = @()
$AllPassed = $true

foreach ($Board in $Boards) {
    Write-Host "`nTarget Architecture: $($Board.Name) [$($Board.FQBN)]" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray

    foreach ($Sketch in $Sketches) {
        $SketchName = Split-Path (Split-Path (Split-Path $Sketch -Parent) -Parent) -Leaf
        Write-Host "  -> Compiling $SketchName ... " -NoNewline

        $Sw = [System.Diagnostics.Stopwatch]::StartNew()
        $Output = & $CliPath compile -j 0 --warnings none --fqbn $Board.FQBN --libraries $Libraries $Sketch 2>&1
        $Sw.Stop()
        $ExitCode = $LASTEXITCODE

        if ($ExitCode -eq 0) {
            Write-Host "PASS" -ForegroundColor Green -NoNewline
            Write-Host " ($([math]::Round($Sw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor DarkGray
            
            # Extract flash / RAM usage
            $Usage = $Output | Where-Object { $_ -match "Sketch usa|Sketch uses|variables Globales|Global variables" }
            if ($Usage) {
                Write-Host "     $($Usage -join ' | ')" -ForegroundColor DarkGray
            }
            $Results += [PSCustomObject]@{
                Board   = $Board.Name
                Example = $SketchName
                Status  = "PASS"
                Time    = "$([math]::Round($Sw.Elapsed.TotalSeconds, 1))s"
            }
        } else {
            Write-Host "FAILED (Exit Code $ExitCode)" -ForegroundColor Red
            Write-Host "     $($Output | Out-String)" -ForegroundColor DarkRed
            $AllPassed = $false
            $Results += [PSCustomObject]@{
                Board   = $Board.Name
                Example = $SketchName
                Status  = "FAIL"
                Time    = "$([math]::Round($Sw.Elapsed.TotalSeconds, 1))s"
            }
        }
    }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  SUMMARY MATRIX" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
$Results | Format-Table -AutoSize

if ($AllPassed) {
    Write-Host ">>> ALL 10 TARGETS COMPILED AND VALIDATED WITH ZERO ERRORS! <<<`n" -ForegroundColor Green
    exit 0
} else {
    Write-Host ">>> SOME TARGETS FAILED COMPILATION <<<`n" -ForegroundColor Red
    exit 1
}
