#!/usr/bin/env pwsh
# ToothFairy3 Multi-Instance-Segmentation Complete Pipeline
# Runs algorithm on test data, then evaluates against ground truth

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== ToothFairy3 Multi-Instance-Segmentation Pipeline ===" -ForegroundColor Green
Write-Host ""

# Step 1: Run Algorithm
Write-Host "Step 1: Running Algorithm..." -ForegroundColor Yellow
Set-Location (Join-Path $SCRIPT_DIR "algorithm")
& ".\build.ps1"
& ".\test.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Algorithm step failed. Pipeline aborted."
    exit 1
}

Write-Host ""
Write-Host "Step 2: Running Evaluation..." -ForegroundColor Yellow
Set-Location (Join-Path $SCRIPT_DIR "evaluation")
& ".\build.ps1"
& ".\test.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation step failed."
    exit 1
}

Write-Host ""
Write-Host "=== Pipeline Completed Successfully ===" -ForegroundColor Green
Write-Host "Results available in:" -ForegroundColor Cyan
Write-Host "  - Algorithm Output: C:\Users\Luca\Desktop\Projects\ToothFairy\ToothFairy3\test-results\algorithm-output"
Write-Host "  - Evaluation Output: C:\Users\Luca\Desktop\Projects\ToothFairy\ToothFairy3\test-results\evaluation-output"
