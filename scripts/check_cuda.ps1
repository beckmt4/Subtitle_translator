<#
.SYNOPSIS
  Diagnostic script for CUDA / cuDNN readiness and faster-whisper execution path.

.DESCRIPTION
  Performs the following checks:
    1. NVIDIA driver & nvidia-smi availability
    2. Reports GPU(s)
    3. Presence of critical runtime DLLs (cudart, cublas, cublasLt, cudnn_ops)
    4. Python virtual environment detection (optional)
    5. Python import tests (ctranslate2, faster_whisper) and CUDA device count
    6. Summarizes readiness + exit code (0 ready, 1 warnings, 2 errors)

.PARAMETER Python
  Optional explicit Python executable path. If omitted, uses `where python` first result.

.PARAMETER Verbose
  Show detailed output (DLL resolution paths, etc.).

.EXAMPLE
  pwsh -File scripts/check_cuda.ps1

.EXAMPLE
  pwsh -File scripts/check_cuda.ps1 -Python .\.venv\Scripts\python.exe -Verbose

#>
param(
  [string]$Python,
  [switch]$Verbose
)

function Write-Section($title) {
  Write-Host "`n==== $title ==== " -ForegroundColor Cyan
}

$ErrorActionPreference = 'Stop'
$overallStatus = 0  # 0 ok, 1 warn, 2 error

Write-Section "1. Resolving Python"
if (-not $Python) {
  try {
    $Python = (where.exe python 2>$null | Select-Object -First 1)
  } catch { }
}
if (-not $Python) { Write-Host "No python found in PATH" -ForegroundColor Red; exit 2 }
Write-Host "Python: $Python"

Write-Section "2. NVIDIA Driver / nvidia-smi"
$nvidiaSmi = $null
try { $nvidiaSmi = (where.exe nvidia-smi 2>$null | Select-Object -First 1) } catch { }
if ($nvidiaSmi) {
  Write-Host "nvidia-smi found: $nvidiaSmi"
  try {
    $smiOut = & $nvidiaSmi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    $smiOut -split "`n" | ForEach-Object { Write-Host "GPU: $_" }
  } catch {
    Write-Host "Warning: Failed to query nvidia-smi: $_" -ForegroundColor Yellow
    if ($overallStatus -lt 1) { $overallStatus = 1 }
  }
} else {
  Write-Host "nvidia-smi not found. GPU detection may still work via driver APIs." -ForegroundColor Yellow
  if ($overallStatus -lt 1) { $overallStatus = 1 }
}

Write-Section "3. Critical CUDA/cuDNN DLLs"
$dlls = 'cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll','cudnn_ops64_9.dll'
$missing = @()
foreach ($d in $dlls) {
  $found = $false
  try {
    $locs = where.exe $d 2>$null
    if ($locs) { $found = $true; if ($Verbose) { $locs | ForEach-Object { Write-Host "Found $d at $_" -ForegroundColor DarkGray } } }
  } catch { }
  if (-not $found) { $missing += $d }
}
if ($missing.Count -gt 0) {
  Write-Host "Missing runtime DLLs: $($missing -join ', ')" -ForegroundColor Yellow
  Write-Host "Will trigger CPU fallback in whisper_clean.py unless you install CUDA Toolkit + cuDNN." -ForegroundColor Yellow
  if ($overallStatus -lt 1) { $overallStatus = 1 }
} else {
  Write-Host "All critical runtime DLLs present." -ForegroundColor Green
}

Write-Section "4. Python Package / CUDA Device Check"
# Base64-encoded Python snippet (avoids complex quoting in PowerShell)
$pyB64 = 'aW1wb3J0IGpzb24saW1wb3J0bGliLnV0aWwgYXMgaXU7b3V0PXsiY3RyYW5zbGF0ZTIiOk5vbmUsImZhc3Rlcl93aGlzcGVyIjpOb25lLCJjdWRhX2RldmljZXMiOk5vbmUsImVycm9ycyI6W119CmlmIGl1LmZpbmRfc3BlYygiY3RyYW5zbGF0ZTIiKToKICAgIGltcG9ydCBjdHJhbnNsYXRlMiBhcyBjdAogICAgb3V0WyJjdHJhbnNsYXRlMiJdPWdldGF0dHIoY3QsIl9fdmVyc2lvbl9fIixOb25lKQogICAgZGMgPSBodWFzaGF0KHRjdCwiZ2V0X2N1ZGFfZGV2aWNlX2NvdW50IikKICAgIGlmIGRjOgogICAgICAgIHRyeToKICAgICAgICAgICAgb3V0WyJjdWRhX2RldmljZXMiXT1jdC5nZXRfY3VkYV9kZXZpY2VfY291bnQoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgb3V0WyJlcnJvcnMiXS5hcHBlbmQoImN1ZGFfZGV2aWNlX2NvdW50X2ZhaWxlZDogIitzdHIoZSkpCmVsc2U6CiAgICBvdXRbImVycm9ycyJdLmFwcGVuZCgiY3RyYW5zbGF0ZTJfbm90X2ZvdW5kIikKaWYgaXUuZmluZF9zcGVjKCJmYXN0ZXJfd2hpc3BlciIpOgoJIGltcG9ydCBmYXN0ZXJfd2hpc3BlciBhcyBmCglvdXRbImZhc3Rlcl93aGlzcGVyIl09Z2V0YXR0cihmLCJfX3ZlcnNpb25fXyIsTm9uZSkKZWxzZToKCW91dFsiZXJyb3JzIl0uYXBwZW5kKCJmYXN0ZXJfd2hpc3Blcl9ub3RfZm91bmQiKQpwcmludChqc29uLmpzb25kbnVtcHMob3V0KQ=='
$json = & $Python -c "import base64,sys;exec(base64.b64decode('$pyB64'));" 2>$null
$parsed = $null
try { $parsed = $json | ConvertFrom-Json } catch { }
if (-not $parsed) {
  Write-Host "Python diagnostic failed (JSON parse)." -ForegroundColor Red
  $overallStatus = 2
} else {
  Write-Host "ctranslate2: $($parsed.ctranslate2)"; Write-Host "faster-whisper: $($parsed.faster_whisper)"; Write-Host "CUDA devices (reported): $($parsed.cuda_devices)"
  if ($parsed.errors.Count -gt 0) {
    Write-Host "Errors:" -ForegroundColor Yellow
    $parsed.errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    if ($overallStatus -lt 1) { $overallStatus = 1 }
  }
  if ($parsed.cuda_devices -eq 0 -and $parsed.cuda_devices -ne $null) {
    Write-Host "No CUDA devices reported (may still run on CPU)." -ForegroundColor Yellow
    if ($overallStatus -lt 1) { $overallStatus = 1 }
  }
}

Write-Section "5. Summary"
switch ($overallStatus) {
  0 { Write-Host "READY: GPU path appears configured." -ForegroundColor Green }
  1 { Write-Host "WARN: Issues detected (fallback likely)." -ForegroundColor Yellow }
  default { Write-Host "ERROR: Blocking issues detected." -ForegroundColor Red }
}
exit $overallStatus