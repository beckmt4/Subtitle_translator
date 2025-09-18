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
$foundInToolkit = @()

# Common CUDA Toolkit paths to check
$cudaToolkitPaths = @(
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin',
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin',
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
  'C:\Program Files\Blackmagic Design\DaVinci Resolve',
  'C:\tools\cuda\bin',
  'C:\cuda\bin'
)

# Common cuDNN paths to check (cuDNN is often installed separately)
$cuDNNPaths = @(
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
  'C:\Program Files\NVIDIA\CUDNN\v9.13\bin\12.9',
  'C:\Program Files\NVIDIA\CUDNN\v9.0\bin',
  'C:\Program Files\NVIDIA\CUDNN\v9.1\bin',
  'C:\Program Files\NVIDIA\CUDNN\v9.2\bin',
  'C:\Program Files\Blackmagic Design\DaVinci Resolve',
  'C:\tools\cudnn\bin',
  'C:\cudnn\bin'
)

foreach ($d in $dlls) {
  $found = $false
  $foundInToolkitPath = $null
  
  # First check if DLL is in PATH
  try {
    $locs = where.exe $d 2>$null
    if ($locs) { 
      $found = $true
      if ($Verbose) { $locs | ForEach-Object { Write-Host "Found $d at $_" -ForegroundColor DarkGray } }
    }
  } catch { }
  
  # If not in PATH, check CUDA Toolkit directories
  if (-not $found) {
    $searchPaths = $cudaToolkitPaths
    # For cuDNN DLLs, also check cuDNN-specific paths
    if ($d -like "*cudnn*") {
      $searchPaths = $searchPaths + $cuDNNPaths
    }
    
    foreach ($toolkitPath in $searchPaths) {
      $dllPath = Join-Path $toolkitPath $d
      if (Test-Path $dllPath) {
        $foundInToolkitPath = $toolkitPath
        break
      }
    }
  }
  
  if (-not $found) {
    $missing += $d
    if ($foundInToolkitPath) {
      $foundInToolkit += @{ DLL = $d; Path = $foundInToolkitPath }
    }
  }
}

if ($missing.Count -gt 0) {
  Write-Host "Missing runtime DLLs: $($missing -join ', ')" -ForegroundColor Yellow
  
  if ($foundInToolkit.Count -gt 0) {
    Write-Host "`nDLLs found but not in PATH:" -ForegroundColor Cyan
    $foundInToolkit | ForEach-Object {
      Write-Host "  $($_.DLL) -> $($_.Path)" -ForegroundColor Gray
    }
    
    Write-Host "`nSOLUTION: Add these directories to your PATH:" -ForegroundColor Green
    $uniquePaths = ($foundInToolkit | ForEach-Object { $_.Path } | Sort-Object -Unique)
    
    # Prioritize official NVIDIA paths over third-party (like DaVinci Resolve)
    $officialPaths = $uniquePaths | Where-Object { $_ -like "*NVIDIA*" -and $_ -notlike "*DaVinci*" }
    $thirdPartyPaths = $uniquePaths | Where-Object { $_ -like "*DaVinci*" -or $_ -notlike "*NVIDIA*" }
    
    if ($officialPaths.Count -gt 0) {
      Write-Host "`n  RECOMMENDED (Official NVIDIA):" -ForegroundColor Yellow
      $officialPaths | ForEach-Object {
        Write-Host "    Add to PATH: $_" -ForegroundColor White
      }
    }
    
    if ($thirdPartyPaths.Count -gt 0) {
      Write-Host "`n  ALTERNATIVE (Third-party applications):" -ForegroundColor Yellow
      $thirdPartyPaths | ForEach-Object {
        Write-Host "    Add to PATH: $_" -ForegroundColor Gray
      }
    }
    
    Write-Host "`nTo add to PATH permanently:" -ForegroundColor Yellow
    Write-Host "  1. Open System Properties -> Advanced -> Environment Variables" -ForegroundColor Gray
    Write-Host "  2. Edit the PATH variable and add the directories above" -ForegroundColor Gray
    Write-Host "  3. Restart PowerShell/VS Code for changes to take effect" -ForegroundColor Gray
  } 
  
  # Check for missing cuDNN specifically
  $missingCuDNN = $missing | Where-Object { $_ -like "*cudnn*" }
  $missingCUDA = $missing | Where-Object { $_ -notlike "*cudnn*" }
  
  if ($missingCuDNN.Count -gt 0 -and $foundInToolkit.Count -eq 0) {
    Write-Host "`nMissing cuDNN DLLs: $($missingCuDNN -join ', ')" -ForegroundColor Red
    Write-Host "cuDNN must be downloaded and installed separately from NVIDIA:" -ForegroundColor Yellow
    Write-Host "  1. Download cuDNN from: https://developer.nvidia.com/cudnn" -ForegroundColor Gray
    Write-Host "  2. Extract to CUDA installation directory or add to PATH" -ForegroundColor Gray
  }
  
  if ($missingCUDA.Count -gt 0 -and $foundInToolkit.Count -eq 0) {
    Write-Host "`nCUDA Toolkit installation may be required for: $($missingCUDA -join ', ')" -ForegroundColor Yellow
    Write-Host "Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Gray
  }
  
  Write-Host "`nWithout these DLLs, whisper_clean.py will fall back to CPU mode." -ForegroundColor Yellow
  if ($overallStatus -lt 1) { $overallStatus = 1 }
} else {
  Write-Host "All critical runtime DLLs present." -ForegroundColor Green
}

Write-Section "4. Python Package / CUDA Device Check"
# Base64-encoded Python snippet (avoids complex quoting in PowerShell)
$pyB64 = 'aW1wb3J0IGpzb24saW1wb3J0bGliLnV0aWwgYXMgaXUKb3V0PXsiY3RyYW5zbGF0ZTIiOk5vbmUsImZhc3Rlcl93aGlzcGVyIjpOb25lLCJjdWRhX2RldmljZXMiOk5vbmUsImVycm9ycyI6W119CmlmIGl1LmZpbmRfc3BlYygiY3RyYW5zbGF0ZTIiKToKICAgIGltcG9ydCBjdHJhbnNsYXRlMiBhcyBjdAogICAgb3V0WyJjdHJhbnNsYXRlMiJdPWdldGF0dHIoY3QsIl9fdmVyc2lvbl9fIixOb25lKQogICAgaWYgaGFzYXR0cihjdCwiZ2V0X2N1ZGFfZGV2aWNlX2NvdW50Iik6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBvdXRbImN1ZGFfZGV2aWNlcyJdPWN0LmdldF9jdWRhX2RldmljZV9jb3VudCgpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBvdXRbImVycm9ycyJdLmFwcGVuZCgiY3VkYV9kZXZpY2VfY291bnRfZmFpbGVkOiAiK3N0cihlKSkKZWxzZToKICAgIG91dFsiZXJyb3JzIl0uYXBwZW5kKCJjdHJhbnNsYXRlMl9ub3RfZm91bmQiKQppZiBpdS5maW5kX3NwZWMoImZhc3Rlcl93aGlzcGVyIik6CiAgICBpbXBvcnQgZmFzdGVyX3doaXNwZXIgYXMgZgogICAgb3V0WyJmYXN0ZXJfd2hpc3BlciJdPWdldGF0dHIoZiwiX192ZXJzaW9uX18iLE5vbmUpCmVsc2U6CiAgICBvdXRbImVycm9ycyJdLmFwcGVuZCgiZmFzdGVyX3doaXNwZXJfbm90X2ZvdW5kIikKcHJpbnQoanNvbi5kdW1wcyhvdXQpKQ=='
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