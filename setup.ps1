<#
.SYNOPSIS
    Automated setup script for Whisper MVP subtitle generator (Windows + PowerShell).

.DESCRIPTION
    Creates/refreshes a Python virtual environment, installs known-good dependency versions, 
    validates FFmpeg availability, optionally downloads a Whisper model (warmup), and runs validation tests.

.PARAMETER Force
    Recreates the virtual environment if it already exists.

.PARAMETER Python
    Path to a python executable to use (defaults to one resolved on PATH).

.PARAMETER Model
    Whisper model to warm up (tiny, base, small, medium, large-v3). If omitted, no warmup.

.PARAMETER NoValidate
    Skip running test_clean.py at the end.

.PARAMETER NoFFmpegCheck
    Skip FFmpeg presence check.

.PARAMETER Quiet
    Suppress non-error output.

.EXAMPLE
    ./setup.ps1

.EXAMPLE
    ./setup.ps1 -Force -Model small

.EXAMPLE
    ./setup.ps1 -Python "C:\Python311\python.exe" -Model medium

.NOTES
    Ensures compatible versions:
        faster-whisper 1.2.0
        ctranslate2 4.5.0
        av 11.0.0
        numpy (latest compatible for py311)
        tokenizers, transformers, rich, onnxruntime

#>
[CmdletBinding()]
param(
    [switch]$Force,
    [string]$Python,
    [ValidateSet('tiny','base','small','medium','large-v3','')]
    [string]$Model = '',
    [switch]$NoValidate,
    [switch]$NoFFmpegCheck,
    [switch]$Quiet,
    [switch]$LaunchShell
)

$ErrorActionPreference = 'Stop'
function Write-Info { param($Msg) if(-not $Quiet){ Write-Host "[INFO] $Msg" -ForegroundColor Cyan } }
function Write-Ok   { param($Msg) if(-not $Quiet){ Write-Host "[ OK ] $Msg" -ForegroundColor Green } }
function Write-Warn { param($Msg) if(-not $Quiet){ Write-Host "[WARN] $Msg" -ForegroundColor Yellow } }
function Write-Err  { param($Msg) Write-Host "[ERR ] $Msg" -ForegroundColor Red }

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
Write-Info "Project root: $projectRoot"

# Resolve Python
if(-not $Python){ $Python = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Path }
if(-not $Python){ Write-Err "Python not found on PATH. Install Python 3.11+ and retry."; exit 1 }
Write-Info "Using Python: $Python"

# FFmpeg check
if(-not $NoFFmpegCheck){
    if(Get-Command ffmpeg -ErrorAction SilentlyContinue){ Write-Ok "FFmpeg detected: $(ffmpeg -version | Select-String -Pattern '^ffmpeg' -CaseSensitive:$false | Select-Object -First 1)" }
    else { Write-Warn "FFmpeg not found. Install before using the tool (see README)." }
}

$venvPath = Join-Path $projectRoot '.venv'
if( (Test-Path $venvPath) -and $Force ){
    Write-Warn "Removing existing virtual environment (Force specified)"
    Remove-Item -Recurse -Force $venvPath
}

if(-not (Test-Path $venvPath)){
    Write-Info "Creating virtual environment"
    & $Python -m venv .venv
    if($LASTEXITCODE -ne 0){ Write-Err "Failed to create virtual environment"; exit 1 }
    Write-Ok "Virtual environment created"
} else {
    Write-Info "Using existing virtual environment"
}

$venvPython = Join-Path $venvPath 'Scripts/python.exe'
if(-not (Test-Path $venvPython)){ Write-Err "Virtual environment python not found at $venvPython"; exit 1 }
Write-Info "Venv Python: $venvPython"

Write-Info "Upgrading pip/setuptools/wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel > $null

Write-Info "Installing core packages (pinned compatibility set)"
$packages = @(
    'faster-whisper==1.2.0',
    'ctranslate2==4.5.0',
    'av==11.0.0',
    'tokenizers',
    'transformers',
    'rich',
    'onnxruntime',
    'huggingface_hub[hf_xet]'
)
& $venvPython -m pip install --upgrade --no-cache $packages
if($LASTEXITCODE -ne 0){ Write-Err "Package installation failed"; exit 1 }
Write-Ok "Dependencies installed"

Write-Info "Validating imports (sanity check)"
$importTest = @"
import importlib, sys, warnings
# Suppress noisy deprecated pkg_resources warning emitted by ctranslate2 loading
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API")
mods = ['faster_whisper','ctranslate2','av','rich']
failed = False
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
        failed = True
if failed:
    sys.exit(1)
print('[SUCCESS] Basic imports passed')
"@
($importTest) | & $venvPython - 2>&1 | ForEach-Object { $_ }
if($LASTEXITCODE -ne 0){ Write-Err "Import validation failed"; exit 1 } else { Write-Ok "Import check passed" }

if($Model){
    Write-Info "Pre-downloading model: $Model (warmup on CPU)"
    $warmup = @"
from faster_whisper import WhisperModel
model = WhisperModel(r'$Model', device='cpu')
print('Model load OK (CPU warmup).')
"@
    ($warmup) | & $venvPython - 2>&1 | ForEach-Object { $_ }
    if($LASTEXITCODE -ne 0){ Write-Warn "Model warmup failed (may still work at runtime)." } else { Write-Ok "Model warmup complete" }
}

if(-not $NoValidate){
    if(Test-Path (Join-Path $projectRoot 'test_clean.py')){
        Write-Info "Running validation script"
        & $venvPython test_clean.py
        if($LASTEXITCODE -ne 0){ Write-Warn "Validation script reported issues" } else { Write-Ok "Validation succeeded" }
    } else {
        Write-Warn "test_clean.py not found, skipping validation"
    }
}

Write-Ok "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Info "Run: .\.venv\Scripts\python.exe whisper_clean.py your_video.mp4 --model small"

if($LaunchShell){
    Write-Info "Launching new PowerShell with virtual environment activated (close it to return)."
    $activate = Join-Path $venvPath 'Scripts/Activate.ps1'
    if(Test-Path $activate){
        # Start a new window so current script context ends cleanly
        Start-Process -FilePath "pwsh" -ArgumentList "-NoLogo","-NoExit","-ExecutionPolicy","Bypass","-Command","& '$activate'; Write-Host '[INFO] Virtual environment active.' -ForegroundColor Cyan" | Out-Null
    } else {
        Write-Warn "Activation script not found at $activate"
    }
}
