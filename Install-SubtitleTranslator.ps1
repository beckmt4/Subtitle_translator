# Install-SubtitleTranslator.ps1
<#
.SYNOPSIS
    Automated installation script for Subtitle Translator

.DESCRIPTION
    This script sets up the Subtitle Translator package with its dependencies,
    configures the Python environment, and validates the installation.

.PARAMETER Model
    Whisper model to download (tiny, base, small, medium, large-v3)

.PARAMETER Force
    Force recreation of virtual environment

.PARAMETER Python
    Path to Python executable (default: auto-detect)

.PARAMETER NoGPU
    Skip GPU dependencies installation (for CPU-only setup)

.PARAMETER MinimalDeps
    Install only core dependencies (skip optional packages)

.EXAMPLE
    .\Install-SubtitleTranslator.ps1 -Model medium

.EXAMPLE
    .\Install-SubtitleTranslator.ps1 -Force -NoGPU

.NOTES
    Requires PowerShell 5.1+ and Python 3.8+
#>

[CmdletBinding()]
param(
    [ValidateSet('tiny', 'base', 'small', 'medium', 'large-v3')]
    [string]$Model = "medium",
    
    [switch]$Force,
    
    [string]$Python,
    
    [switch]$NoGPU,
    
    [switch]$MinimalDeps
)

# Set execution policy for current process if needed
$currentPolicy = Get-ExecutionPolicy -Scope Process
if ($currentPolicy -eq "Restricted") {
    Write-Host "Setting execution policy to RemoteSigned for current process..."
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
}

# Find Python executable
if (-not $Python) {
    $pythonCommands = @("python", "python3")
    
    foreach ($cmd in $pythonCommands) {
        try {
            $pythonVersion = & $cmd --version 2>&1
            if ($pythonVersion -match "Python 3\.[89]|Python 3.1[0-9]") {
                $Python = $cmd
                break
            }
        }
        catch {
            # Command not found, try next
        }
    }
    
    if (-not $Python) {
        Write-Host "Error: Python 3.8+ not found. Please install Python or specify path with -Python" -ForegroundColor Red
        exit 1
    }
}

# Check Python version
$pythonVersionOutput = & $Python --version 2>&1
Write-Host "Using $pythonVersionOutput"

if (-not ($pythonVersionOutput -match "Python 3\.[89]|Python 3.1[0-9]")) {
    Write-Host "Error: Python 3.8+ required, found: $pythonVersionOutput" -ForegroundColor Red
    exit 1
}

# Create virtual environment
$venvPath = ".venv"
if ((Test-Path $venvPath) -and $Force) {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    & $Python -m venv $venvPath
    if (-not $?) {
        Write-Host "Error creating virtual environment. Please install venv: pip install virtualenv" -ForegroundColor Red
        exit 1
    }
}

# Determine pip path
$pipPath = if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Join-Path $venvPath "Scripts\pip.exe"
} else {
    Join-Path $venvPath "bin\pip"
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
& $pipPath install --upgrade pip

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Green

# Core packages - install --no-deps to avoid PyAV on Windows
& $pipPath install faster-whisper --no-deps
& $pipPath install ctranslate2 huggingface_hub tokenizers tqdm onnxruntime rich typer

# Install GPU dependencies if requested
if (-not $NoGPU) {
    Write-Host "Installing GPU dependencies..." -ForegroundColor Green
    & $pipPath install torch --extra-index-url https://download.pytorch.org/whl/cu118
}

# Install optional dependencies unless minimal
if (-not $MinimalDeps) {
    Write-Host "Installing optional dependencies..." -ForegroundColor Green
    & $pipPath install transformers sentencepiece sacremoses
}

# Install dev dependencies
& $pipPath install pytest pytest-cov black isort

# Install the package in development mode
Write-Host "Installing subtitle-translator package..." -ForegroundColor Green
& $pipPath install -e .

# Download model if specified
if ($Model) {
    Write-Host "Downloading Whisper model: $Model..." -ForegroundColor Green
    $pythonScript = @"
from faster_whisper import WhisperModel
model = WhisperModel('$Model', device='cpu')
print(f"Model {model.model_size} downloaded successfully!")
"@
    
    & $Python -c $pythonScript
}

# Verify FFmpeg installation
Write-Host "Checking FFmpeg installation..." -ForegroundColor Green
try {
    $ffmpegVersion = ffmpeg -version 2>&1
    if ($ffmpegVersion -match "ffmpeg version") {
        Write-Host "FFmpeg found: $($ffmpegVersion -split "`n" | Select-Object -First 1)" -ForegroundColor Green
    } else {
        Write-Host "FFmpeg not detected. Please install FFmpeg and add it to your PATH." -ForegroundColor Yellow
        Write-Host "Download from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    }
} catch {
    Write-Host "FFmpeg not found. Please install FFmpeg and add it to your PATH." -ForegroundColor Red
    Write-Host "Download from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

# Run validation script
Write-Host "Validating installation..." -ForegroundColor Green
$venvPython = if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Join-Path $venvPath "Scripts\python.exe"
} else {
    Join-Path $venvPath "bin\python"
}

& $venvPython test_clean.py

# Show usage instructions
Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "`nTo use subtitle-translator:" -ForegroundColor Cyan
Write-Host "  1. Activate the virtual environment:" -ForegroundColor White
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor White
} else {
    Write-Host "     source .venv/bin/activate" -ForegroundColor White
}
Write-Host "  2. Run the tool:" -ForegroundColor White
Write-Host "     subtitle-translator transcribe video.mp4" -ForegroundColor White
Write-Host "     subtitle-translator translate video.mp4 --target-language eng_Latn" -ForegroundColor White
Write-Host "`nOr directly without activation:" -ForegroundColor Cyan
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Write-Host "  .\.venv\Scripts\python.exe -m subtitle_translator transcribe video.mp4" -ForegroundColor White
} else {
    Write-Host "  ./.venv/bin/python -m subtitle_translator transcribe video.mp4" -ForegroundColor White
}