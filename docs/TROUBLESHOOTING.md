# Troubleshooting Guide

Comprehensive reference for diagnosing and fixing issues with the Whisper subtitle tool on Windows (CUDA, cuDNN, FFmpeg, Python deps).

---
## 🔍 Quick Diagnostic Flow
1. Run environment validation:
   ```powershell
   .\.venv\Scripts\python.exe test_clean.py
   ```
2. (Optional upcoming) Run GPU diagnostic script:
   ```powershell
   pwsh -File .\scripts\check_cuda.ps1
   ```
3. If model falls back to CPU unexpectedly, check missing DLL list in output.
4. If transcription crashes immediately on GPU, suspect missing cuDNN or version mismatch.

---
## 🧩 Common Error Matrix
| Symptom / Message | Root Cause | Resolution |
|-------------------|------------|-----------|
| `Could not locate cudnn_ops64_9.dll` | cuDNN not installed / not on PATH | Install cuDNN 9 for CUDA 12, copy DLLs to CUDA `bin` or PATH dir |
| `cublas64_12.dll not found` | CUDA Toolkit runtime missing | Install CUDA Toolkit 12.x (Express) |
| `AttributeError: ctranslate2.StorageView` | Version mismatch faster-whisper/ctranslate2 | Pin: `ctranslate2==4.5.0` & `faster-whisper==1.2.0` |
| `No module named 'av._core'` | PyAV Windows wheel/import issue | Use `whisper_clean.py` (no PyAV) |
| FFmpeg command fails | FFmpeg not installed or not on PATH | Install via script / Chocolatey; re-open terminal |
| Empty segments / silent fail on GPU then CPU works | GPU runtime partial (driver only) | Install CUDA Toolkit + cuDNN; ensure DLLs in PATH |
| Out of memory (GPU) | Model too large for VRAM | Use smaller model or `int8_float16` compute |
| Slow CPU fallback unexpectedly | GPU fallback triggered by missing DLLs | Resolve runtime; confirm with `where cudnn_ops64_9.dll` |

---
## 🖥️ Verifying GPU Runtime

### 1. Check GPU Presence
```powershell
nvidia-smi
```
If this fails: install / update NVIDIA driver from GeForce Experience or NVIDIA driver download page.

### 2. Confirm Core CUDA DLLs
```powershell
where cudart64_12.dll
where cublas64_12.dll
where cublasLt64_12.dll
where cudnn_ops64_9.dll
```
All should return a valid path. If only the first 3 exist (no cuDNN): install cuDNN.

### 3. cuDNN Installation Steps
1. Create a free NVIDIA Developer account.
2. Download cuDNN 9.x for Windows (matching CUDA 12) from https://developer.nvidia.com/cudnn.
3. Extract archive; copy the following into your CUDA Toolkit bin folder, e.g.:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X\bin
   ```
   Files include (names may vary slightly with patch version):
   - `cudnn_ops64_9.dll`
   - `cudnn_cnn_infer64_9.dll`
   - `cudnn_cnn_train64_9.dll`
   - `cudnn_adv_infer64_9.dll` (if present)
   - `cudnn_adv_train64_9.dll`
4. Open a new PowerShell and re-run validation.

### 4. PATH Confirmation
```powershell
$env:PATH -split ';' | Select-String "CUDA\\v12" | Sort-Object -Unique
```
Ensure the CUDA `bin` directory appears at least once.

---
## ⚙️ Device & Fallback Logic

| Mode | Behavior |
|------|----------|
| `--device cuda` | Try CUDA once; on failure (unless `--no-fallback`) switch to CPU with compute cascade. |
| `--device-order cuda,igpu,cpu` | Attempt each in order; only last attempt performs internal fallback. `igpu` maps to optimized CPU (`int8` bias). |
| `--no-fallback` | Abort immediately upon first device failure. |

CPU compute cascade sequence when starting with float16 GPU attempt and falling back:
```
float16 (GPU) → int8_float16 (CPU) → int8 → float32
```

### Why Did It Fall Back?
- Missing one or more DLLs: shown explicitly before fallback.
- Runtime initialization error inside model load or first transcription call.
- Forced by user incorrectly specifying device in environment lacking CUDA runtime.

To enforce GPU-only success (CI or benchmarking):
```powershell
.\.venv\Scripts\python.exe whisper_clean.py video.mp4 --device cuda --no-fallback
```
Exit code will be non-zero on failure.

---
## 🧪 Validation & Diagnostics

### Basic Environment Test
```powershell
.\.venv\Scripts\python.exe test_clean.py
```
Outputs model import + DLL presence warnings.

### (Planned) Extended Diagnostic Script
`scripts/check_cuda.ps1` (in progress) will:
- Report GPU model, driver version
- List presence of required DLLs
- Probe Python imports (ctranslate2, faster_whisper)
- Suggest remediation steps with colored output

### Manual DLL Presence Scriptlet
```powershell
$dlls = 'cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll','cudnn_ops64_9.dll'
foreach ($d in $dlls) {
  $found = (where $d 2>$null)
  if ($LASTEXITCODE -eq 0) { Write-Host "$d : OK" -ForegroundColor Green } else { Write-Host "$d : MISSING" -ForegroundColor Yellow }
}
```

---
## 🧱 Version Pinning (Known-Good Set)
Pinned in `setup.ps1` / `requirements.txt` for reproducibility:
```
faster-whisper==1.2.0
ctranslate2==4.5.0
rich
onnxruntime
transformers
tokenizers
numpy
```
If you deviate and encounter issues, revert to this baseline.

---
## 🛠 Regenerating a Clean Environment
```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe test_clean.py
```

---
## 🐢 Performance Checks
| Issue | Potential Cause | Mitigation |
|-------|-----------------|------------|
| Slow GPU despite CUDA | Running on CPU fallback silently | Look for fallback messages; verify DLLs | 
| GPU memory OOM | Model too large | Use smaller model / int8 precision | 
| High CPU usage | Using CPU path (no DLLs) | Install CUDA+cuDNN | 
| Low transcription speed (<1x) | Large model on modest GPU | Try `small`/`medium`, reduce beam size |

---
## 🧪 Benchmark Quick Commands
```powershell
# Medium model GPU
.\.venv\Scripts\python.exe whisper_clean.py sample.mp4 --model medium --beam 5

# Force CPU for comparison
.\.venv\Scripts\python.exe whisper_clean.py sample.mp4 --model medium --device cpu --compute int8_float16
```

---
## ❓ Still Stuck?
Collect this info and open an issue:
- Command used
- Console output (include fallback messages)
- Output of: `nvidia-smi`
- Output of DLL check scriptlet above
- Python version: `python -V`
- List of installed key packages:
  ```powershell
  .\.venv\Scripts\python.exe -m pip show faster-whisper ctranslate2 onnxruntime
  ```

---
## 📌 Future Enhancements
- Finalize `check_cuda.ps1` with JSON output + exit codes
- Add optional telemetry summary file per run
- Provide auto-download/placement helper for cuDNN (license permitting)

---
Happy transcribing! 📝
