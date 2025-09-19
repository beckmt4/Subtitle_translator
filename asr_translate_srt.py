#!/usr/bin/env python
"""
Two-pass pipeline:
1) ASR (e.g. Japanese) via faster-whisper
2) MT (JP -> EN) via Hugging Face seq2seq model (e.g. NLLB / M2M100)
3) Subtitle shaping with quality constraints:
   - Max chars per line
   - Max lines
   - Max characters-per-second (CPS)
   - Minimum duration
   - Minimum inter-segment gap
   - Optional fallback to Whisper internal translation if MT unavailable

Requirements (extra):
  transformers, sentencepiece, (optional) torch (GPU), rich
  For large MT models you need significant VRAM (3.3B NLLB model is heavy).
"""
import argparse
import os
import sys
import math
import textwrap
import tempfile
import subprocess
import threading
import time
import contextlib
from pathlib import Path
from typing import List, Dict, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

try:
    from faster_whisper import WhisperModel
except ImportError:
    console.print("[red]faster-whisper not installed.[/red]")
    sys.exit(1)

# Optional torch + transformers load (lazy)
def load_mt_model(model_name: str, device: str, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None, arch_pref: str = "auto"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
    except Exception as e:
        console.print(f"[yellow]MT dependencies missing (torch/transformers). Fallback to Whisper internal translation. ({e})[/yellow]")
        return None, None, None

    # Build tokenizer kwargs for models that support language codes (e.g., NLLB)
    def _bcp47(code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        code = code.lower()
        # Minimal mapping; expand as needed
        mapping = {
            "ja": "jpn_Jpan",
            "en": "eng_Latn",
            "zh": "zho_Hans",
            "ko": "kor_Hang",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "es": "spa_Latn",
        }
        return mapping.get(code, None)

    try:
        # Detect architecture (seq2seq vs decoder-only) if possible
        use_decoder_only = False
        try:
            cfg = AutoConfig.from_pretrained(model_name)
            arch_name = None
            if hasattr(cfg, "architectures") and cfg.architectures:
                arch_name = ",".join(cfg.architectures)
            if arch_pref == "decoder":
                use_decoder_only = True
            elif arch_pref == "seq2seq":
                use_decoder_only = False
            else:
                use_decoder_only = bool(arch_name and ("CausalLM" in arch_name or "ForCausalLM" in arch_name))
        except Exception:
            use_decoder_only = (arch_pref == "decoder")

        tok_kwargs = {}
        if "nllb" in model_name.lower():
            src = _bcp47(src_lang)
            tgt = _bcp47(tgt_lang)
            if src:
                tok_kwargs["src_lang"] = src
            if tgt:
                tok_kwargs["tgt_lang"] = tgt
        tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if use_decoder_only:
            with contextlib.suppress(Exception):
                tok.padding_side = "left"
            mt = AutoModelForCausalLM.from_pretrained(model_name)
            mt_arch = "decoder"
        else:
            mt = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            mt_arch = "seq2seq"
        if device == "cuda" and torch.cuda.is_available():
            try:
                mt = mt.half().to("cuda")
            except Exception:
                # Some models may not support half precision; fall back to float32 on CUDA
                mt = mt.to("cuda")
        else:
            mt = mt.to("cpu")
        mt.eval()
        return tok, mt, mt_arch
    except Exception as e:
        console.print(f"[yellow]Failed to load MT model {model_name}: {e}. Falling back to Whisper translation.[/yellow]")
        return None, None, None

def print_diagnostics(asr_model_obj, mt_tok, mt_model, args, mt_arch: Optional[str] = None):
    """Print runtime diagnostics about devices, precision, and availability."""
    try:
        import torch
        torch_available = torch.cuda.is_available()
        torch_version = getattr(torch, '__version__', 'unknown')
        gpu_name = torch.cuda.get_device_name(0) if torch_available else 'N/A'
    except Exception:
        torch_available = False
        torch_version = 'missing'
        gpu_name = 'N/A'
    # WhisperModel exposes model_size and device via attributes we stored in args indirectly
    console.rule("Diagnostics")
    console.print(f"ASR Model: {args.asr_model}")
    console.print(f"ASR Device Requested: {args.device}")
    console.print(f"CUDA Available (torch): {torch_available}")
    console.print(f"Torch Version: {torch_version}")
    console.print(f"GPU: {gpu_name}")
    if mt_model is not None:
        try:
            import torch
            console.print(f"MT Model: {args.mt_model} (device={mt_model.device}, dtype={next(mt_model.parameters()).dtype}, arch={mt_arch or 'unknown'})")
        except Exception:
            console.print(f"MT Model: {args.mt_model} (device=unknown, arch={mt_arch or 'unknown'})")
    else:
        console.print("MT Model: (none / using Whisper translation)")
    console.rule()

def _lang_name(code: Optional[str]) -> str:
    mapping = {"ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish"}
    return mapping.get((code or "").lower(), code or "")

def build_decoder_prompt(text: str, src_lang: str, tgt_lang: str, tok) -> str:
    """Prompt for decoder-only MT models. Uses chat template when available."""
    src_name = _lang_name(src_lang)
    tgt_name = _lang_name(tgt_lang)
    try:
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            messages = [
                {"role": "system", "content": f"You are a professional translation engine. Translate the user message from {src_name} to {tgt_name}. Output only the translation with no explanations."},
                {"role": "user", "content": text},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return (
        f"Translate the following text from {src_name} to {tgt_name}.\n"
        f"Text: {text}\n"
        f"Translation:"
    )

def run_ffmpeg_extract(input_path: Path, target_sr=16000) -> Path:
    wav_path = input_path.with_suffix(".asr.wav")
    if wav_path.exists():
        return wav_path
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "wav",
        str(wav_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg extraction failed: {e}[/red]")
        sys.exit(1)
    return wav_path

def time_format(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def shape_text(
    text: str,
    start: float,
    end: float,
    max_line: int,
    max_lines: int,
    max_cps: float,
    min_duration: float
) -> str:
    dur = max(min_duration, end - start)
    # Limit by CPS (characters per second)
    allowed_chars = int(min(max_line * max_lines, max_cps * dur))
    # Wrap width heuristically
    target_width = min(max_line, max(10, allowed_chars // max_lines if max_lines else max_line))
    wrapped = textwrap.wrap(text.strip(), width=target_width)
    if not wrapped:
        wrapped = [text.strip()]
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
    return "\n".join(wrapped)

def batch_translate(
    lines: List[str],
    tok,
    mt,
    batch_size: int,
    max_new_tokens: int,
    beam_size: int,
    verbose: bool = False,
    arch: str = "seq2seq",
    src_lang: Optional[str] = None,
    tgt_lang: Optional[str] = None,
) -> List[str]:
    import torch
    outputs: List[str] = []
    tried_cpu_fallback = False
    
    # Handle empty input case
    if not lines:
        return outputs
        
    # Auto-adjust batch size if needed based on input
    if batch_size > len(lines):
        if verbose:
            console.print(f"[yellow]Adjusting batch size from {batch_size} to {len(lines)} (total segments)[/yellow]")
        batch_size = max(1, len(lines))
    
    # Progress tracking for verbose mode
    total_chunks = (len(lines) + batch_size - 1) // batch_size
    
    for i in range(0, len(lines), batch_size):
        chunk = lines[i:i+batch_size]
        if verbose:
            console.print(f"[cyan]Translating batch {(i//batch_size)+1}/{total_chunks} ({len(chunk)} segments)[/cyan]")
        
        forced_bos_token_id = None
        try:
            if arch == "decoder":
                prompts = [build_decoder_prompt(txt, src_lang or "ja", tgt_lang or "en", tok) for txt in chunk]
                inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(mt.device)
                with torch.inference_mode():
                    gen = mt.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=max(1, beam_size or 1),
                        eos_token_id=getattr(tok, "eos_token_id", None),
                        pad_token_id=getattr(tok, "pad_token_id", None),
                    )
                sequences = gen.sequences if hasattr(gen, "sequences") else gen
                input_len = inputs.input_ids.shape[1]
                cont = sequences[:, input_len:]
                decoded = tok.batch_decode(cont, skip_special_tokens=True)
                outputs.extend([d.strip() for d in decoded])
                torch.cuda.empty_cache()
            else:
                inputs = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(mt.device)
                # Determine target language BOS if available (NLLB/M2M100/etc.)
                if hasattr(tok, "lang_code_to_id"):
                    forced_bos_token_id = tok.lang_code_to_id.get("eng_Latn", None)
                if forced_bos_token_id is None and hasattr(tok, "get_lang_id"):
                    try:
                        forced_bos_token_id = tok.get_lang_id("en")
                    except Exception:
                        forced_bos_token_id = None
                with torch.inference_mode():
                    gen = mt.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=beam_size,
                        forced_bos_token_id=forced_bos_token_id
                    )
                decoded = tok.batch_decode(gen, skip_special_tokens=True)
                outputs.extend(decoded)
                torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            # Try to recover by reducing batch size
            if len(chunk) > 1:
                if verbose:
                    console.print(f"[yellow]GPU OOM with batch size {len(chunk)}, retrying with smaller batches[/yellow]")
                # Process one by one
                for text in chunk:
                    try:
                        if arch == "decoder":
                            prompt = build_decoder_prompt(text, src_lang or "ja", tgt_lang or "en", tok)
                            single_input = tok([prompt], return_tensors="pt").to(mt.device)
                            with torch.inference_mode():
                                single_gen = mt.generate(
                                    **single_input,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=False,
                                    num_beams=1,
                                    eos_token_id=getattr(tok, "eos_token_id", None),
                                    pad_token_id=getattr(tok, "pad_token_id", None),
                                )
                            sequences = single_gen.sequences if hasattr(single_gen, "sequences") else single_gen
                            input_len = single_input.input_ids.shape[1]
                            cont = sequences[:, input_len:]
                            single_decoded = tok.batch_decode(cont, skip_special_tokens=True)
                        else:
                            single_input = tok([text], return_tensors="pt").to(mt.device)
                            with torch.inference_mode():
                                single_gen = mt.generate(
                                    **single_input,
                                    max_new_tokens=max_new_tokens,
                                    num_beams=max(1, beam_size-2),  # Reduce beam size to conserve memory
                                    forced_bos_token_id=forced_bos_token_id
                                )
                            single_decoded = tok.batch_decode(single_gen, skip_special_tokens=True)
                        outputs.extend([t.strip() for t in single_decoded])
                        torch.cuda.empty_cache()
                    except Exception as e:
                        # Last resort - return empty translation for this segment
                        console.print(f"[red]Failed to translate segment: {e}[/red]")
                        outputs.append("") 
                        torch.cuda.empty_cache()
            else:
                # Even a single segment is too large - add placeholder
                console.print(f"[red]GPU OOM with single segment, skipping[/red]")
                outputs.append("")
                torch.cuda.empty_cache()
        except Exception as e:
            console.print(f"[red]Error in translation batch: {e}[/red]")
            # Attempt a one-time CPU fallback for MT if running on CUDA
            try:
                import torch as _torch
                if mt.device.type == "cuda" and not tried_cpu_fallback:
                    console.print("[yellow]Falling back to CPU for MT due to error. This will be slower.[/yellow]")
                    mt.to("cpu")
                    tried_cpu_fallback = True
                    # Retry current chunk on CPU one-by-one to avoid memory spikes
                    for text in chunk:
                        try:
                            if arch == "decoder":
                                prompt = build_decoder_prompt(text, src_lang or "ja", tgt_lang or "en", tok)
                                single_input = tok([prompt], return_tensors="pt")
                                with _torch.inference_mode():
                                    single_gen = mt.generate(
                                        **single_input,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        num_beams=1,
                                        eos_token_id=getattr(tok, "eos_token_id", None),
                                        pad_token_id=getattr(tok, "pad_token_id", None),
                                    )
                                sequences = single_gen.sequences if hasattr(single_gen, "sequences") else single_gen
                                input_len = single_input.input_ids.shape[1]
                                cont = sequences[:, input_len:]
                                single_decoded = tok.batch_decode(cont, skip_special_tokens=True)
                            else:
                                single_input = tok([text], return_tensors="pt")
                                with _torch.inference_mode():
                                    single_gen = mt.generate(
                                        **single_input,
                                        max_new_tokens=max_new_tokens,
                                        num_beams=max(1, beam_size-2),
                                        forced_bos_token_id=forced_bos_token_id
                                    )
                                single_decoded = tok.batch_decode(single_gen, skip_special_tokens=True)
                            outputs.extend(single_decoded)
                        except Exception as e2:
                            console.print(f"[red]CPU fallback also failed for a segment: {e2}[/red]")
                            outputs.append("")
                    continue
            except Exception:
                pass
            # If we can't recover, add empty translations to maintain segment count
            outputs.extend([""] * len(chunk))
            try:
                import torch as _torch
                _torch.cuda.empty_cache()
            except Exception:
                pass
    return outputs

def write_srt(
    segments: List[Dict],
    out_path: Path,
    config
):
    idx = 1
    prev_end = 0.0
    min_gap = config.min_gap
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            txt = seg["text"]
            # enforce gap
            if start - prev_end < min_gap:
                shift = (min_gap - (start - prev_end))
                start += shift
                if start >= end:
                    end = start + config.min_duration
            shaped = shape_text(
                txt,
                start,
                end,
                config.max_line_chars,
                config.max_lines,
                config.max_cps,
                config.min_duration
            )
            f.write(f"{idx}\n{time_format(start)} --> {time_format(end)}\n{shaped}\n\n")
            prev_end = end
            idx += 1

def main():
    parser = argparse.ArgumentParser(
        description="Two-pass ASR + MT (e.g. Japanese→English) with subtitle quality controls."
    )
    parser.add_argument("input", help="Input media file (video/audio).")
    parser.add_argument("-o", "--output", help="Output SRT path (default: input_name.en.srt)")
    parser.add_argument("--language", default="ja", help="Source language code (default: ja)")
    parser.add_argument("--asr-model", default="medium", help="Whisper model size or path.")
    parser.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M",
                        help="MT model name (HuggingFace). Use larger (e.g. facebook/nllb-200-3.3B) if VRAM allows.")
    parser.add_argument("--mt-arch", choices=["auto", "seq2seq", "decoder"], default="auto",
                        help="Force MT architecture selection if auto-detection fails.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Primary device for ASR (and MT if torch available).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for Whisper decoding.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Whisper temperature.")
    parser.add_argument("--batch-size", type=int, default=8, help="MT batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="MT generation max_new_tokens.")
    parser.add_argument("--mt-beams", type=int, default=4, help="MT beam size.")
    # Accept both --compute-type and --compute for compatibility with docs
    parser.add_argument("--compute-type", dest="compute_type", default=None,
                        help="Override compute_type (default: float16 on CUDA, int8_float16 CPU).")
    parser.add_argument("--compute", dest="compute_type", default=None,
                        help="Alias for --compute-type.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                        help="Whisper task if MT model not used.")
    parser.add_argument("--no-mt", action="store_true",
                        help="Skip external MT and rely on Whisper translation (if task=translate).")

    # Quality controls
    parser.add_argument("--max-line-chars", type=int, default=42)
    parser.add_argument("--max-lines", type=int, default=2)
    parser.add_argument("--max-cps", type=float, default=17.0, help="Characters per second cap.")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--min-gap", type=float, default=0.09, help="Minimum gap between subtitles (seconds).")

    parser.add_argument("--vad-filter", action="store_true", help="Enable VAD filtering in Whisper.")
    parser.add_argument("--patience", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline logic without writing SRT.")
    parser.add_argument("--remux", action="store_true", help="After writing SRT, remux into new media file with .subbed before extension.")
    parser.add_argument("--remux-language", help="Subtitle track language code (default: en if translated else source language).")
    parser.add_argument("--remux-overwrite", action="store_true", help="Overwrite existing .subbed file if present.")
    parser.add_argument("--model-progress", action="store_true", help="Show approximate model download progress (heuristic).")
    parser.add_argument("--diag", action="store_true", help="Print diagnostics about devices, precision, and MT/ASR placement.")

    args = parser.parse_args()
    media_path = Path(args.input)
    if not media_path.exists():
        console.print(f"[red]Input not found: {media_path}[/red]")
        sys.exit(1)
    out_path = Path(args.output) if args.output else media_path.with_suffix(".en.srt")

    # Compute type selection
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if args.device == "cuda" else "int8_float16"

    # Create two separate progress displays for ASR and MT phases
    # First progress display for ASR phase
    asr_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )
    
    # Second progress display for MT phase
    mt_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )
    
    segments_struct: List[Dict] = []

    # First phase: ASR (Speech Recognition)
    with asr_progress:
        console.print("[bold blue]Phase 1: Automatic Speech Recognition[/bold blue]")
        t_model = asr_progress.add_task("Loading ASR model", total=100 if args.model_progress else None)

        stop_flag = threading.Event()
        model_size_map = {
            # Estimated compressed sizes (MB) of model directories for heuristic percentages
            'tiny': 80,
            'base': 150,
            'small': 500,
            'medium': 1500,
            'large': 3100,
            'large-v3': 3100,
        }
        est_total_mb = None
        # Normalize key from asr_model param (may be path or HF id); only apply heuristic if simple token
        if args.asr_model in model_size_map:
            est_total_mb = model_size_map[args.asr_model]

        def poll_cache_progress():
            # Poll Hugging Face cache for size accumulation pre-init
            # WhisperModel caches under ~/.cache/huggingface/hub/models--Systran--faster-whisper-* pattern
            # We'll search for fastest matching folder containing the model token
            token = args.asr_model
            root = Path.home() / '.cache' / 'huggingface' / 'hub'
            while not stop_flag.is_set():
                try:
                    candidates = []
                    if root.exists():
                        for p in root.iterdir():
                            name = p.name.lower()
                            if token.lower() in name and p.is_dir():
                                candidates.append(p)
                    total_bytes = 0
                    for c in candidates:
                        for fp in c.rglob('*'):
                            if fp.is_file():
                                # skip gigantic partial downloads maybe locked
                                with contextlib.suppress(Exception):
                                    total_bytes += fp.stat().st_size
                    if est_total_mb and total_bytes > 0 and asr_progress.tasks[t_model].total:
                        mb = total_bytes / (1024*1024)
                        pct = min(99.0, (mb / est_total_mb) * 100.0)
                        asr_progress.update(t_model, completed=pct)
                except Exception:
                    pass
                time.sleep(0.6)

        import contextlib
        cache_thread = None
        if args.model_progress:
            cache_thread = threading.Thread(target=poll_cache_progress, daemon=True)
            cache_thread.start()
        try:
            asr_model = WhisperModel(
                args.asr_model,
                device=args.device,
                compute_type=compute_type
            )
        except Exception as e:
            # Fix: properly terminated f-string (previous version had newline inside causing syntax error)
            console.print(f"[yellow]ASR model load failed on {args.device}: {e}. Retrying on CPU int8_float16.[/yellow]")
            try:
                asr_model = WhisperModel(
                    args.asr_model,
                    device="cpu",
                    compute_type="int8_float16"
                )
            except Exception as e2:
                stop_flag.set()
                if cache_thread:
                    cache_thread.join(timeout=1)
                console.print(f"[red]Failed to load model on CPU: {e2}[/red]")
                sys.exit(1)
        finally:
            stop_flag.set()
            if cache_thread:
                cache_thread.join(timeout=2)
        if args.model_progress and asr_progress.tasks[t_model].total:
            asr_progress.update(t_model, completed=100)
        if args.diag:
            # Print diagnostics after ASR model is loaded (MT may not be loaded yet)
            print_diagnostics(asr_model, None, None, args)

        t_audio = asr_progress.add_task("Extracting audio", total=100)
        wav_path = run_ffmpeg_extract(media_path)
        asr_progress.update(t_audio, completed=100)

        # Create a pulsing progress bar for transcription since we can't get incremental updates
        t_asr = asr_progress.add_task("Transcribing (GPU working - please wait...)", total=100)
        
        # Start a background thread to update the progress bar to show activity
        transcribe_done = threading.Event()
        
        def update_transcribe_progress():
            # Pulsing progress bar between 0-90% while transcription is running
            progress = 0
            pulse_step = 2
            direction = 1  # 1 for increasing, -1 for decreasing
            
            while not transcribe_done.is_set():
                # Update progress in a pulsing pattern
                progress += pulse_step * direction
                
                # Reverse direction at bounds (0% and 90%)
                if progress >= 90:
                    direction = -1
                elif progress <= 0:
                    direction = 1
                    
                # Update the progress bar
                asr_progress.update(t_asr, completed=progress)
                time.sleep(0.1)
        
        # Start the progress update thread
        progress_thread = threading.Thread(target=update_transcribe_progress, daemon=True)
        progress_thread.start()
        
        try:
            gen = asr_model.transcribe(
                str(wav_path),
                language=args.language,
                task=args.task if args.no_mt else "transcribe",
                beam_size=args.beam_size,
                temperature=args.temperature,
                patience=args.patience,
                vad_filter=args.vad_filter,
                condition_on_previous_text=True
            )
            raw_segments, info = gen
            
            # Signal the progress thread to stop
            transcribe_done.set()
            progress_thread.join(timeout=1)
            
            # Set to 100% when complete
            asr_progress.update(t_asr, completed=100, description="Transcribing [green](complete)")
            
        except Exception as e:
            # Signal the progress thread to stop
            transcribe_done.set()
            progress_thread.join(timeout=1)
            
            # Show error
            asr_progress.update(t_asr, completed=0, description="Transcribing [red](failed)")
            console.print(f"[red]Transcription failed: {e}[/red]")
            sys.exit(1)

        for s in raw_segments:
            segments_struct.append({
                "start": s.start,
                "end": s.end,
                "src": s.text.strip()
            })

        # Start the second phase: MT (Machine Translation)
        console.print("\n[bold green]Phase 2: Machine Translation[/bold green]")
        
        use_external_mt = not args.no_mt and args.task != "translate"
        translations: List[str] = []
        
        with mt_progress:
            if use_external_mt:
                t_mt = mt_progress.add_task("Loading MT model", total=None)
                # For NLLB we prefer explicit src/tgt where possible
                tgt_lang = "en"
                tok, mt, mt_arch = load_mt_model(args.mt_model, args.device, src_lang=args.language, tgt_lang=tgt_lang, arch_pref=args.mt_arch)
                mt_progress.update(t_mt, completed=100)
                if args.diag:
                    print_diagnostics(asr_model, tok, mt, args, mt_arch=mt_arch)

                if tok and mt:
                    # For external MT, we can show per-segment progress since we process each one
                    t_translate = mt_progress.add_task("Translating segments", total=len(segments_struct))
                    batch_texts = [s["src"] for s in segments_struct]
                    try:
                        translated = batch_translate(
                            batch_texts,
                            tok,
                            mt,
                            batch_size=args.batch_size,
                            max_new_tokens=args.max_new_tokens,
                            beam_size=args.mt_beams,
                            verbose=args.verbose,
                            arch=(mt_arch or "seq2seq"),
                            src_lang=args.language,
                            tgt_lang=tgt_lang
                        )
                        for i, tr in enumerate(translated):
                            segments_struct[i]["text"] = tr.strip()
                            mt_progress.advance(t_translate)
                    except Exception as e:
                        console.print(f"[yellow]External MT failed: {e}. Falling back to Whisper internal translation pass.[/yellow]")
                        # second pass: internal translation
                        translations = []
                        
                        # Create pulsing progress bar for Whisper translation
                        t_second = mt_progress.add_task("Whisper translation pass (processing...)", total=100)
                        
                        # Start a background thread to update the progress bar to show activity
                        translate_done = threading.Event()
                        
                        def update_translate_progress():
                            progress = 0
                            pulse_step = 3
                            direction = 1
                            
                            while not translate_done.is_set():
                                progress += pulse_step * direction
                                if progress >= 95:
                                    direction = -1
                                elif progress <= 0:
                                    direction = 1
                                mt_progress.update(t_second, completed=progress)
                                time.sleep(0.15)
                        
                        # Start the progress update thread
                        translate_thread = threading.Thread(target=update_translate_progress, daemon=True)
                        translate_thread.start()
                        
                        try:
                            # re-run with task=translate, using original audio
                            gen2 = asr_model.transcribe(
                                str(wav_path),
                                language=args.language,
                                task="translate",
                                beam_size=args.beam_size,
                                temperature=args.temperature,
                                patience=args.patience,
                                vad_filter=args.vad_filter,
                                condition_on_previous_text=True
                            )
                            seg2, _ = gen2
                            
                            # Signal the progress thread to stop
                            translate_done.set()
                            translate_thread.join(timeout=1)
                            
                            mapped = list(seg2)
                            # naive alignment by order
                            for i, s2 in enumerate(mapped):
                                if i < len(segments_struct):
                                    segments_struct[i]["text"] = s2.text.strip()
                                    
                            # Update progress bar to show completion
                            mt_progress.update(t_second, completed=100, description="Whisper translation [green](complete)")
                        
                        except Exception as e:
                            # Signal the progress thread to stop
                            translate_done.set()
                            translate_thread.join(timeout=1)
                            
                            # Show error in progress bar
                            mt_progress.update(t_second, completed=0, description="Whisper translation [red](failed)")
                            console.print(f"[red]Whisper translation failed: {e}[/red]")
                            sys.exit(1)
                else:
                    # fallback path: internal translation
                    # Create pulsing progress bar for Whisper translation
                    t_fallback = mt_progress.add_task("Whisper translation pass (processing...)", total=100)
                    
                    # Start a background thread to update the progress bar to show activity
                    translate_done = threading.Event()
                    
                    def update_translate_progress():
                        progress = 0
                        pulse_step = 3
                        direction = 1
                        
                        while not translate_done.is_set():
                            progress += pulse_step * direction
                            if progress >= 95:
                                direction = -1
                            elif progress <= 0:
                                direction = 1
                            mt_progress.update(t_fallback, completed=progress)
                            time.sleep(0.15)
                    
                    # Start the progress update thread
                    translate_thread = threading.Thread(target=update_translate_progress, daemon=True)
                    translate_thread.start()
                    
                    try:
                        gen2 = asr_model.transcribe(
                            str(wav_path),
                            language=args.language,
                            task="translate",
                            beam_size=args.beam_size,
                            temperature=args.temperature,
                            patience=args.patience,
                            vad_filter=args.vad_filter,
                            condition_on_previous_text=True
                        )
                        seg2, _ = gen2
                        
                        # Signal the progress thread to stop
                        translate_done.set()
                        translate_thread.join(timeout=1)
                        
                        mapped = list(seg2)
                        for i, s2 in enumerate(mapped):
                            if i < len(segments_struct):
                                segments_struct[i]["text"] = s2.text.strip()
                                
                        # Show completion
                        mt_progress.update(t_fallback, completed=100, description="Whisper translation [green](complete)")
                    
                    except Exception as e:
                        # Signal the progress thread to stop
                        translate_done.set()
                        translate_thread.join(timeout=1)
                        
                        # Show error
                        mt_progress.update(t_fallback, completed=0, description="Whisper translation [red](failed)")
                        console.print(f"[red]Whisper translation failed: {e}[/red]")
                        sys.exit(1)
            else:
                # We already requested translate in first pass (args.task == translate or user forced no_mt)
                t_skip = mt_progress.add_task("Skipping external MT (using direct Whisper translation)", total=100)
                for s in segments_struct:
                    s["text"] = s["src"]
                mt_progress.update(t_skip, completed=100)
        
        # Phase 3: Output Generation
        console.print("\n[bold cyan]Phase 3: Subtitle Formatting and Output[/bold cyan]")
        
        # Create a third progress bar for output phase
        output_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        )
        
        with output_progress:
            t_shape = output_progress.add_task("Shaping SRT", total=len(segments_struct))
            final_segments = []
            for seg in segments_struct:
                final_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                })
                output_progress.advance(t_shape)

            if args.dry_run:
                console.print("[green]Dry run complete (no file written).[/green]")
                return

            t_write = output_progress.add_task("Writing SRT", total=100)
            write_srt(final_segments, out_path, args)
            output_progress.update(t_write, completed=100)

            # Handle remuxing in the same progress group
            if args.remux and not args.dry_run:
                t_remux = output_progress.add_task("Remuxing video with subtitles", total=100)
                remux_lang = args.remux_language or ("en" if (not args.no_mt or args.task == "translate") else (args.language or "und"))
                remux_target = media_path.with_name(f"{media_path.stem}.subbed{media_path.suffix}")
                ff_cmd = [
                    "ffmpeg",
                    "-y" if args.remux_overwrite else "-n",
                    "-i", str(media_path),
                    "-i", str(out_path),
                    "-map", "0", "-map", "1:0",
                    "-c", "copy",
                    "-c:s:1", "srt",
                    "-metadata:s:s:1", f"language={remux_lang}",
                    str(remux_target)
                ]
                r = subprocess.run(ff_cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    console.print(f"[yellow]Remux failed: {r.stderr.splitlines()[-1] if r.stderr else 'ffmpeg error'}[/yellow]")
                    output_progress.update(t_remux, completed=100, description="[red]Remuxing failed")
                else:
                    output_progress.update(t_remux, completed=100)
                    console.print(f"[green]✓ Remux complete: {remux_target}[/green]")

    console.print(f"[bold green]Done.[/bold green] Wrote: {out_path}")

if __name__ == "__main__":
    main()
