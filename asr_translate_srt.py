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
def load_mt_model(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        console.print(f"[yellow]MT dependencies missing (torch/transformers). Fallback to Whisper internal translation. ({e})[/yellow]")
        return None, None

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mt = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "cuda" and torch.cuda.is_available():
            mt = mt.half().to("cuda")
        else:
            mt = mt.to("cpu")
        return tok, mt
    except Exception as e:
        console.print(f"[yellow]Failed to load MT model {model_name}: {e}. Falling back to Whisper translation.[/yellow]")
        return None, None

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
    beam_size: int
) -> List[str]:
    import torch
    outputs: List[str] = []
    for i in range(0, len(lines), batch_size):
        chunk = lines[i:i+batch_size]
        inputs = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(mt.device)
        gen = mt.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size
        )
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)
        torch.cuda.empty_cache()
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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Primary device for ASR (and MT if torch available).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for Whisper decoding.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Whisper temperature.")
    parser.add_argument("--batch-size", type=int, default=8, help="MT batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="MT generation max_new_tokens.")
    parser.add_argument("--mt-beams", type=int, default=4, help="MT beam size.")
    parser.add_argument("--compute-type", default=None,
                        help="Override compute_type (default: float16 on CUDA, int8_float16 CPU).")
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

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=False,
        console=console
    )
    segments_struct: List[Dict] = []

    with progress:
        t_model = progress.add_task("Loading ASR model", total=None)
        try:
            asr_model = WhisperModel(
                args.asr_model,
                device=args.device,
                compute_type=compute_type
            )
        except Exception as e:
            console.print(f"[yellow]ASR model load failed on {args.device}: {e}. Retrying on CPU int8_float16.[/yellow]")
            try:
                asr_model = WhisperModel(
                    args.asr_model,
                    device="cpu",
                    compute_type="int8_float16"
                )
            except Exception as e2:
                console.print(f"[red]Failed to load model on CPU: {e2}[/red]")
                sys.exit(1)
        progress.update(t_model, completed=100)

        t_audio = progress.add_task("Extracting audio", total=100)
        wav_path = run_ffmpeg_extract(media_path)
        progress.update(t_audio, completed=100)

        t_asr = progress.add_task("Transcribing", total=None)
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
        except Exception as e:
            console.print(f"[red]Transcription failed: {e}[/red]")
            sys.exit(1)

        for s in raw_segments:
            segments_struct.append({
                "start": s.start,
                "end": s.end,
                "src": s.text.strip()
            })
        progress.update(t_asr, completed=100)

        use_external_mt = not args.no_mt and args.task != "translate"
        translations: List[str] = []
        if use_external_mt:
            t_mt = progress.add_task("Loading MT model", total=None)
            tok, mt = load_mt_model(args.mt_model, args.device)
            progress.update(t_mt, completed=100)

            if tok and mt:
                t_translate = progress.add_task("Translating segments", total=len(segments_struct))
                batch_texts = [s["src"] for s in segments_struct]
                try:
                    translated = batch_translate(
                        batch_texts,
                        tok,
                        mt,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        beam_size=args.mt_beams
                    )
                    for i, tr in enumerate(translated):
                        segments_struct[i]["text"] = tr.strip()
                        progress.advance(t_translate)
                except Exception as e:
                    console.print(f"[yellow]External MT failed: {e}. Falling back to Whisper internal translation pass.[/yellow]")
                    # second pass: internal translation
                    translations = []
                    t_second = progress.add_task("Whisper translation pass", total=None)
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
                    mapped = list(seg2)
                    # naive alignment by order
                    for i, s2 in enumerate(mapped):
                        if i < len(segments_struct):
                            segments_struct[i]["text"] = s2.text.strip()
                    progress.update(t_second, completed=100)
            else:
                # fallback path: internal translation
                t_fallback = progress.add_task("Whisper translation pass", total=None)
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
                mapped = list(seg2)
                for i, s2 in enumerate(mapped):
                    if i < len(segments_struct):
                        segments_struct[i]["text"] = s2.text.strip()
                progress.update(t_fallback, completed=100)
        else:
            # We already requested translate in first pass (args.task == translate or user forced no_mt)
            for s in segments_struct:
                s["text"] = s["src"]

        t_shape = progress.add_task("Shaping SRT", total=len(segments_struct))
        final_segments = []
        for seg in segments_struct:
            final_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })
            progress.advance(t_shape)

        if args.dry_run:
            console.print("[green]Dry run complete (no file written).[/green]")
            return

        t_write = progress.add_task("Writing SRT", total=100)
        write_srt(final_segments, out_path, args)
        progress.update(t_write, completed=100)

        if args.remux and not args.dry_run:
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
            else:
                console.print(f"[green]✓ Remux complete: {remux_target}[/green]")

    console.print(f"[bold green]Done.[/bold green] Wrote: {out_path}")

if __name__ == "__main__":
    main()
