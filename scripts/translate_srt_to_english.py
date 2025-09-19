"""
Translate an existing SRT file's text into English, preserving timings.

Usage (PowerShell):
  .\.venv\Scripts\python.exe scripts\translate_srt_to_english.py "C:\path\to\subs.srt" -o "C:\path\to\subs.en.srt" --device cuda

Notes:
  - Uses facebook/nllb-200-distilled-600M by default
  - Forces English output (eng_Latn) when tokenizer supports language codes
  - Runs on CUDA if available, else CPU
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def parse_srt(text: str) -> List[Tuple[int, str, str, str]]:
    """Parse SRT text into a list of (index, start, end, payload_text)."""
    blocks: List[Tuple[int, str, str, str]] = []
    entries = re.split(r"\r?\n\s*\r?\n", text.strip())
    ts_pat = re.compile(r"\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")
    for entry in entries:
        lines = entry.splitlines()
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            # Not starting with numeric index; try to continue anyway
            idx = len(blocks) + 1
            ts_line = lines[0]
            text_lines = lines[1:]
        else:
            if len(lines) < 3:
                continue
            ts_line = lines[1]
            text_lines = lines[2:]

        m = ts_pat.match(ts_line)
        if not m:
            # Skip malformed block
            continue
        start, end = m.group(1), m.group(2)
        payload = "\n".join(text_lines).strip()
        blocks.append((idx, start.replace(".", ","), end.replace(".", ","), payload))
    return blocks


def write_srt(blocks: List[Tuple[int, str, str, str]]) -> str:
    out_lines: List[str] = []
    for i, (idx, start, end, text) in enumerate(blocks, start=1):
        out_lines.append(str(i))
        out_lines.append(f"{start} --> {end}")
        out_lines.append(text)
        out_lines.append("")
    return "\n".join(out_lines) + "\n"


def load_mt(model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(model_name)
    mt = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                mt = mt.to("cuda")
            else:
                console.print("[yellow]CUDA requested but not available; using CPU[/yellow]")
        except Exception:
            console.print("[yellow]Torch not available; using CPU[/yellow]")
    return tok, mt


def batch_translate_text(
    texts: List[str], tok, mt, batch_size: int = 8, max_new_tokens: int = 200
) -> List[str]:
    from torch import no_grad
    import torch

    device = next(mt.parameters()).device
    out: List[str] = []
    # Determine forced BOS for English when available
    forced_bos_token_id = None
    if hasattr(tok, "lang_code_to_id"):
        forced_bos_token_id = tok.lang_code_to_id.get("eng_Latn", None)
    elif hasattr(tok, "get_lang_id"):
        try:
            forced_bos_token_id = tok.get_lang_id("en")
        except Exception:
            forced_bos_token_id = None

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        with no_grad():
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
            gen = mt.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                forced_bos_token_id=forced_bos_token_id,
            )
            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            out.extend([d.strip() for d in decoded])
    return out


def main():
    p = argparse.ArgumentParser(description="Translate an SRT file to English using NLLB.")
    p.add_argument("input", help="Path to input .srt")
    p.add_argument("-o", "--output", help="Path to output .srt (default: <input>.en.srt)")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="HuggingFace MT model")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to run MT")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        console.print(f"[red]Input not found: {in_path}[/red]")
        raise SystemExit(1)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".en.srt")

    # Read and parse
    text = in_path.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_srt(text)
    if not blocks:
        console.print("[red]No valid SRT entries found.[/red]")
        raise SystemExit(1)

    tok, mt = load_mt(args.model, args.device)

    # Translate block texts
    payloads = [b[3] for b in blocks]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    translated: List[str] = []
    with progress:
        task = progress.add_task("Translating blocks", total=len(blocks))
        # Process in chunks to preserve timing blocks
        for i in range(0, len(payloads), args.batch_size):
            chunk = payloads[i:i+args.batch_size]
            outs = batch_translate_text(chunk, tok, mt, batch_size=len(chunk), max_new_tokens=args.max_new_tokens)
            translated.extend(outs)
            progress.update(task, advance=len(chunk))

    new_blocks: List[Tuple[int, str, str, str]] = []
    for (idx, start, end, _), new_text in zip(blocks, translated):
        # Keep line breaks modest: wrap long lines lightly by punctuation
        new_blocks.append((idx, start, end, new_text))

    out_text = write_srt(new_blocks)
    out_path.write_text(out_text, encoding="utf-8")
    console.print(f"[green]✓ Wrote English SRT:[/green] {out_path}")


if __name__ == "__main__":
    main()
