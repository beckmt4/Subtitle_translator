r"""
Remove common credit/fansub lines from an SRT while preserving timings.

Heuristics:
 - Drop lines that match typical credit keywords (e.g., "Subtitled by", "Typeset", "Karaoke", "Encoder")
 - Drop all-uppercase non-dialog blocks early in the file
 - Keep ordinary dialog; maintain numbering and timestamps

Usage (PowerShell):
    .\.venv\Scripts\python.exe scripts\clean_srt_credits.py "C:/path/to/in.srt" -o "C:/path/to/out.clean.srt"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

console = Console()


def parse_srt(text: str) -> List[Tuple[int, str, str, str]]:
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
            # If no numeric index, skip
            continue
        if len(lines) < 3:
            continue
        ts_line = lines[1]
        m = ts_pat.match(ts_line)
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        payload = "\n".join(lines[2:]).strip()
        blocks.append((idx, start.replace(".", ","), end.replace(".", ","), payload))
    return blocks


def write_srt(blocks: List[Tuple[int, str, str, str]]) -> str:
    out: List[str] = []
    for i, (idx, start, end, text) in enumerate(blocks, start=1):
        out.append(str(i))
        out.append(f"{start} --> {end}")
        out.append(text)
        out.append("")
    return "\n".join(out) + "\n"


KEYWORDS = [
    r"subtitl(?:ed|e)\s+by",
    r"typeset",
    r"karaoke",
    r"encode(?:r|d)",
    r"tim(?:e|ing)\s*:\s*",
    r"credit(s)?",
    r"op\s*\/\s*ed",
    r"opening|ending\s+song",
    r"qc\s*:\s*",
    r"sponsor",
]
KEYWORD_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)


def looks_like_credit(text: str, idx: int) -> bool:
    if KEYWORD_RE.search(text):
        return True
    # early non-dialog, all caps or many non-letter chars
    if idx < 15:
        lines = text.splitlines()
        joined = " ".join(lines).strip()
        if joined and (joined.upper() == joined):
            return True
        # very few letters => likely logo or FX
        letters = sum(c.isalpha() for c in joined)
        if letters < max(10, len(joined)//4):
            return True
    return False


def main():
    p = argparse.ArgumentParser(description="Remove fansub/credit lines from SRT")
    p.add_argument("input", help="Input .srt path")
    p.add_argument("-o", "--output", help="Output .srt path (default: <input>.clean.srt)")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        console.print(f"[red]Input not found: {src}[/red]")
        raise SystemExit(1)
    dst = Path(args.output) if args.output else src.with_suffix(".clean.srt")

    text = src.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_srt(text)
    if not blocks:
        console.print("[red]No valid entries parsed.[/red]")
        raise SystemExit(1)

    kept: List[Tuple[int, str, str, str]] = []
    dropped = 0
    for i, (idx, start, end, payload) in enumerate(blocks, start=1):
        if looks_like_credit(payload, i):
            dropped += 1
            continue
        kept.append((idx, start, end, payload))

    out_text = write_srt(kept)
    dst.write_text(out_text, encoding="utf-8")
    console.print(f"[green]✓ Cleaned SRT written:[/green] {dst} (dropped {dropped} blocks)")


if __name__ == "__main__":
    main()
