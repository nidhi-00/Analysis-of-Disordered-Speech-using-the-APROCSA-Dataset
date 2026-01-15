#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FILLED_PAUSES = {"uh", "um", "er", "ah", "hmm", "mm", "uhm", "umm"}

# Pause markers in APROCSA transcripts often appear like (.) (..) (...) etc.
PAUSE_MARKER_RE = re.compile(r"\(\.{1,}\)")  # matches (.) (..) (...) (....)
PAREN_ANNOT_RE = re.compile(r"\([^)]*\)")    # remove any (...) annotation from token stream


@dataclass
class SpeakerStats:
    file: str
    speaker: str  # "PAR" or "INV"
    utterances: int = 0
    tokens: int = 0
    types: int = 0
    filled_pauses: int = 0
    pause_markers: int = 0

    def mean_utt_len(self) -> float:
        return (self.tokens / self.utterances) if self.utterances else 0.0

    def ttr(self) -> float:
        return (self.types / self.tokens) if self.tokens else 0.0

    def filled_pauses_per_100(self) -> float:
        return (self.filled_pauses / self.tokens * 100.0) if self.tokens else 0.0

    def pause_markers_per_100(self) -> float:
        return (self.pause_markers / self.tokens * 100.0) if self.tokens else 0.0


def iter_cha_files(root: Path, recursive: bool = True) -> list[Path]:
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.rglob("*.cha") if recursive else root.glob("*.cha"))
    raise FileNotFoundError(f"Path not found: {root}")


def extract_main_tier_lines(text: str, speaker_code: str) -> list[str]:
    """
    Return list of raw main-tier lines for a given speaker code ("PAR", "INV").
    Each returned item is the content after "*XXX:".
    """
    prefix = f"*{speaker_code}:"
    out = []
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith(prefix):
            out.append(s[len(prefix):].strip())
    return out


def count_pause_markers(raw_line: str) -> int:
    return len(PAUSE_MARKER_RE.findall(raw_line))


def clean_and_tokenize(raw_line: str) -> list[str]:
    """
    Cleaning choices:
    - remove any (...) annotations from the token stream (including pauses)
    - lowercase
    - keep apostrophes within words
    - strip punctuation around words
    """
    # Remove all parenthetical annotations from the token stream
    no_paren = PAREN_ANNOT_RE.sub(" ", raw_line)

    # Lowercase
    s = no_paren.lower()

    # Replace non-word-ish chars with space, but keep apostrophes
    # Keep letters/numbers/apostrophes
    s = re.sub(r"[^a-z0-9'\s]+", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return []

    return s.split(" ")


def compute_stats_for_speaker(file_path: Path, speaker_code: str) -> SpeakerStats:
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    lines = extract_main_tier_lines(raw, speaker_code)

    stats = SpeakerStats(file=file_path.name, speaker=speaker_code)

    vocab = set()

    for line in lines:
        stats.utterances += 1

        # Count explicit pause markers before removing parentheses
        stats.pause_markers += count_pause_markers(line)

        toks = clean_and_tokenize(line)
        if not toks:
            continue

        stats.tokens += len(toks)

        for t in toks:
            vocab.add(t)
            if t in FILLED_PAUSES:
                stats.filled_pauses += 1

    stats.types = len(vocab)
    return stats


def write_csv(rows: Iterable[SpeakerStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "file",
        "speaker",
        "utterances",
        "tokens",
        "mean_utterance_length",
        "ttr",
        "filled_pauses_per_100_tokens",
        "pause_markers_per_100_tokens",
        "filled_pauses_count",
        "pause_markers_count",
        "types",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.file,
                r.speaker,
                r.utterances,
                r.tokens,
                f"{r.mean_utt_len():.4f}",
                f"{r.ttr():.4f}",
                f"{r.filled_pauses_per_100():.4f}",
                f"{r.pause_markers_per_100():.4f}",
                r.filled_pauses,
                r.pause_markers,
                r.types,
            ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path", help="Path to folder containing .cha files (or a single .cha file)")
    ap.add_argument(
        "--out",
        default="Outputs/Statistics/a3_table.csv",
        help="Where to write the CSV table"
    )
    ap.add_argument("--no-recursive", action="store_true", help="Do not search subfolders")
    args = ap.parse_args()

    root = Path(args.dataset_path).expanduser().resolve()
    cha_files = iter_cha_files(root, recursive=(not args.no_recursive))

    all_rows: list[SpeakerStats] = []
    for fp in cha_files:
        all_rows.append(compute_stats_for_speaker(fp, "PAR"))
        all_rows.append(compute_stats_for_speaker(fp, "INV"))

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    out_path = (PROJECT_ROOT / args.out).resolve()
    write_csv(all_rows, out_path)

    print(f"Wrote {out_path}")
    print(f"Rows: {len(all_rows)} (2 per transcript: PAR + INV)")


if __name__ == "__main__":
    main()
