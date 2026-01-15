#!/usr/bin/env python3
"""
Extract only participant (*PAR:) speech from CHAT (.cha) files.

Usage:
  python extract_par.py /path/to/file.cha
  python extract_par.py /path/to/folder_with_cha_files
  python extract_par.py /path/to/folder --out out_dir

Output:
  For each input .cha file, writes a .par.txt file containing only participant lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_par_lines(text: str) -> list[str]:
    """
    Keep only lines that start with '*PAR:'.
    Strips the '*PAR:' prefix and leading/trailing whitespace.
    """
    out: list[str] = []
    for line in text.splitlines():
        # CHAT lines sometimes have leading spaces; be robust
        s = line.lstrip()
        if s.startswith("*PAR:"):
            # Remove prefix and normalize whitespace around content
            content = s[len("*PAR:"):].strip()
            if content:
                out.append(content)
    return out


def iter_cha_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".cha":
            raise ValueError(f"Expected a .cha file, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.cha"))
        if not files:
            raise ValueError(f"No .cha files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract participant (*PAR:) speech from CHAT files.")
    parser.add_argument("input_path", type=str, help="Path to a .cha file or a directory containing .cha files")
    parser.add_argument("--out", type=str, default="", help="Output directory (default: alongside input file(s))")
    args = parser.parse_args()

    in_path = Path(args.input_path).expanduser().resolve()
    out_dir = Path(args.out or "Dataset with PAR only").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cha_files = iter_cha_files(in_path)

    for cha in cha_files:
        raw = cha.read_text(encoding="utf-8", errors="replace")
        par_lines = extract_par_lines(raw)

        # Decide output location
        out_path = out_dir / f"{cha.stem}.par.txt"

        out_path.write_text("\n".join(par_lines) + ("\n" if par_lines else ""), encoding="utf-8")

        print(f"Wrote {out_path}  ({len(par_lines)} *PAR lines)")

    print("Done.")


if __name__ == "__main__":
    main()
