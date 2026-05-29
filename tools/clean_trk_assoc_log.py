#!/usr/bin/env python3
"""
Keep only tracker/association debug lines from a log.

Default keep tokens:
  - [TRK]
  - [ASSOC]

Usage:
  python tools/clean_trk_assoc_log.py -i raw.log -o filtered.log
  cat raw.log | python tools/clean_trk_assoc_log.py > filtered.log
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable, List


DEFAULT_KEEP = ["[TRK]", "[ASSOC]"]
USEFUL_TRK_KEYS = (
    "status_0=",
    "stage1 ",
    "stage2 ",
    "births=",
    "dead=",
)

# Typical tqdm suffix, e.g. " | 3/40 [00:01<00:14,  2.52it/s]"
TQDM_SUFFIX_RE = re.compile(r"\s+\|\s+\d+/\d+\s+\[.*$")


def normalize_line(line: str, keep_tokens: List[str]) -> str | None:
    pos = [line.find(token) for token in keep_tokens if token in line]
    if not pos:
        return None

    # Drop everything before the first keep token.
    line = line[min(pos):]
    # Remove tqdm progress tail.
    line = TQDM_SUFFIX_RE.sub("", line)
    # Compact repeated spaces and trim.
    line = re.sub(r"\s+", " ", line).strip()
    return line


def should_keep_line(line: str) -> bool:
    if line.startswith("[ASSOC]"):
        return True
    if line.startswith("[TRK]"):
        return any(key in line for key in USEFUL_TRK_KEYS)
    return False


def filter_lines(lines: Iterable[str], keep_tokens: List[str], minimal: bool) -> Iterable[str]:
    prev = None
    for line in lines:
        normalized = normalize_line(line, keep_tokens)
        if normalized is None:
            continue
        if minimal and not should_keep_line(normalized):
            continue
        # Drop immediate duplicates introduced by tqdm refresh.
        if normalized == prev:
            continue
        prev = normalized
        yield normalized + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter log lines by keep tokens.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="-",
        help="Input log file path, or '-' for stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output file path, or '-' for stdout.",
    )
    parser.add_argument(
        "--keep",
        nargs="+",
        default=DEFAULT_KEEP,
        help="Tokens to keep. Default: [TRK] [ASSOC].",
    )
    parser.add_argument(
        "--no-minimal",
        action="store_true",
        help="Disable minimal pruning and keep all normalized [TRK]/[ASSOC] lines.",
    )
    args = parser.parse_args()

    infile = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8", errors="replace")
    outfile = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")

    try:
        for line in filter_lines(infile, args.keep, minimal=not args.no_minimal):
            outfile.write(line)
    finally:
        if infile is not sys.stdin:
            infile.close()
        if outfile is not sys.stdout:
            outfile.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
