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
import sys
from typing import Iterable, List


DEFAULT_KEEP = ["[TRK]", "[ASSOC]"]


def filter_lines(lines: Iterable[str], keep_tokens: List[str]) -> Iterable[str]:
    for line in lines:
        if any(token in line for token in keep_tokens):
            yield line


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
    args = parser.parse_args()

    infile = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8", errors="replace")
    outfile = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")

    try:
        for line in filter_lines(infile, args.keep):
            outfile.write(line)
    finally:
        if infile is not sys.stdin:
            infile.close()
        if outfile is not sys.stdout:
            outfile.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
