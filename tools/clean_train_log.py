#!/usr/bin/env python3
"""Clean a Mamba-DEKF training log, keeping only epoch summary lines.

Usage:
    python tools/clean_train_log.py checkpoints/mamba_dekf/train.log
    python tools/clean_train_log.py train.log -o clean.log
    python tools/clean_train_log.py train.log --in-place
"""

import argparse
import os
import re
import sys
from datetime import datetime


def clean_log(input_path: str, output_path: str | None = None) -> int:
    """Extract epoch summary lines from a training log.

    Returns the number of epoch lines found.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Extract info lines (dataset/model/mode summaries)
    info_lines = [L for L in lines if re.search(
        r"(?:Logging to|Device:|Train:|Val:|TemporalMamba:|^#)", L
    )]

    # Extract epoch summary lines (new format: "loss=", legacy: "train_loss=")
    epoch_pattern = re.compile(r"Epoch\s+\d+/\d+.*\bloss=")
    epoch_lines = [L for L in lines if epoch_pattern.search(L)]

    header = [
        f"# Mamba-DEKF Training Log (cleaned — {len(epoch_lines)} epoch summaries)\n",
        f"# Source: {os.path.abspath(input_path)}\n",
        f"# Cleaned: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "#\n",
    ]

    out_path = output_path or input_path
    with open(out_path, "w") as f:
        f.writelines(header)
        if info_lines:
            f.writelines(info_lines)
            f.write("\n")
        f.writelines(epoch_lines)

    return len(epoch_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Clean Mamba-DEKF training log — keep only epoch summaries"
    )
    parser.add_argument("input", help="Path to training log file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: overwrite input)"
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite input file in place"
    )
    args = parser.parse_args()

    output = args.output
    if args.in_place:
        output = args.input

    n = clean_log(args.input, output)
    out = output or args.input
    original_lines = sum(1 for _ in open(args.input))
    print(f"Input:  {original_lines} lines  →  {args.input}")
    print(f"Output: {n} epoch summaries + header  →  {out}")


if __name__ == "__main__":
    main()
