#!/usr/bin/env python3
"""Clean a training log file, keeping only epoch summaries and checkpoint lines.

Usage:
    python tools/clean_train_log.py <input.log> [--output <output.log>]

If --output is omitted, prints to stdout.

Kept lines:
    - Epoch summary:  "HH:MM:SS [INFO] Epoch N/M (Xs) | loss=... | val=..."
    - Checkpoint:     "HH:MM:SS [INFO]   Saved checkpoint → ..."
    - Best model:     "HH:MM:SS [INFO]   New best model → ..."
"""

import argparse
import sys


def should_keep(line: str) -> bool:
    """Return True if the line should be kept in the cleaned log."""
    if "[INFO]" not in line:
        return False

    # Epoch summary: "Epoch N/M" followed by "(Xs)" and metrics
    if "Epoch" in line and "(raw=" in line:
        return True

    # Checkpoint save lines
    if "Saved checkpoint" in line:
        return True

    # New best model lines
    if "New best model" in line:
        return True

    return False


def clean_log(input_path: str) -> list[str]:
    """Read a log file and return only the lines worth keeping."""
    kept: list[str] = []
    with open(input_path, "r") as f:
        for line in f:
            if should_keep(line):
                kept.append(line.rstrip("\n"))
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Clean training log — keep only epoch summaries and checkpoint lines."
    )
    parser.add_argument("input", help="Path to the input log file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write the cleaned log (default: stdout)",
    )
    args = parser.parse_args()

    kept = clean_log(args.input)

    if args.output:
        with open(args.output, "w") as f:
            for line in kept:
                f.write(line + "\n")
        print(f"Cleaned log written to {args.output} ({len(kept)} lines kept)")
    else:
        for line in kept:
            print(line)


if __name__ == "__main__":
    main()
