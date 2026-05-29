#!/usr/bin/env python3
"""Clean a result.log file by trimming trailing whitespace, removing
extra blank lines, and dropping repeated metric table headers.

Usage:
    python tools/clean_result_log.py result.log
    python tools/clean_result_log.py result.log -o result_clean.log
    python tools/clean_result_log.py result.log --in-place
"""

from __future__ import annotations

import argparse
from pathlib import Path


def clean_result_log(input_path: str, output_path: str | None = None) -> int:
    """Clean a result log while preserving its readable table layout.

    Internal spacing is left untouched so the column layout in evaluation
    tables stays readable.

    Returns the number of non-empty lines written.
    """
    table_header = (
        "MOTAR   MOTP    Recall  Frames  GT      GT-Mtch GT-Miss GT-IDS  "
        "Pred    Pred-TP Pred-FP Pred-IDS"
    )
    source_path = Path(input_path)
    text = source_path.read_bytes().decode("utf-8")
    lines = text.splitlines(keepends=True)

    cleaned_lines: list[str] = []
    non_empty_count = 0
    in_class_block = False
    saw_table_header = False
    previous_was_blank = False
    for line in lines:
        if line.endswith("\r\n"):
            line_body = line[:-2]
            line_ending = "\r\n"
        elif line.endswith(("\n", "\r")):
            line_body = line[:-1]
            line_ending = line[-1]
        else:
            line_body = line
            line_ending = ""

        normalized = line_body.rstrip()
        stripped = normalized.strip()

        if not stripped:
            if in_class_block:
                continue
            if cleaned_lines and not previous_was_blank:
                cleaned_lines.append(line_ending)
                previous_was_blank = True
            continue

        if stripped.startswith("Computing metrics for class "):
            in_class_block = True
            saw_table_header = False
        elif stripped.startswith("Calculating metrics") or stripped.startswith("Saving metrics to:"):
            in_class_block = False
            saw_table_header = False

        if in_class_block and stripped == table_header:
            if saw_table_header:
                continue
            saw_table_header = True

        cleaned_lines.append(normalized + line_ending)
        non_empty_count += 1
        previous_was_blank = False

    target_path = Path(output_path) if output_path else source_path
    target_path.write_bytes("".join(cleaned_lines).encode("utf-8"))
    return non_empty_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean result.log by trimming trailing whitespace"
    )
    parser.add_argument("input", help="Path to result.log")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite input file in place",
    )
    args = parser.parse_args()

    output = args.input if args.in_place else args.output
    line_count = clean_result_log(args.input, output)
    target = output or args.input
    print(f"Input:  {args.input}")
    print(f"Output: {target}")
    print(f"Cleaned non-empty lines: {line_count}")


if __name__ == "__main__":
    main()