#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from kalmanfilter.noise_audit import FAMILIES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize one or more noise audit JSON files with compact per-family mode comparisons."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more noise audit JSON files.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument("--output", default="", help="Optional output path. Prints to stdout when omitted.")
    return parser.parse_args(argv)


def load_audits(paths):
    payloads = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"{path}: unsupported schema_version={schema_version!r}")
        payloads.append(payload)
    return payloads


def _collect_families(payloads):
    ordered = list(FAMILIES)
    seen = set(ordered)
    for payload in payloads:
        for family in payload.get("families", []):
            if family not in seen:
                ordered.append(family)
                seen.add(family)
    return ordered


def _extract_metrics(bucket, family):
    family_stats = bucket.get("families", {}).get(family, {})
    ratio_stats = bucket.get("ratios", {}).get(family, {})
    metrics = {
        "count": bucket.get("count"),
        "family_median": family_stats.get("median"),
        "family_p90": family_stats.get("p90"),
        "ratio_median": ratio_stats.get("median"),
        "ratio_p90": ratio_stats.get("p90"),
    }
    if all(metrics[name] is None for name in ("family_median", "family_p90", "ratio_median", "ratio_p90")):
        return None
    return metrics


def build_summary(payloads):
    families = _collect_families(payloads)
    rows = {}
    for payload in payloads:
        for bucket in payload.get("buckets", []):
            for family in families:
                metrics = _extract_metrics(bucket, family)
                if metrics is None:
                    continue
                key = (
                    bucket.get("class_name"),
                    bucket.get("state"),
                    bucket.get("history_len"),
                    family,
                )
                split = bucket.get("split")
                row = rows.setdefault(
                    key,
                    {
                        "split": split,
                        "splits": [] if split is None else [split],
                        "class_name": bucket.get("class_name"),
                        "state": bucket.get("state"),
                        "history_len": bucket.get("history_len"),
                        "family": family,
                        "modes": {},
                    },
                )
                if split is not None:
                    if row["splits"] and split not in row["splits"]:
                        raise ValueError(
                            "mixed split comparison entry for "
                            f"class_name={row['class_name']!r}, state={row['state']!r}, "
                            f"history_len={row['history_len']!r}, family={family!r}: "
                            f"{sorted(row['splits'])!r} vs {split!r}"
                        )
                    if split not in row["splits"]:
                        row["splits"].append(split)
                mode = bucket.get("mode")
                if mode in row["modes"]:
                    raise ValueError(
                        "duplicate bucket comparison entry for "
                        f"splits={sorted(row['splits'])!r}, class_name={row['class_name']!r}, "
                        f"state={row['state']!r}, history_len={row['history_len']!r}, family={family!r}, mode={mode!r}"
                    )
                row["modes"][mode] = metrics

    summary_rows = []
    for key in sorted(rows, key=lambda item: (item[0], item[1], item[2] is None, item[2], item[3])):
        row = rows[key]
        row["splits"] = sorted(row["splits"])
        if len(row["splits"]) == 1:
            row["split"] = row["splits"][0]
        elif row["splits"]:
            row["split"] = ",".join(row["splits"])
        else:
            row["split"] = None
        row["modes"] = {mode: row["modes"][mode] for mode in sorted(row["modes"])}
        summary_rows.append(row)
    return {
        "schema_version": 1,
        "families": families,
        "rows": summary_rows,
    }


def _format_number(value):
    if value is None:
        return "-"
    return f"{value:g}"


def render_text(summary):
    lines = []
    for row in summary.get("rows", []):
        splits = row.get("splits")
        if splits:
            split_label = ",".join(splits)
        else:
            split_label = row.get("split") or "-"
        parts = [
            split_label,
            row.get("class_name") or "-",
            row.get("state") or "-",
        ]
        history_len = row.get("history_len")
        if history_len is not None:
            parts.append(f"h={history_len}")
        parts.append(row.get("family") or "-")
        line = " ".join(parts)
        mode_chunks = []
        for mode, metrics in row.get("modes", {}).items():
            mode_chunks.append(
                f"{mode} family={_format_number(metrics.get('family_median'))}/{_format_number(metrics.get('family_p90'))} "
                f"ratio={_format_number(metrics.get('ratio_median'))}/{_format_number(metrics.get('ratio_p90'))}"
            )
        lines.append(f"{line} | " + " | ".join(mode_chunks))
    if not lines:
        return "No comparable buckets found."
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    summary = build_summary(load_audits(args.inputs))
    if args.format == "json":
        output = json.dumps(summary, indent=2, sort_keys=True)
    else:
        output = render_text(summary)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
