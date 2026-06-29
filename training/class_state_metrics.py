"""Lightweight class/state metric aggregation utilities."""

import math


def class_state_bucket_key(class_id, state_bucket) -> str:
    return f"class_{int(class_id)}/{str(state_bucket)}"


def init_class_state_metric_accumulator() -> dict:
    return {}


def update_class_state_metric_accumulator(acc: dict, *, class_ids, state_buckets, metrics: dict) -> None:
    class_id_list = [int(v.item()) if hasattr(v, "item") else int(v) for v in class_ids]
    state_list = [str(v) for v in state_buckets]
    if len(class_id_list) != len(state_list):
        raise ValueError("class_ids and state_buckets length mismatch")
    for metric_name, values in metrics.items():
        if len(values) != len(class_id_list):
            raise ValueError(f"metric {metric_name} length mismatch")

    for idx, (class_id, state_bucket) in enumerate(zip(class_id_list, state_list)):
        key = class_state_bucket_key(class_id, state_bucket)
        bucket = acc.setdefault(key, {"count": 0, "sums": {}, "metric_counts": {}})
        bucket["count"] += 1
        for metric_name, values in metrics.items():
            value = values[idx]
            if hasattr(value, "item"):
                value = value.item()
            if value is None:
                continue
            value = float(value)
            if math.isnan(value):
                continue
            bucket["sums"][metric_name] = bucket["sums"].get(metric_name, 0.0) + value
            bucket["metric_counts"][metric_name] = bucket["metric_counts"].get(metric_name, 0) + 1


def finalize_class_state_metric_accumulator(acc: dict) -> dict:
    out = {}
    for key, bucket in sorted(acc.items()):
        count = int(bucket.get("count", 0))
        out[f"{key}/count"] = count
        metric_counts = bucket.get("metric_counts", {})
        for metric_name, total in sorted(bucket.get("sums", {}).items()):
            denom = max(int(metric_counts.get(metric_name, count)), 1)
            out[f"{key}/{metric_name}"] = float(total) / float(denom)
    return out


def extract_class_validation_losses(avg_val: dict, *, min_samples: int = 1) -> dict:
    class_totals = {}
    class_counts = {}
    prefix = "class_state/class_"
    for key, value in avg_val.items():
        if not str(key).startswith(prefix) or not str(key).endswith("/loss_real"):
            continue
        parts = str(key).split("/")
        if len(parts) != 4:
            continue
        class_id = int(parts[1].replace("class_", ""))
        state = parts[2]
        count_key = f"class_state/class_{class_id}/{state}/count"
        count = int(avg_val.get(count_key, 0))
        if count <= 0:
            continue
        class_totals[class_id] = class_totals.get(class_id, 0.0) + float(value) * count
        class_counts[class_id] = class_counts.get(class_id, 0) + count

    out = {}
    for class_id, total in class_totals.items():
        count = class_counts.get(class_id, 0)
        if count >= int(min_samples):
            out[class_id] = total / float(count)
    return out
