"""Lightweight class/state metric aggregation utilities."""


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
            bucket["sums"][metric_name] = bucket["sums"].get(metric_name, 0.0) + float(value)
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
