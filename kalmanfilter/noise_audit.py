import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Optional


FAMILIES = ("q_pos", "r_pos", "r_siz", "r_ori")


def _safe_ratio(value, prior):
    if prior is None:
        return None
    if not math.isfinite(value):
        return None
    if not math.isfinite(prior) or prior <= 0:
        return None
    ratio = value / prior
    if not math.isfinite(ratio):
        return None
    return ratio


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def _summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p90": _percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class _Bucket:
    split: str
    mode: str
    class_id: int
    class_name: str
    state: str
    history_len: Optional[int]
    sample_count: int = 0
    families: dict = field(default_factory=lambda: {name: [] for name in FAMILIES})
    ratios: dict = field(default_factory=lambda: {name: [] for name in FAMILIES})

    def add(self, families, prior_families):
        self.sample_count += 1
        for name in FAMILIES:
            value = float(families[name])
            if not math.isfinite(value):
                continue
            self.families[name].append(value)
            ratio = _safe_ratio(value, prior_families.get(name))
            if ratio is not None:
                self.ratios[name].append(ratio)


class NoiseAuditAccumulator:
    def __init__(self):
        self._buckets = {}

    def add_sample(
        self,
        *,
        split,
        mode,
        class_id,
        class_name,
        state,
        history_len,
        families,
        prior_families,
    ):
        key = (split, mode, class_id, class_name, state, history_len)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                split=split,
                mode=mode,
                class_id=class_id,
                class_name=class_name,
                state=state,
                history_len=history_len,
            )
            self._buckets[key] = bucket
        bucket.add(families, prior_families)

    def to_summary(self):
        buckets = []
        for bucket in self._buckets.values():
            buckets.append(
                {
                    "split": bucket.split,
                    "mode": bucket.mode,
                    "class_id": bucket.class_id,
                    "class_name": bucket.class_name,
                    "state": bucket.state,
                    "history_len": bucket.history_len,
                    "count": bucket.sample_count,
                    "families": {name: _summarize(bucket.families[name]) for name in FAMILIES},
                    "ratios": {name: _summarize(bucket.ratios[name]) for name in FAMILIES},
                }
            )
        buckets.sort(
            key=lambda item: (
                item["split"],
                item["mode"],
                item["class_id"],
                item["class_name"],
                item["state"],
                item["history_len"] is None,
                item["history_len"],
            )
        )
        return {
            "schema_version": 1,
            "families": list(FAMILIES),
            "buckets": buckets,
        }

    def export_state(self):
        buckets = []
        for bucket in self._buckets.values():
            buckets.append(
                {
                    "split": bucket.split,
                    "mode": bucket.mode,
                    "class_id": bucket.class_id,
                    "class_name": bucket.class_name,
                    "state": bucket.state,
                    "history_len": bucket.history_len,
                    "count": bucket.sample_count,
                    "families": {name: list(bucket.families[name]) for name in FAMILIES},
                    "ratios": {name: list(bucket.ratios[name]) for name in FAMILIES},
                }
            )
        buckets.sort(
            key=lambda item: (
                item["split"],
                item["mode"],
                item["class_id"],
                item["class_name"],
                item["state"],
                item["history_len"] is None,
                item["history_len"],
            )
        )
        return {"buckets": buckets}

    def merge_state(self, state):
        if not state:
            return
        for item in state.get("buckets", []):
            key = (
                item["split"],
                item["mode"],
                item["class_id"],
                item["class_name"],
                item["state"],
                item["history_len"],
            )
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    split=item["split"],
                    mode=item["mode"],
                    class_id=item["class_id"],
                    class_name=item["class_name"],
                    state=item["state"],
                    history_len=item["history_len"],
                )
                self._buckets[key] = bucket
            bucket.sample_count += int(item.get("count", 0))
            for name in FAMILIES:
                bucket.families[name].extend(float(value) for value in item["families"].get(name, []))
                bucket.ratios[name].extend(float(value) for value in item["ratios"].get(name, []))

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
