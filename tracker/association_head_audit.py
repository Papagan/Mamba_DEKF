import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class _Bucket:
    class_id: int
    class_name: str
    state_bucket: str
    pair_count: int = 0
    finite_pair_count: int = 0
    active_pair_count: int = 0
    penalized_pair_count: int = 0
    score_sum: float = 0.0
    delta_sum: float = 0.0
    cost_before_sum: float = 0.0
    cost_after_sum: float = 0.0
    scores: list = field(default_factory=list)
    deltas: list = field(default_factory=list)
    cost_befores: list = field(default_factory=list)
    cost_afters: list = field(default_factory=list)
    samples: list = field(default_factory=list)

    def add(
        self,
        *,
        score: float,
        delta: float,
        cost_before: float,
        cost_after: float,
        active: bool,
        finite: bool,
        sample=None,
        max_samples: int = 0,
    ) -> None:
        self.pair_count += 1
        if finite:
            self.finite_pair_count += 1
        if active:
            self.active_pair_count += 1
        if delta > 0.0:
            self.penalized_pair_count += 1
        self.score_sum += float(score)
        self.delta_sum += float(delta)
        self.cost_before_sum += float(cost_before)
        self.cost_after_sum += float(cost_after)
        self.scores.append(float(score))
        self.deltas.append(float(delta))
        self.cost_befores.append(float(cost_before))
        self.cost_afters.append(float(cost_after))
        if sample is not None and len(self.samples) < int(max_samples):
            self.samples.append(dict(sample))

    @staticmethod
    def _quantiles(values):
        if not values:
            return {"min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(np.max(arr)),
        }

    def to_summary(self):
        pair_count = max(int(self.pair_count), 1)
        active_count = max(int(self.active_pair_count), 1)
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "state_bucket": str(self.state_bucket),
            "pair_count": int(self.pair_count),
            "finite_pair_count": int(self.finite_pair_count),
            "active_pair_count": int(self.active_pair_count),
            "penalized_pair_count": int(self.penalized_pair_count),
            "active_ratio": float(self.active_pair_count / pair_count),
            "penalized_ratio": float(self.penalized_pair_count / active_count),
            "avg_score": float(self.score_sum / pair_count),
            "avg_delta": float(self.delta_sum / pair_count),
            "avg_cost_before": float(self.cost_before_sum / pair_count),
            "avg_cost_after": float(self.cost_after_sum / pair_count),
            "score_quantiles": self._quantiles(self.scores),
            "delta_quantiles": self._quantiles(self.deltas),
            "cost_before_quantiles": self._quantiles(self.cost_befores),
            "cost_after_quantiles": self._quantiles(self.cost_afters),
            "sample_count": int(len(self.samples)),
            "samples": list(self.samples),
        }


class AssociationHeadAuditAccumulator:
    def __init__(self, *, max_samples_per_bucket: int = 0):
        self._buckets = {}
        self.max_samples_per_bucket = int(max_samples_per_bucket)

    def add_pair(
        self,
        *,
        class_id: int,
        class_name: str,
        state_bucket: str,
        score: float,
        delta: float,
        cost_before: float,
        cost_after: float,
        active: bool,
        finite: bool,
        sample=None,
    ) -> None:
        key = (int(class_id), str(class_name), str(state_bucket))
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                class_id=int(class_id),
                class_name=str(class_name),
                state_bucket=str(state_bucket),
            )
            self._buckets[key] = bucket
        bucket.add(
            score=float(score),
            delta=float(delta),
            cost_before=float(cost_before),
            cost_after=float(cost_after),
            active=bool(active),
            finite=bool(finite),
            sample=sample,
            max_samples=self.max_samples_per_bucket,
        )

    def to_summary(self):
        buckets = [bucket.to_summary() for bucket in self._buckets.values()]
        buckets.sort(key=lambda item: (item["class_id"], item["state_bucket"], item["class_name"]))
        return {
            "schema_version": 1,
            "buckets": buckets,
        }

    def export_state(self):
        return self.to_summary()

    def merge_state(self, state):
        if not state:
            return
        for item in state.get("buckets", []):
            key = (
                int(item["class_id"]),
                str(item["class_name"]),
                str(item["state_bucket"]),
            )
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    state_bucket=str(item["state_bucket"]),
                )
                self._buckets[key] = bucket

            pair_count = int(item.get("pair_count", 0))
            bucket.pair_count += pair_count
            bucket.finite_pair_count += int(item.get("finite_pair_count", 0))
            bucket.active_pair_count += int(item.get("active_pair_count", 0))
            bucket.penalized_pair_count += int(item.get("penalized_pair_count", 0))
            bucket.score_sum += float(item.get("avg_score", 0.0)) * pair_count
            bucket.delta_sum += float(item.get("avg_delta", 0.0)) * pair_count
            bucket.cost_before_sum += float(item.get("avg_cost_before", 0.0)) * pair_count
            bucket.cost_after_sum += float(item.get("avg_cost_after", 0.0)) * pair_count
            # Scene-level states are already summarized; keep bounded proxy
            # points for merged quantiles instead of expanding millions of
            # pairs back into memory.
            for attr, field_name, avg_name in (
                ("scores", "score_quantiles", "avg_score"),
                ("deltas", "delta_quantiles", "avg_delta"),
                ("cost_befores", "cost_before_quantiles", "avg_cost_before"),
                ("cost_afters", "cost_after_quantiles", "avg_cost_after"),
            ):
                quantiles = item.get(field_name, {}) or {}
                values = [
                    quantiles.get("min", item.get(avg_name, 0.0)),
                    quantiles.get("p10", item.get(avg_name, 0.0)),
                    quantiles.get("p50", item.get(avg_name, 0.0)),
                    quantiles.get("p90", item.get(avg_name, 0.0)),
                    quantiles.get("max", item.get(avg_name, 0.0)),
                ]
                getattr(bucket, attr).extend(float(value) for value in values)
            for sample in item.get("samples", []):
                if len(bucket.samples) >= self.max_samples_per_bucket:
                    break
                bucket.samples.append(dict(sample))

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
