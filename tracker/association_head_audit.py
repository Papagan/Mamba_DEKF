import json
from dataclasses import dataclass
from pathlib import Path


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

    def add(
        self,
        *,
        score: float,
        delta: float,
        cost_before: float,
        cost_after: float,
        active: bool,
        finite: bool,
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
        }


class AssociationHeadAuditAccumulator:
    def __init__(self):
        self._buckets = {}

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

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
