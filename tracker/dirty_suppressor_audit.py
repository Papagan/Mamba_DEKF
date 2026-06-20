import json
from dataclasses import dataclass, field
from pathlib import Path


_FEATURE_KEYS = (
    "recent_fake_len",
    "low_score_ratio",
    "pos_trace_ratio",
    "recent_match_cost_mean",
)


@dataclass
class _Bucket:
    class_id: int
    class_name: str
    profile_name: str | None
    evaluated_count: int = 0
    emitted_count: int = 0
    soft_hit_count: int = 0
    hard_reject_count: int = 0
    penalty_sum: float = 0.0
    hit_count: int = 0
    hit_feature_sums: dict = field(default_factory=lambda: {key: 0.0 for key in _FEATURE_KEYS})

    def add(self, *, penalty: float, hard_reject: bool, features: dict):
        self.evaluated_count += 1
        self.penalty_sum += float(penalty)
        if hard_reject:
            self.hard_reject_count += 1
        else:
            self.emitted_count += 1

        soft_hit = (float(penalty) < 0.999999) and not hard_reject
        hit = soft_hit or hard_reject
        if soft_hit:
            self.soft_hit_count += 1
        if hit:
            self.hit_count += 1
            for key in _FEATURE_KEYS:
                self.hit_feature_sums[key] += float((features or {}).get(key, 0.0))

    def to_summary(self):
        avg_penalty = (
            self.penalty_sum / self.evaluated_count
            if self.evaluated_count > 0
            else None
        )
        avg_hit_features = {
            key: (
                self.hit_feature_sums[key] / self.hit_count
                if self.hit_count > 0
                else None
            )
            for key in _FEATURE_KEYS
        }
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "profile_name": self.profile_name,
            "evaluated_count": self.evaluated_count,
            "emitted_count": self.emitted_count,
            "soft_hit_count": self.soft_hit_count,
            "hard_reject_count": self.hard_reject_count,
            "avg_penalty": avg_penalty,
            "avg_hit_features": avg_hit_features,
        }


class DirtySuppressorAuditAccumulator:
    def __init__(self):
        self._buckets = {}

    def add_sample(
        self,
        *,
        class_id: int,
        class_name: str,
        profile_name: str | None,
        penalty: float,
        hard_reject: bool,
        features: dict,
    ):
        key = (int(class_id), str(class_name), profile_name)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                class_id=int(class_id),
                class_name=str(class_name),
                profile_name=profile_name,
            )
            self._buckets[key] = bucket
        bucket.add(
            penalty=float(penalty),
            hard_reject=bool(hard_reject),
            features=features or {},
        )

    def to_summary(self):
        buckets = [bucket.to_summary() for bucket in self._buckets.values()]
        buckets.sort(key=lambda item: (item["class_id"], item["class_name"]))
        return {
            "schema_version": 1,
            "feature_keys": list(_FEATURE_KEYS),
            "buckets": buckets,
        }

    def export_state(self):
        return self.to_summary()

    def merge_state(self, state):
        if not state:
            return
        for item in state.get("buckets", []):
            key = (int(item["class_id"]), str(item["class_name"]), item.get("profile_name"))
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    profile_name=item.get("profile_name"),
                )
                self._buckets[key] = bucket
            bucket.evaluated_count += int(item.get("evaluated_count", 0))
            bucket.emitted_count += int(item.get("emitted_count", 0))
            bucket.soft_hit_count += int(item.get("soft_hit_count", 0))
            bucket.hard_reject_count += int(item.get("hard_reject_count", 0))
            bucket.penalty_sum += float(item.get("avg_penalty") or 0.0) * int(item.get("evaluated_count", 0))
            hit_count = int(item.get("soft_hit_count", 0)) + int(item.get("hard_reject_count", 0))
            bucket.hit_count += hit_count
            avg_hit_features = item.get("avg_hit_features", {}) or {}
            for feature_key in _FEATURE_KEYS:
                avg_value = avg_hit_features.get(feature_key, None)
                if avg_value is not None:
                    bucket.hit_feature_sums[feature_key] += float(avg_value) * hit_count

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
