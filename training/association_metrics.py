from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def _binary_auc(labels: List[int], scores: List[float]) -> float:
    positives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if label == 1]
    negatives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return float(wins / total)


def _topk(records: List[Dict[str, Any]], k: int) -> float:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["anchor_key"])].append(record)
    valid = 0
    hits = 0
    for items in grouped.values():
        if not any(int(item["label"]) == 1 for item in items):
            continue
        valid += 1
        ranked = sorted(items, key=lambda item: float(item["score"]), reverse=True)
        if any(int(item["label"]) == 1 for item in ranked[: int(k)]):
            hits += 1
    return float(hits / valid) if valid else 0.0


def _hard_negative_accuracy(records: List[Dict[str, Any]]) -> float:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["anchor_key"])].append(record)
    valid = 0
    correct = 0
    for items in grouped.values():
        positives = [item for item in items if int(item["label"]) == 1]
        hard_negs = [item for item in items if int(item["label"]) == 0 and item.get("negative_type") == "hard"]
        if not positives or not hard_negs:
            continue
        valid += len(hard_negs)
        pos_score = max(float(item["score"]) for item in positives)
        correct += sum(1 for item in hard_negs if pos_score > float(item["score"]))
    return float(correct / valid) if valid else 0.0


def _summarize(records: List[Dict[str, Any]], topk: Sequence[int]) -> Dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    scores = [float(record["score"]) for record in records]
    out = {
        "pairs": len(records),
        "positive_pairs": sum(1 for label in labels if label == 1),
        "negative_pairs": sum(1 for label in labels if label == 0),
        "auc": _binary_auc(labels, scores),
        "hard_negative_accuracy": _hard_negative_accuracy(records),
    }
    for k in topk:
        out[f"top{k}"] = _topk(records, int(k))
    return out


def compute_association_metrics(
    records: Iterable[Dict[str, Any]],
    *,
    topk: Sequence[int] = (1, 3),
) -> Dict[str, Any]:
    records = list(records)
    per_class_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        per_class_records[str(int(record["class_id"]))].append(record)
    return {
        "overall": _summarize(records, topk),
        "per_class": {
            class_id: _summarize(items, topk)
            for class_id, items in sorted(per_class_records.items(), key=lambda kv: int(kv[0]))
        },
    }
