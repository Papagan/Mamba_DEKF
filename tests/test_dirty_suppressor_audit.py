import json
import tempfile
import unittest
from pathlib import Path

from tracker.dirty_suppressor_audit import DirtySuppressorAuditAccumulator


class DirtySuppressorAuditAccumulatorTest(unittest.TestCase):
    def test_accumulates_and_summarizes_by_class(self):
        acc = DirtySuppressorAuditAccumulator()
        acc.add_sample(
            class_id=5,
            class_name="trailer",
            profile_name="trailer_only",
            penalty=0.8,
            hard_reject=False,
            triggered_reasons=["recent_fake_len", "pos_trace_ratio"],
            features={
                "recent_fake_len": 1,
                "low_score_ratio": 0.0,
                "pos_trace_ratio": 2.5,
                "recent_match_cost_mean": 0.2,
            },
        )
        acc.add_sample(
            class_id=5,
            class_name="trailer",
            profile_name="trailer_only",
            penalty=1.0,
            hard_reject=False,
            triggered_reasons=[],
            features={
                "recent_fake_len": 0,
                "low_score_ratio": 0.0,
                "pos_trace_ratio": 1.0,
                "recent_match_cost_mean": 0.1,
            },
        )
        summary = acc.to_summary()
        bucket = summary["buckets"][0]
        self.assertEqual(bucket["class_name"], "trailer")
        self.assertEqual(bucket["profile_name"], "trailer_only")
        self.assertEqual(bucket["evaluated_count"], 2)
        self.assertEqual(bucket["soft_hit_count"], 1)
        self.assertEqual(bucket["hard_reject_count"], 0)
        self.assertAlmostEqual(bucket["avg_penalty"], 0.9)
        self.assertAlmostEqual(bucket["avg_hit_features"]["pos_trace_ratio"], 2.5)
        self.assertEqual(
            bucket["reason_counts"],
            {"recent_fake_len": 1, "pos_trace_ratio": 1},
        )

    def test_merge_and_write_json(self):
        first = DirtySuppressorAuditAccumulator()
        first.add_sample(
            class_id=2,
            class_name="bicycle",
            profile_name="agile_weak",
            penalty=0.8,
            hard_reject=False,
            triggered_reasons=["recent_fake_len"],
            features={
                "recent_fake_len": 1,
                "low_score_ratio": 0.0,
                "pos_trace_ratio": 2.0,
                "recent_match_cost_mean": 0.3,
            },
        )
        second = DirtySuppressorAuditAccumulator()
        second.merge_state(first.export_state())
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "dirty.json"
            second.write_json(out)
            payload = json.loads(out.read_text())
        self.assertEqual(payload["buckets"][0]["class_name"], "bicycle")
