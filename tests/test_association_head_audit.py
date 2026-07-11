import tempfile
import unittest
from pathlib import Path

from tracker.association_head_audit import AssociationHeadAuditAccumulator


class AssociationHeadAuditAccumulatorTest(unittest.TestCase):
    def test_summarizes_penalties_by_class_and_state(self):
        audit = AssociationHeadAuditAccumulator()

        audit.add_pair(
            class_id=5,
            class_name="trailer",
            state_bucket="unmatched",
            score=0.4,
            delta=0.01,
            cost_before=1.0,
            cost_after=1.01,
            active=True,
            finite=True,
        )
        audit.add_pair(
            class_id=5,
            class_name="trailer",
            state_bucket="unmatched",
            score=0.8,
            delta=0.0,
            cost_before=0.9,
            cost_after=0.9,
            active=True,
            finite=True,
        )

        summary = audit.to_summary()
        bucket = summary["buckets"][0]

        self.assertEqual(bucket["class_id"], 5)
        self.assertEqual(bucket["class_name"], "trailer")
        self.assertEqual(bucket["state_bucket"], "unmatched")
        self.assertEqual(bucket["pair_count"], 2)
        self.assertEqual(bucket["active_pair_count"], 2)
        self.assertEqual(bucket["penalized_pair_count"], 1)
        self.assertAlmostEqual(bucket["penalized_ratio"], 0.5)
        self.assertAlmostEqual(bucket["avg_score"], 0.6)
        self.assertAlmostEqual(bucket["avg_delta"], 0.005)
        self.assertAlmostEqual(bucket["score_quantiles"]["p50"], 0.6)
        self.assertAlmostEqual(bucket["score_quantiles"]["p90"], 0.76)
        self.assertAlmostEqual(bucket["delta_quantiles"]["p50"], 0.005)

    def test_keeps_bounded_sample_records_for_replay_debugging(self):
        audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)

        for idx in range(3):
            audit.add_pair(
                class_id=2,
                class_name="bicycle",
                state_bucket="matched",
                score=0.1 * idx,
                delta=0.01 * idx,
                cost_before=1.0 + idx,
                cost_after=1.0 + idx + 0.01 * idx,
                active=idx != 2,
                finite=True,
                sample={
                    "scene_id": "scene",
                    "track_index": 0,
                    "det_index": idx,
                    "center_distance": float(idx),
                },
            )

        bucket = audit.to_summary()["buckets"][0]
        self.assertEqual(bucket["sample_count"], 2)
        self.assertEqual(len(bucket["samples"]), 2)
        self.assertEqual(bucket["samples"][0]["det_index"], 0)
        self.assertEqual(bucket["samples"][1]["det_index"], 1)

    def test_merges_exported_states_and_writes_json(self):
        first = AssociationHeadAuditAccumulator()
        second = AssociationHeadAuditAccumulator()
        first.add_pair(
            class_id=3,
            class_name="motorcycle",
            state_bucket="matched",
            score=0.2,
            delta=0.02,
            cost_before=1.0,
            cost_after=1.02,
            active=True,
            finite=True,
        )
        second.add_pair(
            class_id=3,
            class_name="motorcycle",
            state_bucket="matched",
            score=0.6,
            delta=0.0,
            cost_before=1.0,
            cost_after=1.0,
            active=False,
            finite=True,
        )

        merged = AssociationHeadAuditAccumulator()
        merged.merge_state(first.export_state())
        merged.merge_state(second.export_state())

        bucket = merged.to_summary()["buckets"][0]
        self.assertEqual(bucket["pair_count"], 2)
        self.assertEqual(bucket["active_pair_count"], 1)
        self.assertEqual(bucket["penalized_pair_count"], 1)
        self.assertAlmostEqual(bucket["avg_score"], 0.4)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            merged.write_json(path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
