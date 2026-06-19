import json
import tempfile
import unittest
from pathlib import Path

from kalmanfilter.noise_audit import NoiseAuditAccumulator


class NoiseAuditAccumulatorTest(unittest.TestCase):
    def test_summary_has_schema_version_and_top_level_keys(self):
        acc = NoiseAuditAccumulator()

        payload = acc.to_summary()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["families"], ["q_pos", "r_pos", "r_siz", "r_ori"])
        self.assertIn("buckets", payload)

    def test_records_grouped_stats_and_ratios(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="infer",
            mode="fusion",
            class_id=5,
            class_name="trailer",
            state="matched",
            history_len=6,
            families={
                "q_pos": 2.0,
                "r_pos": 4.0,
                "r_siz": 6.0,
                "r_ori": 8.0,
            },
            prior_families={
                "q_pos": 1.0,
                "r_pos": 2.0,
                "r_siz": 3.0,
                "r_ori": 4.0,
            },
        )

        payload = acc.to_summary()

        bucket = payload["buckets"][0]
        self.assertEqual(bucket["split"], "infer")
        self.assertEqual(bucket["mode"], "fusion")
        self.assertEqual(bucket["class_id"], 5)
        self.assertEqual(bucket["state"], "matched")
        self.assertEqual(bucket["count"], 1)
        self.assertAlmostEqual(bucket["families"]["q_pos"]["median"], 2.0)
        self.assertAlmostEqual(bucket["ratios"]["q_pos"]["median"], 2.0)

    def test_ignores_missing_optional_history_length(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="train",
            mode="mamba",
            class_id=2,
            class_name="bicycle",
            state="unmatched",
            history_len=None,
            families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
            prior_families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
        )

        payload = acc.to_summary()

        self.assertIsNone(payload["buckets"][0]["history_len"])

    def test_skips_non_positive_or_non_finite_ratio_priors(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="infer",
            mode="pure_dekf",
            class_id=1,
            class_name="car",
            state="matched",
            history_len=3,
            families={"q_pos": 3.0, "r_pos": 5.0, "r_siz": 7.0, "r_ori": 11.0},
            prior_families={"q_pos": 0.0, "r_pos": -1.0, "r_siz": float("inf"), "r_ori": 2.0},
        )

        bucket = acc.to_summary()["buckets"][0]

        self.assertEqual(bucket["ratios"]["q_pos"]["count"], 0)
        self.assertIsNone(bucket["ratios"]["q_pos"]["median"])
        self.assertEqual(bucket["ratios"]["r_pos"]["count"], 0)
        self.assertEqual(bucket["ratios"]["r_siz"]["count"], 0)
        self.assertEqual(bucket["ratios"]["r_ori"]["count"], 1)
        self.assertAlmostEqual(bucket["ratios"]["r_ori"]["median"], 5.5)

    def test_skips_non_finite_observed_family_values(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="infer",
            mode="fusion",
            class_id=3,
            class_name="bus",
            state="matched",
            history_len=5,
            families={
                "q_pos": float("nan"),
                "r_pos": float("inf"),
                "r_siz": 9.0,
                "r_ori": float("-inf"),
            },
            prior_families={
                "q_pos": 1.0,
                "r_pos": 1.0,
                "r_siz": 3.0,
                "r_ori": 1.0,
            },
        )

        bucket = acc.to_summary()["buckets"][0]

        self.assertEqual(bucket["count"], 1)
        self.assertEqual(bucket["families"]["q_pos"]["count"], 0)
        self.assertEqual(bucket["families"]["r_pos"]["count"], 0)
        self.assertEqual(bucket["families"]["r_ori"]["count"], 0)
        self.assertEqual(bucket["families"]["r_siz"]["count"], 1)
        self.assertAlmostEqual(bucket["families"]["r_siz"]["median"], 9.0)
        self.assertEqual(bucket["ratios"]["r_siz"]["count"], 1)
        self.assertAlmostEqual(bucket["ratios"]["r_siz"]["median"], 3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit" / "summary.json"
            acc.write_json(path)
            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(written["buckets"][0]["families"]["q_pos"]["median"], None)
        self.assertAlmostEqual(written["buckets"][0]["families"]["r_siz"]["median"], 9.0)

    def test_sorts_buckets_stably(self):
        acc = NoiseAuditAccumulator()
        samples = [
            ("train", "mamba", 2, "bicycle", "matched", None),
            ("infer", "fusion", 5, "trailer", "unmatched", 4),
            ("infer", "fusion", 5, "trailer", "matched", 2),
            ("infer", "fusion", 5, "trailer", "matched", None),
        ]

        for split, mode, class_id, class_name, state, history_len in samples:
            acc.add_sample(
                split=split,
                mode=mode,
                class_id=class_id,
                class_name=class_name,
                state=state,
                history_len=history_len,
                families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
                prior_families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
            )

        buckets = acc.to_summary()["buckets"]

        self.assertEqual(
            [
                (bucket["split"], bucket["mode"], bucket["class_id"], bucket["state"], bucket["history_len"])
                for bucket in buckets
            ],
            [
                ("infer", "fusion", 5, "matched", 2),
                ("infer", "fusion", 5, "matched", None),
                ("infer", "fusion", 5, "unmatched", 4),
                ("train", "mamba", 2, "matched", None),
            ],
        )

    def test_writes_json_summary(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="infer",
            mode="fusion",
            class_id=0,
            class_name="car",
            state="matched",
            history_len=1,
            families={"q_pos": 2.0, "r_pos": 2.0, "r_siz": 2.0, "r_ori": 2.0},
            prior_families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit" / "summary.json"
            acc.write_json(path)
            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(written, acc.to_summary())

    def test_exports_and_merges_bucket_state_exactly(self):
        first = NoiseAuditAccumulator()
        first.add_sample(
            split="infer",
            mode="fusion",
            class_id=0,
            class_name="car",
            state="matched",
            history_len=2,
            families={"q_pos": 1.0, "r_pos": 2.0, "r_siz": 3.0, "r_ori": 4.0},
            prior_families={"q_pos": 0.5, "r_pos": 1.0, "r_siz": 1.5, "r_ori": 2.0},
        )
        second = NoiseAuditAccumulator()
        second.add_sample(
            split="infer",
            mode="fusion",
            class_id=0,
            class_name="car",
            state="matched",
            history_len=2,
            families={"q_pos": 5.0, "r_pos": 6.0, "r_siz": 7.0, "r_ori": 8.0},
            prior_families={"q_pos": 2.5, "r_pos": 3.0, "r_siz": 3.5, "r_ori": 4.0},
        )

        merged = NoiseAuditAccumulator()
        merged.merge_state(first.export_state())
        merged.merge_state(second.export_state())

        bucket = merged.to_summary()["buckets"][0]
        self.assertEqual(bucket["count"], 2)
        self.assertAlmostEqual(bucket["families"]["q_pos"]["median"], 3.0)
        self.assertAlmostEqual(bucket["families"]["r_ori"]["max"], 8.0)
        self.assertAlmostEqual(bucket["ratios"]["q_pos"]["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
