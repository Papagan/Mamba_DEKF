import unittest

from training.class_state_metrics import (
    class_state_bucket_key,
    init_class_state_metric_accumulator,
    update_class_state_metric_accumulator,
    finalize_class_state_metric_accumulator,
)


class ClassStateMetricTest(unittest.TestCase):
    def test_class_state_bucket_key_is_stable(self):
        self.assertEqual(class_state_bucket_key(2, "unmatched"), "class_2/unmatched")
        self.assertEqual(class_state_bucket_key("5", "matched"), "class_5/matched")

    def test_metric_accumulator_averages_by_sample_count(self):
        acc = init_class_state_metric_accumulator()
        update_class_state_metric_accumulator(
            acc,
            class_ids=[2, 2, 5],
            state_buckets=["unmatched", "unmatched", "matched"],
            metrics={
                "loss_real": [1.0, 3.0, 9.0],
                "q_pos_ratio_mean": [1.1, 1.3, 0.9],
            },
        )
        out = finalize_class_state_metric_accumulator(acc)
        self.assertEqual(out["class_2/unmatched/count"], 2)
        self.assertAlmostEqual(out["class_2/unmatched/loss_real"], 2.0)
        self.assertAlmostEqual(out["class_2/unmatched/q_pos_ratio_mean"], 1.2)
        self.assertEqual(out["class_5/matched/count"], 1)
        self.assertAlmostEqual(out["class_5/matched/loss_real"], 9.0)

    def test_metric_accumulator_uses_per_metric_counts(self):
        acc = init_class_state_metric_accumulator()
        update_class_state_metric_accumulator(
            acc,
            class_ids=[2],
            state_buckets=["matched"],
            metrics={"loss_real": [10.0]},
        )
        update_class_state_metric_accumulator(
            acc,
            class_ids=[2],
            state_buckets=["matched"],
            metrics={"q_pos_ratio_mean": [4.0]},
        )
        out = finalize_class_state_metric_accumulator(acc)
        self.assertEqual(out["class_2/matched/count"], 2)
        self.assertAlmostEqual(out["class_2/matched/loss_real"], 10.0)
        self.assertAlmostEqual(out["class_2/matched/q_pos_ratio_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
