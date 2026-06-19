import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_noise_audit import build_summary, load_audits, render_text


def _sample_payload(mode, *, split="infer", q_pos_family, q_pos_ratio, q_pos_p90=None, q_pos_ratio_p90=None):
    return {
        "schema_version": 1,
        "families": ["q_pos", "r_pos", "r_siz", "r_ori"],
        "buckets": [
            {
                "split": split,
                "mode": mode,
                "class_id": 5,
                "class_name": "trailer",
                "state": "matched",
                "history_len": 6,
                "count": 10,
                "families": {
                    "q_pos": {
                        "count": 10,
                        "mean": q_pos_family,
                        "median": q_pos_family,
                        "p90": q_pos_p90 if q_pos_p90 is not None else q_pos_family + 1.0,
                        "min": q_pos_family,
                        "max": q_pos_family + 2.0,
                    }
                },
                "ratios": {
                    "q_pos": {
                        "count": 10,
                        "mean": q_pos_ratio,
                        "median": q_pos_ratio,
                        "p90": q_pos_ratio_p90 if q_pos_ratio_p90 is not None else q_pos_ratio + 0.5,
                        "min": q_pos_ratio,
                        "max": q_pos_ratio + 1.0,
                    }
                },
            }
        ],
    }


class SummarizeNoiseAuditTest(unittest.TestCase):
    def _write_payload(self, directory, name, payload):
        path = Path(directory) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_loads_single_audit_and_emits_compact_bucket_summary(self):
        payload = _sample_payload("fusion", q_pos_family=2.0, q_pos_ratio=1.4, q_pos_p90=3.0, q_pos_ratio_p90=1.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_payload(tmpdir, "fusion.json", payload)
            summary = build_summary(load_audits([path]))

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["families"], ["q_pos", "r_pos", "r_siz", "r_ori"])
        self.assertEqual(len(summary["rows"]), 1)
        row = summary["rows"][0]
        self.assertEqual(row["split"], "infer")
        self.assertEqual(row["class_name"], "trailer")
        self.assertEqual(row["state"], "matched")
        self.assertEqual(row["history_len"], 6)
        self.assertEqual(row["family"], "q_pos")
        self.assertEqual(list(row["modes"]), ["fusion"])
        self.assertEqual(row["modes"]["fusion"]["family_median"], 2.0)
        self.assertEqual(row["modes"]["fusion"]["family_p90"], 3.0)
        self.assertEqual(row["modes"]["fusion"]["ratio_median"], 1.4)
        self.assertEqual(row["modes"]["fusion"]["ratio_p90"], 1.9)

    def test_compares_multiple_modes_for_same_bucket(self):
        fusion = _sample_payload("fusion", q_pos_family=2.0, q_pos_ratio=1.4, q_pos_p90=3.0, q_pos_ratio_p90=1.9)
        mamba = _sample_payload("mamba", q_pos_family=4.0, q_pos_ratio=2.1, q_pos_p90=5.0, q_pos_ratio_p90=2.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            fusion_path = self._write_payload(tmpdir, "fusion.json", fusion)
            mamba_path = self._write_payload(tmpdir, "mamba.json", mamba)
            summary = build_summary(load_audits([fusion_path, mamba_path]))

        self.assertEqual(len(summary["rows"]), 1)
        row = summary["rows"][0]
        self.assertEqual(row["splits"], ["infer"])
        self.assertEqual(row["split"], "infer")
        self.assertEqual(list(row["modes"]), ["fusion", "mamba"])
        self.assertEqual(row["modes"]["fusion"]["family_median"], 2.0)
        self.assertEqual(row["modes"]["mamba"]["family_median"], 4.0)
        self.assertEqual(row["modes"]["fusion"]["ratio_p90"], 1.9)
        self.assertEqual(row["modes"]["mamba"]["ratio_p90"], 2.8)

    def test_rejects_mixed_split_comparison_rows(self):
        fusion = _sample_payload(
            "fusion",
            split="infer",
            q_pos_family=2.0,
            q_pos_ratio=1.4,
            q_pos_p90=3.0,
            q_pos_ratio_p90=1.9,
        )
        mamba = _sample_payload(
            "mamba",
            split="train",
            q_pos_family=4.0,
            q_pos_ratio=2.1,
            q_pos_p90=5.0,
            q_pos_ratio_p90=2.8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fusion_path = self._write_payload(tmpdir, "fusion.json", fusion)
            mamba_path = self._write_payload(tmpdir, "mamba.json", mamba)
            with self.assertRaisesRegex(ValueError, "mixed split comparison entry"):
                build_summary(load_audits([fusion_path, mamba_path]))

    def test_render_text_keeps_mode_information_and_ratio_stats_compact(self):
        payload = _sample_payload("pure_dekf", q_pos_family=2.5, q_pos_ratio=1.2, q_pos_p90=3.5, q_pos_ratio_p90=1.7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_payload(tmpdir, "pure_dekf.json", payload)
            summary = build_summary(load_audits([path]))

        rendered = render_text(summary)

        self.assertIn("infer trailer matched h=6 q_pos", rendered)
        self.assertIn("pure_dekf", rendered)
        self.assertIn("family=2.5/3.5", rendered)
        self.assertIn("ratio=1.2/1.7", rendered)


if __name__ == "__main__":
    unittest.main()
