import contextlib
import io
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.build_centerpoint_mini_train_dataset import (
    DEFAULT_CLASS_DIST_TH,
    attach_unmatched_candidates,
    build_mini_dataset,
    filter_detections_for_runtime,
    match_classwise,
    normalize_timestamp_seconds,
    parse_class_dist_thresholds,
    resolve_class_dist_thresholds,
    resolve_split_scene_names,
)
from training.pairwise_association_cache import build_pairwise_association_samples


class BuildCenterPointMiniTrainDatasetTest(unittest.TestCase):
    def test_matching_maximizes_valid_pairs_before_minimizing_distance(self):
        gt = [
            {"global_xyz": [-0.1064643151, 0.9950107155, 0.0]},
            {"global_xyz": [1.7669064330, 1.3291223266, 0.0]},
        ]
        det = [
            {"global_xyz": [1.7353721092, -0.4699308120, 0.0]},
            {"global_xyz": [1.4151327680, 2.1463194659, 0.0]},
        ]
        matches, unmatched_gt, unmatched_det = match_classwise(gt, det, 2.0)
        self.assertEqual({(g, d) for g, d, _ in matches}, {(0, 1), (1, 0)})
        self.assertEqual(unmatched_gt, [])
        self.assertEqual(unmatched_det, [])

    def test_timestamp_diagnostic_accepts_microseconds_and_seconds(self):
        self.assertEqual(normalize_timestamp_seconds(1600000000500000, 1600000000.5), 1600000000.5)
        self.assertEqual(normalize_timestamp_seconds(1600000000.5, 1600000000.5), 1600000000.5)

    def test_detection_filter_uses_runtime_score_gate_and_nms(self):
        cfg = {
            "CATEGORY_MAP_TO_NUMBER": {"car": 0},
            "TRACKING_MODE": "ONLINE",
            "THRESHOLD": {"INPUT_SCORE": {"ONLINE": {0: 0.2}}},
        }
        boxes = [
            {"category": "car", "detection_score": 0.2},
            {"category": "car", "detection_score": 0.8},
            {"category": "car", "detection_score": 0.7},
        ]
        nms_inputs = []
        def fake_nms(items, _cfg):
            nms_inputs.extend(items)
            return items[:1]
        self.assertEqual(filter_detections_for_runtime(boxes, cfg, fake_nms), [boxes[1]])
        self.assertEqual(nms_inputs, boxes[1:])

    def test_attaches_unmatched_detection_only_once(self):
        frames = [{"scene_id": "scene", "sample_token": "s1"}]
        tracklets = [{"category": "car", "frames": frames}]
        candidate = {"candidate_id": "fp-1", "det_global_xyz": [4.0, 0.0, 0.0]}
        count = attach_unmatched_candidates(tracklets, {("scene", "s1", "car"): [candidate]})
        self.assertEqual(count, 1)
        self.assertEqual(frames[0]["unmatched_detections"], [candidate])

    def test_generated_cache_keeps_safe_unmatched_detection_for_association(self):
        class FakeNuScenes:
            def __init__(self, **_kwargs):
                pass

            def get(self, table, token):
                if table == "scene":
                    return {"name": "scene-1"}
                if table == "sample":
                    frame_id = int(token[-1])
                    return {"scene_token": "scene-token", "timestamp": 1600000000000000 + frame_id * 500000,
                            "anns": [f"ann-{frame_id}"]}
                frame_id = int(token[-1])
                return {"token": token, "instance_token": "gt-car", "category_name": "vehicle.car",
                        "translation": [float(frame_id), 0.0, 0.0], "size": [2.0, 4.0, 1.5],
                        "rotation": [1.0, 0.0, 0.0, 0.0]}

            def box_velocity(self, _token):
                return [2.0, 0.0, 0.0]

        scenes = {"scene-id": []}
        for frame_id in range(2):
            boxes = []
            for x in (float(frame_id), float(frame_id) + 2.5, float(frame_id) + 5.2):
                boxes.append({"category": "car", "global_xyz": [x, 0.0, 0.0],
                              "lwh": [4.0, 2.0, 1.5], "global_yaw": 0.0,
                              "global_velocity": [2.0, 0.0], "detection_score": 0.8})
            scenes["scene-id"].append({"cur_sample_token": f"sample-{frame_id}", "frame_id": frame_id,
                                       "timestamp": 1600000000000000 + frame_id * 500000, "bboxes": boxes})

        fake_package = types.ModuleType("nuscenes")
        fake_nusc_module = types.ModuleType("nuscenes.nuscenes")
        fake_nusc_module.NuScenes = FakeNuScenes
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "nuscenes": fake_package, "nuscenes.nuscenes": fake_nusc_module,
        }), patch("tools.build_centerpoint_mini_train_dataset.resolve_split_scene_names", return_value={"scene-1"}):
            det_path = Path(tmp) / "detections.json"
            output_path = Path(tmp) / "tracklets.pkl"
            det_path.write_text(json.dumps(scenes), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = build_mini_dataset(
                    str(det_path), "v1.0-trainval", tmp, str(output_path), 0, 2.0, 2, 2,
                    True, None, "val", None, False,
                )
            with output_path.open("rb") as stream:
                tracklets = pickle.load(stream)

        self.assertEqual(summary["timestamp_diff_max"], 0.0)
        self.assertEqual(summary["unmatched_candidates_attached"], 2)
        self.assertEqual(summary["ambiguous_unmatched_dets"], 2)
        self.assertEqual(len(tracklets), 1)
        self.assertEqual(len(tracklets[0]["frames"]), 2)
        samples, _ = build_pairwise_association_samples(
            tracklets, history_len=2, hard_negative_distance=6.0,
            max_hard_negatives=1, max_easy_negatives=0,
        )
        self.assertEqual([sample["label"] for sample in samples], [1, 0])
        self.assertTrue(samples[1]["candidate_is_unmatched_gt"])

    def test_resolve_split_scene_names_selects_requested_split(self):
        splits = {
            "train": ["scene-0001", "scene-0002"],
            "val": ["scene-0101"],
        }

        self.assertEqual(
            resolve_split_scene_names("train", splits=splits),
            {"scene-0001", "scene-0002"},
        )
        self.assertEqual(
            resolve_split_scene_names("val", splits=splits),
            {"scene-0101"},
        )

    def test_resolve_split_scene_names_rejects_unknown_split(self):
        with self.assertRaises(ValueError):
            resolve_split_scene_names("mini_train", splits={"train": []})

    def test_parse_class_dist_thresholds(self):
        parsed = parse_class_dist_thresholds("bicycle=1.2,motorcycle=1.5,trailer=2.5")

        self.assertEqual(parsed["bicycle"], 1.2)
        self.assertEqual(parsed["motorcycle"], 1.5)
        self.assertEqual(parsed["trailer"], 2.5)

    def test_resolve_class_dist_thresholds_uses_default_table_and_overrides(self):
        resolved = resolve_class_dist_thresholds(
            fallback_dist_th=2.0,
            use_default_class_dist_th=True,
            class_dist_th="bicycle=1.1,truck=2.7",
        )

        self.assertEqual(resolved["pedestrian"], DEFAULT_CLASS_DIST_TH["pedestrian"])
        self.assertEqual(resolved["bicycle"], 1.1)
        self.assertEqual(resolved["truck"], 2.7)

    def test_default_class_dist_thresholds_keep_large_vehicle_thresholds_wide(self):
        self.assertEqual(DEFAULT_CLASS_DIST_TH["bus"], 3.0)
        self.assertEqual(DEFAULT_CLASS_DIST_TH["trailer"], 3.0)
        self.assertEqual(DEFAULT_CLASS_DIST_TH["truck"], 3.0)

    def test_parse_class_dist_thresholds_rejects_unknown_class(self):
        with self.assertRaises(ValueError):
            parse_class_dist_thresholds("animal=1.0")


if __name__ == "__main__":
    unittest.main()
