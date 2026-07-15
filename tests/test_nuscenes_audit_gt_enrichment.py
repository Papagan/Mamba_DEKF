import copy
import sys
import types
import unittest
from unittest import mock

import numpy as np


if "pyquaternion" not in sys.modules:
    pyquaternion = types.ModuleType("pyquaternion")
    pyquaternion.Quaternion = object
    sys.modules["pyquaternion"] = pyquaternion

if "shapely.geometry" not in sys.modules:
    sys.modules.setdefault("shapely", types.ModuleType("shapely"))
    shapely_geometry = types.ModuleType("shapely.geometry")
    shapely_geometry.MultiPoint = object
    shapely_geometry.box = object
    sys.modules["shapely.geometry"] = shapely_geometry

if "nuscenes.nuscenes" not in sys.modules:
    sys.modules.setdefault("nuscenes", types.ModuleType("nuscenes"))
    nuscenes_nuscenes = types.ModuleType("nuscenes.nuscenes")
    nuscenes_nuscenes.NuScenes = object
    sys.modules["nuscenes.nuscenes"] = nuscenes_nuscenes

if "nuscenes.utils" not in sys.modules:
    nuscenes_utils = types.ModuleType("nuscenes.utils")
    sys.modules["nuscenes.utils"] = nuscenes_utils

if "nuscenes.utils.splits" not in sys.modules:
    nuscenes_splits = types.ModuleType("nuscenes.utils.splits")
    nuscenes_splits.create_splits_scenes = lambda: {"test": [], "val": []}
    sys.modules["nuscenes.utils.splits"] = nuscenes_splits
sys.modules["nuscenes.utils"].splits = sys.modules["nuscenes.utils.splits"]

from preprocess import convert_nuscenes
from tracker.bbox import BBox


class FakeNuScenes:
    def __init__(self, annotations):
        self.annotations = {
            annotation["token"]: annotation for annotation in annotations
        }

    def get(self, table, token):
        if table != "sample_annotation":
            raise KeyError(table)
        return self.annotations[token]


def _annotation(token, instance_token, category_name, xy):
    return {
        "token": token,
        "instance_token": instance_token,
        "category_name": category_name,
        "translation": [xy[0], xy[1], 0.0],
    }


def _detection(category, xy, score=0.75):
    return {
        "sample_token": "sample-1",
        "translation": [xy[0], xy[1], 0.25],
        "size": [1.8, 4.2, 1.6],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "velocity": [0.1, -0.2],
        "detection_name": category,
        "detection_score": score,
        "attribute_name": "",
    }


class NuScenesAuditGtEnrichmentTest(unittest.TestCase):
    def _enrich(self, annotations, detections, **kwargs):
        sample = {"token": "sample-1", "anns": [ann["token"] for ann in annotations]}
        return convert_nuscenes.enrich_detections_with_gt_identity(
            FakeNuScenes(annotations),
            sample,
            detections,
            enabled=True,
            split="val",
            **kwargs,
        )

    def test_maps_all_tracking_categories_and_matches_only_same_class(self):
        category_pairs = [
            ("bicycle", "vehicle.bicycle"),
            ("bus", "vehicle.bus.rigid"),
            ("car", "vehicle.car"),
            ("motorcycle", "vehicle.motorcycle"),
            ("pedestrian", "human.pedestrian.adult"),
            ("trailer", "vehicle.trailer"),
            ("truck", "vehicle.truck"),
        ]
        annotations = [
            _annotation(
                f"ann-{index}",
                f"instance-{category}",
                gt_category,
                (10.0 * index, 0.0),
            )
            for index, (category, gt_category) in enumerate(category_pairs)
        ]
        detections = [
            _detection(category, (10.0 * index + 0.1, 0.0))
            for index, (category, _) in enumerate(category_pairs)
        ]
        detections.append(_detection("car", (10.1, 0.0)))

        enriched = self._enrich(annotations, detections)

        self.assertEqual(
            [det.get("instance_token") for det in enriched[:-1]],
            [f"instance-{category}" for category, _ in category_pairs],
        )
        self.assertNotIn("instance_token", enriched[-1])

    def test_nearest_matching_is_one_to_one_and_deterministic(self):
        annotations = [
            _annotation("ann-a", "instance-a", "vehicle.car", (0.0, 0.0)),
            _annotation("ann-b", "instance-b", "vehicle.car", (10.0, 0.0)),
        ]
        detections = [
            _detection("car", (0.1, 0.0)),
            _detection("car", (9.9, 0.0)),
            _detection("car", (10.2, 0.0)),
        ]

        first = self._enrich(annotations, detections)
        second = self._enrich(annotations, detections)

        self.assertEqual(
            [det.get("instance_token") for det in first],
            ["instance-a", "instance-b", None],
        )
        self.assertEqual(first, second)
        matched = [det["instance_token"] for det in first if "instance_token" in det]
        self.assertEqual(len(matched), len(set(matched)))

    def test_outside_class_gate_and_unknown_categories_remain_without_tokens(self):
        annotations = [
            _annotation("ann-car", "instance-car", "vehicle.car", (0.0, 0.0)),
            _annotation("ann-animal", "instance-animal", "animal", (5.0, 0.0)),
        ]
        detections = [
            _detection("car", (2.01, 0.0)),
            _detection("animal", (5.0, 0.0)),
        ]

        enriched = self._enrich(annotations, detections)

        self.assertTrue(all("instance_token" not in det for det in enriched))

    def test_enabled_enrichment_does_not_trust_stale_detection_tokens(self):
        annotation = _annotation(
            "ann-car", "instance-car", "vehicle.car", (0.0, 0.0)
        )
        detection = _detection("car", (3.0, 0.0))
        detection.update(
            {
                "instance_token": "stale-instance",
                "gt_token": "stale-gt",
                "annotation_token": "stale-annotation",
            }
        )

        enriched = self._enrich([annotation], [detection])

        for token_name in ("instance_token", "gt_token", "annotation_token"):
            self.assertNotIn(token_name, enriched[0])

    def test_default_disabled_test_split_and_missing_annotations_are_safe(self):
        annotation = _annotation(
            "ann-car", "instance-car", "vehicle.car", (0.0, 0.0)
        )
        detection = _detection("car", (0.0, 0.0))
        nusc = FakeNuScenes([annotation])
        sample = {"token": "sample-1", "anns": [annotation["token"]]}

        disabled = convert_nuscenes.enrich_detections_with_gt_identity(
            nusc, sample, [detection], split="val"
        )
        test_split = convert_nuscenes.enrich_detections_with_gt_identity(
            nusc, sample, [detection], enabled=True, split="test"
        )
        no_anns = convert_nuscenes.enrich_detections_with_gt_identity(
            nusc, {"token": "sample-1"}, [detection], enabled=True, split="val"
        )

        for result in (disabled, test_split, no_anns):
            self.assertEqual(result, [detection])
            self.assertNotIn("instance_token", result[0])

    def test_enrichment_only_adds_tokens_without_changing_payload_or_order(self):
        annotations = [
            _annotation("ann-a", "instance-a", "vehicle.car", (0.0, 0.0)),
            _annotation("ann-b", "instance-b", "vehicle.truck", (10.0, 0.0)),
        ]
        detections = [
            _detection("truck", (10.1, 0.0), score=0.61),
            _detection("car", (0.1, 0.0), score=0.82),
        ]
        original = copy.deepcopy(detections)

        enriched = self._enrich(annotations, detections)

        stripped = [
            {key: value for key, value in detection.items() if key != "instance_token"}
            for detection in enriched
        ]
        self.assertEqual(stripped, original)
        self.assertEqual(detections, original)
        self.assertEqual(
            [det["instance_token"] for det in enriched],
            ["instance-b", "instance-a"],
        )

    def test_explicit_tokens_reach_converted_bbox_and_bbox_object(self):
        raw_detection = _detection("car", (1.0, 2.0))
        raw_detection.update(
            {
                "instance_token": "instance-real",
                "gt_token": "gt-real",
                "annotation_token": "annotation-real",
            }
        )
        with mock.patch.object(
            convert_nuscenes,
            "transform_box_global2image",
            return_value=(None, [0.0, 0.0, 0.0, 0.0]),
        ), mock.patch.object(
            convert_nuscenes, "transform_quaternion2yaw", return_value=0.0
        ):
            converted = convert_nuscenes.convert_baseversion_bboxes(
                [raw_detection], np.eye(4), {}
            )[0]

        bbox = BBox(0, converted)

        self.assertEqual(converted["instance_token"], "instance-real")
        self.assertEqual(converted["gt_token"], "gt-real")
        self.assertEqual(converted["annotation_token"], "annotation-real")
        self.assertEqual(bbox.instance_token, "instance-real")
        self.assertEqual(bbox.gt_token, "gt-real")
        self.assertEqual(bbox.annotation_token, "annotation-real")

    def test_cli_and_function_switch_are_explicit_and_default_off(self):
        parser = convert_nuscenes.build_arg_parser()

        self.assertFalse(parser.parse_args([]).audit_gt_enrichment)
        self.assertTrue(
            parser.parse_args(["--audit_gt_enrichment"]).audit_gt_enrichment
        )
        self.assertFalse(
            convert_nuscenes.nuscenes_main.__kwdefaults__["audit_gt_enrichment"]
        )


if __name__ == "__main__":
    unittest.main()
