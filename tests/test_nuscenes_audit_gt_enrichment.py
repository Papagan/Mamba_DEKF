import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = ("gt_token", "instance_token", "annotation_token")


def _module_from_file(module_name, relative_path, injected_modules):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, injected_modules):
        spec.loader.exec_module(module)
    return module


def _dependency_stubs():
    pyquaternion = types.ModuleType("pyquaternion")
    pyquaternion.Quaternion = object
    shapely = types.ModuleType("shapely")
    shapely_geometry = types.ModuleType("shapely.geometry")
    shapely_geometry.MultiPoint = object
    shapely_geometry.box = object
    nuscenes = types.ModuleType("nuscenes")
    nuscenes_nuscenes = types.ModuleType("nuscenes.nuscenes")
    nuscenes_nuscenes.NuScenes = object
    nuscenes_utils = types.ModuleType("nuscenes.utils")
    nuscenes_splits = types.ModuleType("nuscenes.utils.splits")
    nuscenes_splits.create_splits_scenes = lambda: {"test": [], "val": []}
    nuscenes_utils.splits = nuscenes_splits
    return {
        "pyquaternion": pyquaternion,
        "shapely": shapely,
        "shapely.geometry": shapely_geometry,
        "nuscenes": nuscenes,
        "nuscenes.nuscenes": nuscenes_nuscenes,
        "nuscenes.utils": nuscenes_utils,
        "nuscenes.utils.splits": nuscenes_splits,
    }


CONVERTER = _module_from_file(
    "_test_convert_nuscenes",
    "preprocess/convert_nuscenes.py",
    _dependency_stubs(),
)


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


def _converted_bbox(category="car", score=0.9, **extra):
    bbox = {
        "category": category,
        "detection_score": score,
        "lwh": [4.0, 2.0, 1.5],
        "global_xyz": [1.0, 2.0, 0.0],
        "global_orientation": [1.0, 0.0, 0.0, 0.0],
        "global_yaw": 0.0,
        "global_velocity": [0.0, 0.0],
        "global_acceleration": [0.0, 0.0],
        "bbox_image": {},
    }
    bbox.update(extra)
    return bbox


class NuScenesAuditGtEnrichmentTest(unittest.TestCase):
    def _enrich(self, annotations, detections, **kwargs):
        sample = {"token": "sample-1", "anns": [ann["token"] for ann in annotations]}
        return CONVERTER.enrich_detections_with_gt_identity(
            FakeNuScenes(annotations),
            sample,
            detections,
            enabled=True,
            split="val",
            **kwargs,
        )

    def test_hungarian_maximizes_feasible_matches_before_distance(self):
        annotations = [
            _annotation("ann-a", "instance-a", "vehicle.car", (0.0, 0.0)),
            _annotation("ann-b", "instance-b", "vehicle.car", (1.8, 0.0)),
        ]
        detections = [
            _detection("car", (0.8, 0.0)),
            _detection("car", (-0.9, 0.0)),
        ]

        cleaned, tokens = self._enrich(
            annotations,
            detections,
            distance_by_class={"car": 1.0},
        )

        self.assertEqual(tokens, ["instance-b", "instance-a"])
        self.assertEqual(cleaned, detections)

    def test_matches_only_same_class_and_preserves_detection_order(self):
        annotations = [
            _annotation("ann-car", "instance-car", "vehicle.car", (0.0, 0.0)),
            _annotation("ann-truck", "instance-truck", "vehicle.truck", (5.0, 0.0)),
        ]
        detections = [
            _detection("truck", (5.1, 0.0), score=0.61),
            _detection("car", (0.1, 0.0), score=0.82),
            _detection("car", (5.0, 0.0)),
        ]

        cleaned, tokens = self._enrich(annotations, detections)

        self.assertEqual(cleaned, detections)
        self.assertEqual(tokens, ["instance-truck", "instance-car", None])

    def test_all_paths_strip_stale_identity_and_disabled_has_no_sidecar(self):
        annotation = _annotation(
            "ann-car", "instance-car", "vehicle.car", (0.0, 0.0)
        )
        stale = _detection("car", (3.0, 0.0))
        stale.update(dict.fromkeys(IDENTITY_FIELDS, "stale"))
        sample = {"token": "sample-1", "anns": [annotation["token"]]}
        nusc = FakeNuScenes([annotation])

        results = [
            CONVERTER.enrich_detections_with_gt_identity(
                nusc, sample, [stale], split="val"
            ),
            CONVERTER.enrich_detections_with_gt_identity(
                nusc, sample, [stale], enabled=True, split="test"
            ),
            CONVERTER.enrich_detections_with_gt_identity(
                nusc, {"token": "sample-1"}, [stale], enabled=True, split="val"
            ),
        ]

        for cleaned, tokens in results:
            self.assertIsNone(tokens)
            self.assertTrue(all(field not in cleaned[0] for field in IDENTITY_FIELDS))
        self.assertTrue(all(field in stale for field in IDENTITY_FIELDS))

    def test_converter_never_copies_identity_into_bbox_payload(self):
        raw_detection = _detection("car", (1.0, 2.0))
        raw_detection.update(dict.fromkeys(IDENTITY_FIELDS, "must-not-survive"))
        with mock.patch.object(
            CONVERTER,
            "transform_box_global2image",
            return_value=(None, [0.0, 0.0, 0.0, 0.0]),
        ), mock.patch.object(
            CONVERTER, "transform_quaternion2yaw", return_value=0.0
        ):
            converted = CONVERTER.convert_baseversion_bboxes(
                [raw_detection], np.eye(4), {}
            )[0]

        self.assertTrue(all(field not in converted for field in IDENTITY_FIELDS))

    def test_cli_and_function_switch_are_explicit_and_default_off(self):
        parser = CONVERTER.build_arg_parser()

        self.assertFalse(parser.parse_args([]).audit_gt_enrichment)
        self.assertTrue(parser.parse_args(["--audit_gt_enrichment"]).audit_gt_enrichment)
        self.assertFalse(CONVERTER.nuscenes_main.__kwdefaults__["audit_gt_enrichment"])


class ConverterMainIntegrationTest(unittest.TestCase):
    class _FakeNuScenes:
        scene = [{"name": "scene-1", "first_sample_token": "sample-1"}]

        def __init__(self, annotation):
            self.annotation = annotation

        def get(self, table, token):
            records = {
                "sample": {
                    "token": "sample-1",
                    "anns": [self.annotation["token"]],
                    "data": {"LIDAR_TOP": "lidar-1"},
                    "timestamp": 500000,
                    "next": "",
                },
                "sample_data": {
                    "ego_pose_token": "ego-1",
                    "calibrated_sensor_token": "sensor-1",
                },
                "ego_pose": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                },
                "calibrated_sensor": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                },
                "sample_annotation": self.annotation,
            }
            return records[table]

    def _run(self, enabled):
        annotation = _annotation(
            "ann-car", "instance-car", "vehicle.car", (0.0, 0.0)
        )
        detection = _detection("car", (0.1, 0.0))
        detection.update(dict.fromkeys(IDENTITY_FIELDS, "stale"))
        fake_nusc = self._FakeNuScenes(annotation)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detector_dir = root / "dets" / "centerpoint"
            detector_dir.mkdir(parents=True)
            (detector_dir / "val.json").write_text(
                json.dumps({"results": {"sample-1": [detection]}}),
                encoding="utf-8",
            )
            with mock.patch.object(CONVERTER, "NuScenes", return_value=fake_nusc), \
                 mock.patch.object(
                     CONVERTER.splits,
                     "create_splits_scenes",
                     return_value={"val": ["scene-1"], "test": []},
                 ), \
                 mock.patch.object(CONVERTER, "load_camera_info", return_value={}), \
                 mock.patch.object(CONVERTER, "transform_matrix", return_value=np.eye(4)), \
                 mock.patch.object(
                     CONVERTER,
                     "convert_baseversion_bboxes",
                     side_effect=lambda boxes, *_: [
                         _converted_bbox(category=box["detection_name"])
                         for box in boxes
                     ],
                 ):
                CONVERTER.nuscenes_main(
                    "raw",
                    str(root / "dets"),
                    "centerpoint",
                    str(root / "out"),
                    "val",
                    audit_gt_enrichment=enabled,
                )
            return json.loads(
                (root / "out" / "centerpoint" / "val.json").read_text(
                    encoding="utf-8"
                )
            )["scene-1"][0]

    def test_nuscenes_main_writes_sidecar_only_when_flag_is_enabled(self):
        enabled = self._run(True)
        disabled = self._run(False)

        self.assertEqual(enabled["audit_gt_tokens"], ["instance-car"])
        self.assertNotIn("audit_gt_tokens", disabled)
        for frame in (enabled, disabled):
            self.assertTrue(
                all(field not in frame["bboxes"][0] for field in IDENTITY_FIELDS)
            )


class DatasetAuditSidecarTest(unittest.TestCase):
    def _load_dataset_module(self, nms):
        nusc_utils = types.ModuleType("utils.nusc_utils")
        nusc_utils.obtain_box_bottom_corners = lambda *args, **kwargs: None
        nusc_utils.filter_bboxes_with_nms = nms
        stubs = _dependency_stubs()
        bbox_module = _module_from_file(
            "_test_tracker_bbox", "tracker/bbox.py", stubs
        )
        tracker_bbox = types.ModuleType("tracker.bbox")
        tracker_bbox.BBox = bbox_module.BBox
        tracker_package = types.ModuleType("tracker")
        tracker_package.__path__ = []
        frame_stubs = dict(stubs)
        frame_stubs.update({"tracker": tracker_package, "tracker.bbox": tracker_bbox})
        frame_module = _module_from_file(
            "tracker.frame", "tracker/frame.py", frame_stubs
        )
        tracker_frame = types.ModuleType("tracker.frame")
        tracker_frame.Frame = frame_module.Frame
        stubs.update(
            {
                "tracker": tracker_package,
                "tracker.bbox": tracker_bbox,
                "tracker.frame": tracker_frame,
            }
        )
        stubs["utils.nusc_utils"] = nusc_utils
        return _module_from_file(
            "_test_baseversion_dataset",
            "dataset/baseversion_dataset.py",
            stubs,
        )

    def test_filtering_and_nms_keep_sidecar_aligned_without_bbox_tokens(self):
        module = self._load_dataset_module(
            lambda boxes, cfg: [boxes[2], boxes[0]]
        )
        scene_bboxes = [
            _converted_bbox(category="truck"),
            _converted_bbox(score=0.1),
            _converted_bbox(global_xyz=[2.0, 0.0, 0.0]),
            _converted_bbox(global_xyz=[3.0, 0.0, 0.0]),
            _converted_bbox(global_xyz=[4.0, 0.0, 0.0]),
        ]
        scene_bboxes[2].update(dict.fromkeys(IDENTITY_FIELDS, "stale"))
        dataset = module.BaseVersionTrackingDataset(
            "scene-real",
            [{
                "frame_id": "3",
                "timestamp": 12.5,
                "cur_sample_token": "sample-real",
                "transform_matrix": np.eye(4).tolist(),
                "bboxes": scene_bboxes,
                "audit_gt_tokens": ["truck", "low", "keep-a", "drop", "keep-b"],
            }],
            cfg={
                "CATEGORY_MAP_TO_NUMBER": {"car": 0},
                "TRACKING_MODE": "ONLINE",
                "THRESHOLD": {
                    "INPUT_SCORE": {"ONLINE": {0: 0.5}},
                    "NMS_THRE": 0.5,
                },
                "DATASET": "nuscenes",
            },
        )

        frame = dataset[0]

        self.assertEqual(frame.audit_gt_tokens, ["keep-b", "keep-a"])
        self.assertEqual(
            [bbox.global_xyz[0] for bbox in frame.bboxes],
            [4.0, 2.0],
        )
        self.assertTrue(
            all(
                not hasattr(bbox, field)
                for bbox in frame.bboxes
                for field in IDENTITY_FIELDS
            )
        )

    def test_disabled_frame_has_no_audit_sidecar_and_strips_stale_tokens(self):
        module = self._load_dataset_module(lambda boxes, cfg: boxes)
        dataset = module.BaseVersionTrackingDataset(
            "scene-real",
            [{
                "frame_id": 0,
                "timestamp": 0.0,
                "cur_sample_token": "sample-real",
                "transform_matrix": {},
                "bboxes": [
                    _converted_bbox(**dict.fromkeys(IDENTITY_FIELDS, "stale"))
                ],
            }],
            cfg={
                "CATEGORY_MAP_TO_NUMBER": {"car": 0},
                "TRACKING_MODE": "ONLINE",
                "THRESHOLD": {"INPUT_SCORE": {"ONLINE": {0: 0.0}}},
                "DATASET": "kitti",
            },
        )

        frame = dataset[0]

        self.assertFalse(hasattr(frame, "audit_gt_tokens"))
        self.assertTrue(
            all(not hasattr(frame.bboxes[0], field) for field in IDENTITY_FIELDS)
        )


def _load_base_tracker_methods(*names):
    tree = ast.parse((ROOT / "tracker/base_tracker.py").read_text(encoding="utf-8"))
    tracker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Base3DTracker"
    )
    selected = [
        node
        for node in tracker_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f"Missing Base3DTracker methods: {sorted(missing)}")
    namespace = {
        "Dict": dict,
        "List": list,
        "Sequence": __import__("collections.abc").abc.Sequence,
        "infer_state_bucket": lambda length: "unmatched" if length else "matched",
        "np": np,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "base_tracker.py", "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


class TrackerAuditSidecarTest(unittest.TestCase):
    METHODS = (
        "_association_frame_gt_tokens",
        "_association_det_gt_tokens",
        "_update_association_gt_sidecar",
        "_association_shadow_options",
        "_record_association_shadow_decisions",
        "_record_association_shadow_error",
    )

    def _tracker(self, methods):
        audit = types.SimpleNamespace(add_decision=lambda record: None, add_error=lambda record: None)
        tracker = types.SimpleNamespace(
            association_head_audit=audit,
            association_head_audit_cfg={"ENABLED": True, "SHADOW_MODE": True},
            cfg={"CATEGORY_MAP_TO_NUMBER": {"trailer": 5}},
            _association_track_gt_tokens={17: "track-gt"},
        )
        for name, method in methods.items():
            setattr(tracker, name, types.MethodType(method, tracker))
        return tracker

    def test_shadow_metadata_reads_only_frame_and_track_sidecars(self):
        methods = _load_base_tracker_methods(*self.METHODS)
        tracker = self._tracker(methods)
        traj = types.SimpleNamespace(
            track_id=17,
            unmatch_length=2,
            bboxes=[types.SimpleNamespace(category="trailer")],
        )
        det = types.SimpleNamespace(category="trailer")
        frame = types.SimpleNamespace(
            scene_id="scene-real",
            cur_sample_token="sample-real",
            frame_id=9,
            bboxes=[det],
            audit_gt_tokens=["det-gt"],
        )

        metadata = tracker._association_shadow_options(
            frame, [traj], [det]
        )["association_shadow_metadata"]

        self.assertEqual(metadata["track_gt_tokens"], ["track-gt"])
        self.assertEqual(metadata["det_gt_tokens"], ["det-gt"])
        self.assertEqual(metadata["state_buckets"], ["unmatched"])

    def test_matches_and_births_replace_track_sidecar_identity(self):
        methods = _load_base_tracker_methods(*self.METHODS)
        tracker = self._tracker(methods)
        frame = types.SimpleNamespace(
            bboxes=[object(), object()],
            audit_gt_tokens=["matched-gt", "birth-gt"],
        )

        tracker._update_association_gt_sidecar(17, frame, 0)
        tracker._update_association_gt_sidecar(18, frame, 1)

        self.assertEqual(
            tracker._association_track_gt_tokens,
            {17: "matched-gt", 18: "birth-gt"},
        )

    def test_missing_sidecar_never_falls_back_to_bbox_attributes(self):
        methods = _load_base_tracker_methods(*self.METHODS)
        tracker = self._tracker(methods)
        det = types.SimpleNamespace(category="trailer", instance_token="forbidden")
        frame = types.SimpleNamespace(bboxes=[det])
        traj = types.SimpleNamespace(
            track_id=19,
            unmatch_length=0,
            bboxes=[types.SimpleNamespace(category="trailer", instance_token="forbidden")],
        )

        metadata = tracker._association_shadow_options(
            frame, [traj], [det]
        )["association_shadow_metadata"]
        tracker._update_association_gt_sidecar(19, frame, 0)

        self.assertEqual(metadata["track_gt_tokens"], [None])
        self.assertEqual(metadata["det_gt_tokens"], [None])
        self.assertNotIn(19, tracker._association_track_gt_tokens)


class ModuleIsolationTest(unittest.TestCase):
    def test_dependency_stubs_restore_preexisting_sys_modules(self):
        stubs = _dependency_stubs()
        missing = object()
        before = {name: sys.modules.get(name, missing) for name in stubs}

        _module_from_file(
            "_test_convert_nuscenes_isolation",
            "preprocess/convert_nuscenes.py",
            stubs,
        )

        for name, previous in before.items():
            self.assertIs(sys.modules.get(name, missing), previous)


if __name__ == "__main__":
    unittest.main()
