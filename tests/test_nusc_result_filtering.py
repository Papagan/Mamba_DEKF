import unittest
import sys
import types


if "pyquaternion" not in sys.modules:
    pyquaternion = types.ModuleType("pyquaternion")

    class _Quaternion:
        def __init__(self, *args, **kwargs):
            pass

    pyquaternion.Quaternion = _Quaternion
    sys.modules["pyquaternion"] = pyquaternion

if "nuscenes" not in sys.modules:
    nuscenes = types.ModuleType("nuscenes")
    nuscenes_utils = types.ModuleType("nuscenes.utils")
    nuscenes_data_classes = types.ModuleType("nuscenes.utils.data_classes")

    class _Box:
        def __init__(self, *args, **kwargs):
            pass

    nuscenes_data_classes.Box = _Box
    nuscenes.utils = nuscenes_utils
    sys.modules["nuscenes"] = nuscenes
    sys.modules["nuscenes.utils"] = nuscenes_utils
    sys.modules["nuscenes.utils.data_classes"] = nuscenes_data_classes

if "shapely" not in sys.modules:
    shapely = types.ModuleType("shapely")
    shapely_geometry = types.ModuleType("shapely.geometry")

    class _Polygon:
        def __init__(self, *args, **kwargs):
            pass

    shapely_geometry.Polygon = _Polygon
    sys.modules["shapely"] = shapely
    sys.modules["shapely.geometry"] = shapely_geometry

from utils.nusc_utils import filter_nuscenes_sample_results_by_score


class NuScenesResultFilteringTest(unittest.TestCase):
    def test_filters_results_below_per_class_minimum(self):
        sample_results = [
            {"tracking_name": "bicycle", "tracking_score": 0.54, "tracking_id": "1"},
            {"tracking_name": "bicycle", "tracking_score": 0.55, "tracking_id": "2"},
            {"tracking_name": "bus", "tracking_score": 0.12, "tracking_id": "3"},
            {"tracking_name": "car", "tracking_score": 0.29, "tracking_id": "4"},
            {"tracking_name": "car", "tracking_score": 0.30, "tracking_id": "5"},
            {"tracking_name": "truck", "tracking_score": 0.45, "tracking_id": "6"},
            {"tracking_name": "truck", "tracking_score": 0.44, "tracking_id": "7"},
        ]
        score_cfg = {
            0: 0.55,  # bicycle
            1: 0.0,   # bus
            2: 0.30,  # car
            6: 0.45,  # truck
        }
        category_map = {
            "bicycle": 0,
            "bus": 1,
            "car": 2,
            "truck": 6,
        }

        filtered = filter_nuscenes_sample_results_by_score(
            sample_results,
            score_cfg=score_cfg,
            category_map=category_map,
        )

        self.assertEqual(
            [item["tracking_id"] for item in filtered],
            ["2", "3", "5", "6"],
        )

    def test_keeps_unknown_class_when_no_threshold_exists(self):
        sample_results = [
            {"tracking_name": "unknown", "tracking_score": 0.01, "tracking_id": "9"}
        ]

        filtered = filter_nuscenes_sample_results_by_score(
            sample_results,
            score_cfg={},
            category_map={},
        )

        self.assertEqual(filtered, sample_results)


if __name__ == "__main__":
    unittest.main()
