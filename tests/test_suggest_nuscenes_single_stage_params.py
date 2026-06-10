import copy
import unittest

from tools.suggest_nuscenes_single_stage_params import (
    apply_suggestions,
    compute_diagnostics,
)


def _base_cfg():
    return {
        "CATEGORY_MAP_TO_NUMBER": {
            "car": 0,
            "pedestrian": 1,
            "bicycle": 2,
            "motorcycle": 3,
            "bus": 4,
            "trailer": 5,
            "truck": 6,
        },
        "THRESHOLD": {
            "INPUT_SCORE": {
                "ONLINE": {0: 0.18, 1: 0.24, 2: 0.20, 3: 0.20, 4: 0.28, 5: 0.24, 6: 0.18},
                "OFFLINE": {0: 0.18, 1: 0.24, 2: 0.20, 3: 0.20, 4: 0.28, 5: 0.24, 6: 0.18},
            },
            "BEV": {
                "COST_THRE": {0: 1.35, 1: 1.85, 2: 1.70, 3: 1.75, 4: 1.30, 5: 1.65, 6: 1.35},
            },
            "TRAJECTORY_THRE": {
                "CONFIRMED_DET_SCORE": {0: 0.44, 1: 0.50, 2: 0.39, 3: 0.41, 4: 0.52, 5: 0.43, 6: 0.40},
                "OUTPUT_SCORE": {0: 0.46, 1: 0.52, 2: 0.40, 3: 0.42, 4: 0.54, 5: 0.44, 6: 0.43},
                "SINGLE_STAGE_BIRTH_SCORE": {},
            },
        },
        "TRACK_SCORE": {
            "W_DET": {0: 0.34, 1: 0.34, 2: 0.42, 3: 0.42, 4: 0.34, 5: 0.36, 6: 0.34},
            "W_ASSOC": {0: 0.24, 1: 0.20, 2: 0.14, 3: 0.14, 4: 0.20, 5: 0.18, 6: 0.22},
            "W_CONT": {0: 0.24, 1: 0.24, 2: 0.14, 3: 0.14, 4: 0.22, 5: 0.18, 6: 0.20},
            "W_MATURE": {0: 0.18, 1: 0.22, 2: 0.30, 3: 0.30, 4: 0.24, 5: 0.28, 6: 0.24},
            "CURRENT_FAKE_SCALE": {0: 0.78, 1: 0.76, 2: 0.82, 3: 0.82, 4: 0.78, 5: 0.68, 6: 0.68},
            "MATURE_LEN": {0: 6, 1: 5, 2: 3, 3: 3, 4: 6, 5: 4, 6: 5},
        },
    }


def _comparison(agg_delta, weak_delta=0.0, strong_delta=0.0):
    classes = ["bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"]
    payload = {
        "aggregate": {
            "amota": {"orig": 0.70, "cal": 0.70 + agg_delta, "delta": agg_delta},
            "recall": {"orig": 0.72, "cal": 0.72 + agg_delta / 2, "delta": agg_delta / 2},
            "mota": {"orig": 0.60, "cal": 0.60 + agg_delta / 2, "delta": agg_delta / 2},
        },
        "per_class": {},
    }
    weak = {"bicycle", "motorcycle", "trailer", "truck"}
    for cls in classes:
        delta = weak_delta if cls in weak else strong_delta
        payload["per_class"][cls] = {
            "amota": {"orig": 0.6, "cal": 0.6 + delta, "delta": delta},
            "recall": {"orig": 0.7, "cal": 0.7 + delta / 2, "delta": delta / 2},
            "mota": {"orig": 0.55, "cal": 0.55 + delta / 2, "delta": delta / 2},
        }
    return payload


def _calibration(score_weight=0.8, quality_weight=0.2):
    feature_names = [
        "score_mean",
        "score_last",
        "score_std",
        "num_frames",
        "duration_sec",
        "tp_ratio",
        "purity",
        "dominant_recall",
        "straightness",
        "mean_match_dist",
        "gap_count",
    ]
    weights = [
        score_weight,
        score_weight,
        0.1,
        0.0,
        0.0,
        quality_weight,
        quality_weight,
        quality_weight,
        0.0,
        quality_weight,
        quality_weight,
    ]
    return {
        "feature_names": feature_names,
        "weights": weights,
        "preview_top20": [],
    }


class SuggestNuScenesSingleStageParamsTest(unittest.TestCase):
    def test_ranking_gain_prefers_track_score(self):
        cfg = _base_cfg()
        comparison = _comparison(agg_delta=0.02, weak_delta=0.03, strong_delta=0.0)
        calibration = _calibration(score_weight=0.7, quality_weight=0.1)
        diagnostics = compute_diagnostics(comparison, calibration)
        self.assertEqual(diagnostics["strategy"], "weak_class_track_score")

        new_cfg, report = apply_suggestions(copy.deepcopy(cfg), comparison, calibration, diagnostics)

        # weak classes changed in TRACK_SCORE
        self.assertNotEqual(new_cfg["TRACK_SCORE"]["W_DET"][2], cfg["TRACK_SCORE"]["W_DET"][2])
        self.assertNotEqual(new_cfg["TRACK_SCORE"]["W_DET"][3], cfg["TRACK_SCORE"]["W_DET"][3])
        # strong classes stay untouched
        self.assertEqual(new_cfg["TRACK_SCORE"]["W_DET"][0], cfg["TRACK_SCORE"]["W_DET"][0])
        self.assertEqual(new_cfg["TRACK_SCORE"]["W_DET"][4], cfg["TRACK_SCORE"]["W_DET"][4])
        self.assertTrue(any(item["group"] == "track_score" for item in report["changes"]))

    def test_no_gain_prefers_matching_and_birth(self):
        cfg = _base_cfg()
        comparison = _comparison(agg_delta=0.001, weak_delta=0.0, strong_delta=0.0)
        # simulate a class with good recall but weak mota => tighten birth
        comparison["per_class"]["truck"]["recall"]["orig"] = 0.80
        comparison["per_class"]["truck"]["mota"]["orig"] = 0.45
        # simulate a class with poor recall => loosen matching/lifecycle
        comparison["per_class"]["bicycle"]["recall"]["orig"] = 0.55
        comparison["per_class"]["bicycle"]["mota"]["orig"] = 0.40
        calibration = _calibration(score_weight=0.2, quality_weight=0.2)
        diagnostics = compute_diagnostics(comparison, calibration)
        self.assertEqual(diagnostics["strategy"], "matching_lifecycle")

        new_cfg, report = apply_suggestions(copy.deepcopy(cfg), comparison, calibration, diagnostics)

        self.assertLess(new_cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"][2], cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"][2])
        self.assertGreaterEqual(
            new_cfg["THRESHOLD"]["TRAJECTORY_THRE"]["SINGLE_STAGE_BIRTH_SCORE"][6],
            new_cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"][6],
        )
        self.assertTrue(any(item["group"] == "matching_lifecycle" for item in report["changes"]))


if __name__ == "__main__":
    unittest.main()
