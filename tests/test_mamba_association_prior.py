import unittest
import sys
import types

import numpy as np

if "lap" not in sys.modules:
    lap_stub = types.ModuleType("lap")

    def _unused_lapjv(*args, **kwargs):
        raise AssertionError("lapjv should not be used by association-prior unit tests")

    lap_stub.lapjv = _unused_lapjv
    sys.modules["lap"] = lap_stub

if "pyquaternion" not in sys.modules:
    pyquaternion_stub = types.ModuleType("pyquaternion")

    class _UnusedQuaternion:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Quaternion should not be used by association-prior unit tests")

    pyquaternion_stub.Quaternion = _UnusedQuaternion
    sys.modules["pyquaternion"] = pyquaternion_stub

import tracker.matching as matching
from tracker.matching import (
    apply_mamba_association_prior_to_cost_matrix,
    apply_pairwise_association_head_to_cost_matrix,
)


class _Box:
    def __init__(self, category, det_score=0.8):
        self.category = category
        self.det_score = det_score


class _Traj:
    def __init__(self, category, unmatch_length=0):
        self.bboxes = [_Box(category)]
        self.unmatch_length = unmatch_length


class MambaAssociationPriorTest(unittest.TestCase):
    def test_disabled_config_leaves_cost_matrix_unchanged(self):
        cost = np.array([[1.0, 1.5], [2.0, 0.5]], dtype=np.float32)
        out = apply_mamba_association_prior_to_cost_matrix(
            cost,
            [_Traj("car"), _Traj("trailer")],
            [_Box("car"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"car": 0, "trailer": 5},
                "MAMBA_ASSOCIATION_PRIOR": {"ENABLED": False},
            },
            trk_embeddings=np.eye(2, dtype=np.float32),
            det_embeddings=np.eye(2, dtype=np.float32),
        )

        np.testing.assert_array_equal(out, cost)
        self.assertIsNot(out, cost)

    def test_enabled_config_applies_only_active_class_state_with_bounds(self):
        cost = np.ones((2, 2), dtype=np.float32)
        trk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        det_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        cfg = {
            "CATEGORY_MAP_TO_NUMBER": {"motorcycle": 3, "trailer": 5},
            "MAMBA_ASSOCIATION_PRIOR": {
                "ENABLED": True,
                "MAX_DELTA": 0.05,
                "ACTIVE_CLASS_STATES": {
                    3: ["unmatched"],
                    5: ["matched"],
                },
                "ALPHA": {
                    3: {"unmatched": 0.20},
                    5: {"matched": 0.20},
                },
            },
        }

        out = apply_mamba_association_prior_to_cost_matrix(
            cost,
            [_Traj("motorcycle", unmatch_length=1), _Traj("trailer", unmatch_length=1)],
            [_Box("motorcycle"), _Box("motorcycle")],
            cfg,
            trk_embeddings=trk_embeddings,
            det_embeddings=det_embeddings,
        )

        # Motorcycle/unmatched is active: good embedding pair unchanged,
        # bad embedding pair is penalized but clipped to MAX_DELTA.
        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)
        # Trailer/unmatched is inactive because only matched is configured.
        self.assertAlmostEqual(float(out[1, 0]), 1.0)
        self.assertAlmostEqual(float(out[1, 1]), 1.0)

    def test_cross_class_pairs_are_not_rescored(self):
        cost = np.ones((1, 2), dtype=np.float32)
        out = apply_mamba_association_prior_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("car"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"car": 0, "trailer": 5},
                "MAMBA_ASSOCIATION_PRIOR": {
                    "ENABLED": True,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                    "ALPHA": {5: {"unmatched": 0.20}},
                },
            },
            trk_embeddings=np.array([[0.0, 1.0]], dtype=np.float32),
            det_embeddings=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)

    def test_match_trajs_applies_mamba_prior_before_assignment(self):
        seen = {}
        original_cost = matching.cost_calculate_general
        original_hungarian = matching.Hungarian

        def fake_cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
            return (
                np.array([[1.0, 1.0]], dtype=np.float32),
                np.array(["motorcycle"]),
                np.array(["motorcycle", "motorcycle"]),
            )

        def fake_hungarian(trans_cost_matrix, thresholds):
            seen["trans_cost_matrix"] = np.array(trans_cost_matrix, copy=True)
            return [0], [0], np.array([1]), np.array([], dtype=int), np.array([0.0])

        matching.cost_calculate_general = fake_cost_calculate_general
        matching.Hungarian = fake_hungarian
        try:
            matched, _ = matching.match_trajs_and_dets(
                [_Traj("motorcycle", unmatch_length=1)],
                [_Box("motorcycle"), _Box("motorcycle")],
                {
                    "CATEGORY_LIST": ["motorcycle"],
                    "CATEGORY_MAP_TO_NUMBER": {"motorcycle": 0},
                    "MATCHING": {
                        "BEV": {"MATCHING_MODE": "Hungarian"},
                        "RV": {"MATCHING_MODE": "Hungarian"},
                    },
                    "THRESHOLD": {
                        "BEV": {"COST_THRE": {0: 2.0}},
                        "RV": {"COST_THRE": {0: 2.0}},
                    },
                    "MAMBA_ASSOCIATION_PRIOR": {
                        "ENABLED": True,
                        "MAX_DELTA": 0.05,
                        "ACTIVE_CLASS_STATES": {0: ["unmatched"]},
                        "ALPHA": {0: {"unmatched": 0.20}},
                    },
                },
                trk_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
                det_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
        finally:
            matching.cost_calculate_general = original_cost
            matching.Hungarian = original_hungarian

        np.testing.assert_array_equal(matched, np.array([[0, 0]]))
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 1, 0]), 1.05, places=6)

    def test_pairwise_head_disabled_leaves_cost_unchanged(self):
        cost = np.array([[1.0, 1.5]], dtype=np.float32)
        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("trailer"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {"ENABLED": False},
            },
            association_scores=np.array([[0.1, 0.9]], dtype=np.float32),
        )

        np.testing.assert_array_equal(out, cost)
        self.assertIsNot(out, cost)

    def test_pairwise_head_only_increases_low_score_active_pairs(self):
        cost = np.ones((2, 2), dtype=np.float32)
        scores = np.array([[0.9, 0.2], [0.1, 0.1]], dtype=np.float32)
        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("motorcycle", unmatch_length=1), _Traj("trailer", unmatch_length=1)],
            [_Box("motorcycle"), _Box("motorcycle")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"motorcycle": 3, "trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {
                    "ENABLED": True,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.2,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {3: ["unmatched"]},
                },
            },
            association_scores=scores,
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)
        self.assertAlmostEqual(float(out[1, 0]), 1.0)
        self.assertAlmostEqual(float(out[1, 1]), 1.0)

    def test_pairwise_head_reports_audit_records(self):
        records = []
        cost = np.ones((1, 2), dtype=np.float32)

        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("trailer"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {
                    "ENABLED": True,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.2,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                },
            },
            association_scores=np.array([[0.9, 0.1]], dtype=np.float32),
            audit_callback=records.append,
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["class_name"], "trailer")
        self.assertEqual(records[0]["state_bucket"], "unmatched")
        self.assertAlmostEqual(records[0]["score"], 0.9, places=6)
        self.assertAlmostEqual(records[0]["delta"], 0.0, places=6)
        self.assertAlmostEqual(records[1]["score"], 0.1, places=6)
        self.assertAlmostEqual(records[1]["delta"], 0.05, places=6)

    def test_pairwise_head_margin_tiebreak_only_penalizes_near_best_candidates(self):
        cost = np.array([[1.00, 1.03, 1.30]], dtype=np.float32)
        scores = np.array([[0.9, 0.1, 0.1]], dtype=np.float32)
        records = []

        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("trailer"), _Box("trailer"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {
                    "ENABLED": True,
                    "APPLY_MODE": "margin_tiebreak",
                    "COST_MARGIN_EPS": 0.05,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.2,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                },
            },
            association_scores=scores,
            audit_callback=records.append,
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.00, places=6)
        self.assertAlmostEqual(float(out[0, 1]), 1.08, places=6)
        self.assertAlmostEqual(float(out[0, 2]), 1.30, places=6)
        self.assertEqual(len(records), 3)
        self.assertTrue(records[0]["active"])
        self.assertTrue(records[1]["active"])
        self.assertFalse(records[2]["active"])
        self.assertAlmostEqual(records[2]["delta"], 0.0, places=6)

    def test_match_trajs_applies_pairwise_head_scores_before_assignment(self):
        seen = {}
        original_cost = matching.cost_calculate_general
        original_hungarian = matching.Hungarian

        def fake_cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
            return (
                np.array([[1.0, 1.0]], dtype=np.float32),
                np.array(["trailer"]),
                np.array(["trailer", "trailer"]),
            )

        def fake_hungarian(trans_cost_matrix, thresholds):
            seen["trans_cost_matrix"] = np.array(trans_cost_matrix, copy=True)
            return [0], [0], np.array([1]), np.array([], dtype=int), np.array([0.0])

        matching.cost_calculate_general = fake_cost_calculate_general
        matching.Hungarian = fake_hungarian
        try:
            matched, _ = matching.match_trajs_and_dets(
                [_Traj("trailer", unmatch_length=1)],
                [_Box("trailer"), _Box("trailer")],
                {
                    "CATEGORY_LIST": ["trailer"],
                    "CATEGORY_MAP_TO_NUMBER": {"trailer": 0},
                    "MATCHING": {
                        "BEV": {"MATCHING_MODE": "Hungarian"},
                        "RV": {"MATCHING_MODE": "Hungarian"},
                    },
                    "THRESHOLD": {
                        "BEV": {"COST_THRE": {0: 2.0}},
                        "RV": {"COST_THRE": {0: 2.0}},
                    },
                    "MAMBA_ASSOCIATION_HEAD": {
                        "ENABLED": True,
                        "MIN_SCORE": 0.6,
                        "ALPHA": 0.2,
                        "MAX_DELTA": 0.05,
                        "ACTIVE_CLASS_STATES": {0: ["unmatched"]},
                    },
                },
                association_scores=np.array([[0.9, 0.1]], dtype=np.float32),
            )
        finally:
            matching.cost_calculate_general = original_cost
            matching.Hungarian = original_hungarian

        np.testing.assert_array_equal(matched, np.array([[0, 0]]))
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 1, 0]), 1.05, places=6)


if __name__ == "__main__":
    unittest.main()
