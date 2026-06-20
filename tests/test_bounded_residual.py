import builtins
import math
import unittest

import kalmanfilter.bounded_residual as bounded_residual_module
from kalmanfilter.bounded_residual import (
    STATE_MATCHED,
    STATE_UNMATCHED,
    PROFILE_STABLE_LARGE,
    apply_bounded_residuals,
    infer_state_bucket,
    map_class_to_profile,
    clamp_ratio_value,
    get_family_ratio_bounds,
)


class _FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def view(self, *shape):
        return self

    def __truediv__(self, other):
        other_vals = _as_list(other, len(self.values))
        return _FakeVector([lhs / rhs for lhs, rhs in zip(self.values, other_vals)])

    def __rtruediv__(self, other):
        other_vals = _as_list(other, len(self.values))
        return _FakeVector([lhs / rhs for lhs, rhs in zip(other_vals, self.values)])

    def __gt__(self, other):
        other_vals = _as_list(other, len(self.values))
        return _FakeMask([lhs > rhs for lhs, rhs in zip(self.values, other_vals)])


class _FakeMask:
    def __init__(self, values):
        self.values = [bool(value) for value in values]

    def __and__(self, other):
        return _FakeMask([lhs and rhs for lhs, rhs in zip(self.values, other.values)])

    def view(self, *shape):
        return self


class _FakeCovariance:
    def __init__(self, trace_values):
        self.trace_values = [float(value) for value in trace_values]
        self.device = "cpu"
        self.dtype = "float32"

    def diagonal(self, dim1=-2, dim2=-1):
        return _FakeDiagonal(self.trace_values)

    def __mul__(self, other):
        other_vals = _as_list(other, len(self.trace_values))
        return _FakeCovariance(
            [lhs * rhs for lhs, rhs in zip(self.trace_values, other_vals)]
        )

    def __rmul__(self, other):
        return self.__mul__(other)


class _FakeDiagonal:
    def __init__(self, trace_values):
        self.trace_values = list(trace_values)

    def sum(self, dim):
        return _FakeVector(self.trace_values)


def _as_list(value, length):
    if isinstance(value, _FakeVector):
        return list(value.values)
    if isinstance(value, _FakeMask):
        return [1.0 if item else 0.0 for item in value.values]
    return [float(value)] * length


def _nan_sensitive_min(lhs, rhs):
    if math.isnan(lhs) or math.isnan(rhs):
        return float("nan")
    return builtins.min(lhs, rhs)


def _nan_sensitive_max(lhs, rhs):
    if math.isnan(lhs) or math.isnan(rhs):
        return float("nan")
    return builtins.max(lhs, rhs)


class _FakeTorch:
    @staticmethod
    def tensor(values, device=None, dtype=None):
        if isinstance(values, (list, tuple)):
            return _FakeVector(values)
        return float(values)

    @staticmethod
    def clamp(values, min=None, max=None):
        vals = _as_list(values, len(values.values) if isinstance(values, _FakeVector) else 1)
        min_vals = _as_list(min, len(vals)) if min is not None else [None] * len(vals)
        max_vals = _as_list(max, len(vals)) if max is not None else [None] * len(vals)
        out = []
        for value, min_value, max_value in zip(vals, min_vals, max_vals):
            if math.isnan(value):
                out.append(float("nan"))
                continue
            if min_value is not None:
                value = builtins.max(min_value, value)
            if max_value is not None:
                value = builtins.min(max_value, value)
            out.append(value)
        return _FakeVector(out) if len(out) > 1 else out[0]

    @staticmethod
    def maximum(lhs, rhs):
        lhs_vals = _as_list(lhs, len(lhs.values))
        rhs_vals = _as_list(rhs, len(lhs.values))
        return _FakeVector([_nan_sensitive_max(a, b) for a, b in zip(lhs_vals, rhs_vals)])

    @staticmethod
    def minimum(lhs, rhs):
        lhs_vals = _as_list(lhs, len(lhs.values))
        rhs_vals = _as_list(rhs, len(lhs.values))
        return _FakeVector([_nan_sensitive_min(a, b) for a, b in zip(lhs_vals, rhs_vals)])

    @staticmethod
    def isfinite(values):
        vals = _as_list(values, len(values.values))
        return _FakeMask([math.isfinite(value) for value in vals])

    @staticmethod
    def abs(values):
        vals = _as_list(values, len(values.values))
        return _FakeVector([abs(value) for value in vals])

    @staticmethod
    def where(mask, when_true, when_false):
        if isinstance(when_true, _FakeCovariance):
            return _FakeCovariance([
                true_value if choose else false_value
                for choose, true_value, false_value in zip(
                    mask.values,
                    when_true.trace_values,
                    when_false.trace_values,
                )
            ])
        true_vals = _as_list(when_true, len(mask.values))
        false_vals = _as_list(when_false, len(mask.values))
        return _FakeVector([
            true_value if choose else false_value
            for choose, true_value, false_value in zip(mask.values, true_vals, false_vals)
        ])

    @staticmethod
    def ones_like(values):
        return _FakeVector([1.0] * len(values.values))


class BoundedResidualHelperTest(unittest.TestCase):
    def setUp(self):
        self._orig_torch = bounded_residual_module.torch
        bounded_residual_module.torch = _FakeTorch

    def tearDown(self):
        bounded_residual_module.torch = self._orig_torch

    def test_map_class_to_profile(self):
        self.assertEqual(map_class_to_profile(0), PROFILE_STABLE_LARGE)
        self.assertEqual(map_class_to_profile(4), PROFILE_STABLE_LARGE)
        self.assertEqual(map_class_to_profile(2), "agile_weak")
        self.assertEqual(map_class_to_profile(3), "agile_weak")
        self.assertEqual(map_class_to_profile(5), "heavy_long")
        self.assertEqual(map_class_to_profile(6), "heavy_long")
        self.assertEqual(map_class_to_profile(1), "human")
        self.assertIsNone(map_class_to_profile(99))

    def test_infer_state_bucket(self):
        self.assertEqual(infer_state_bucket(0), STATE_MATCHED)
        self.assertEqual(infer_state_bucket(2), STATE_UNMATCHED)

    def test_clamp_ratio_value(self):
        self.assertEqual(clamp_ratio_value(0.5, min_ratio=0.8, max_ratio=1.6), 0.8)
        self.assertEqual(clamp_ratio_value(1.2, min_ratio=0.8, max_ratio=1.6), 1.2)
        self.assertEqual(clamp_ratio_value(2.4, min_ratio=0.8, max_ratio=1.6), 1.6)

    def test_get_family_ratio_bounds(self):
        closure_cfg = {
            "ENABLED": True,
            "PROFILES": {
                "stable_large": {
                    "matched": {"q_pos": [0.8, 1.8]},
                    "unmatched": {"q_pos": [0.7, 1.4]},
                }
            },
        }

        self.assertEqual(
            get_family_ratio_bounds(0, STATE_MATCHED, "q_pos", closure_cfg),
            (0.8, 1.8),
        )
        self.assertEqual(
            get_family_ratio_bounds(4, STATE_UNMATCHED, "q_pos", closure_cfg),
            (0.7, 1.4),
        )
        self.assertIsNone(get_family_ratio_bounds(0, STATE_MATCHED, "r_pos", closure_cfg))
        self.assertIsNone(get_family_ratio_bounds(0, STATE_MATCHED, "q_pos", {"ENABLED": False}))

    def test_get_family_ratio_bounds_returns_none_for_missing_profile_or_state(self):
        closure_cfg = {
            "ENABLED": True,
            "PROFILES": {
                "stable_large": {
                    "matched": {"q_pos": [0.8, 1.8]},
                }
            },
        }

        self.assertIsNone(get_family_ratio_bounds(1, STATE_MATCHED, "q_pos", closure_cfg))
        self.assertIsNone(get_family_ratio_bounds(0, STATE_UNMATCHED, "q_pos", closure_cfg))
        self.assertIsNone(get_family_ratio_bounds(99, STATE_MATCHED, "q_pos", closure_cfg))

    def test_apply_bounded_residuals_dispatches_per_sample_bounds_and_clamps_valid_ratios(self):
        bounded = apply_bounded_residuals(
            raw_tensors={"Q_pos": _FakeCovariance([5.0, 0.2])},
            prior_tensors={"Q_pos": _FakeCovariance([2.0, 0.5])},
            class_ids=[0, 2],
            state_buckets=[STATE_MATCHED, STATE_UNMATCHED],
            closure_cfg={
                "ENABLED": True,
                "PROFILES": {
                    "stable_large": {
                        "matched": {"q_pos": [0.8, 1.8]},
                    },
                    "agile_weak": {
                        "unmatched": {"q_pos": [0.9, 1.3]},
                    },
                },
            },
        )

        self.assertAlmostEqual(bounded["Q_pos"].trace_values[0], 3.6)
        self.assertAlmostEqual(bounded["Q_pos"].trace_values[1], 0.45)

    def test_apply_bounded_residuals_falls_back_to_prior_scaled_when_traces_are_invalid(self):
        bounded = apply_bounded_residuals(
            raw_tensors={"Q_pos": _FakeCovariance([float("nan"), 5.0])},
            prior_tensors={"Q_pos": _FakeCovariance([2.0, 0.0])},
            class_ids=[0, 0],
            state_buckets=[STATE_MATCHED, STATE_MATCHED],
            closure_cfg={
                "ENABLED": True,
                "PROFILES": {
                    "stable_large": {
                        "matched": {"q_pos": [0.8, 1.8]},
                    },
                },
            },
        )

        self.assertEqual(bounded["Q_pos"].trace_values, [3.6, 0.0])

    def test_apply_bounded_residuals_leaves_unbounded_samples_on_raw_path(self):
        bounded = apply_bounded_residuals(
            raw_tensors={"Q_pos": _FakeCovariance([5.0, float("nan")])},
            prior_tensors={"Q_pos": _FakeCovariance([2.0, 7.0])},
            class_ids=[0, 99],
            state_buckets=[STATE_MATCHED, STATE_MATCHED],
            closure_cfg={
                "ENABLED": True,
                "PROFILES": {
                    "stable_large": {
                        "matched": {"q_pos": [0.8, 1.8]},
                    },
                },
            },
        )

        self.assertAlmostEqual(bounded["Q_pos"].trace_values[0], 3.6)
        self.assertTrue(math.isnan(bounded["Q_pos"].trace_values[1]))
