import ast
from pathlib import Path
import unittest

import numpy as np

from tools.check_numpy_environment import classify_result


ROOT = Path(__file__).resolve().parents[1]


class NumpyCompatibilityTest(unittest.TestCase):
    def test_abi_warning_is_failure_even_when_import_exits_zero(self):
        self.assertEqual(classify_result(0, "", "A module that was compiled using NumPy 1.x"),
                         "numpy_abi_failure")
        self.assertEqual(classify_result(1, "", "RuntimeError: Numpy is not available"),
                         "numpy_abi_failure")
        self.assertEqual(classify_result(0, "", ""), "ok")
        self.assertEqual(classify_result(1, "", "ModuleNotFoundError: torch"), "failed")

    def test_camera_transform_preserves_matrix_multiplication_in_both_trackers(self):
        transform = np.array([[0., -1., 0., 10.], [1., 0., 0., 20.],
                              [0., 0., 1., 30.], [0., 0., 0., 1.]])
        bbox = np.array([1., 2., 3., 4., 5., 6., .4])
        for relative in ("tracker/bbox.py", "MCtracker/bbox.py"):
            # Extract just this method: quaternion SDKs are not needed to test its math.
            tree = ast.parse((ROOT / relative).read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BBox")
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                          and node.name == "transform_bbox_lidar2camera")
            namespace = {"np": np}
            exec(compile(ast.Module(body=[method], type_ignores=[]), relative, "exec"), namespace)
            result = namespace[method.name](None, bbox, transform)
            np.testing.assert_allclose(result, [8., 21., 33., 4., 5., 6., .4])

    def test_hota_empty_sequence_uses_supported_float_dtype(self):
        path = ROOT / "evaluation/static_evaluation/kitti/evaluation_HOTA/trackeval/metrics/hota.py"
        cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("__init__", "eval_sequence")]
        for method in methods:
            method.decorator_list = []
        isolated = ast.ClassDef(name="HOTA", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
        namespace = {"np": np}
        exec(compile(module, str(path), "exec"), namespace)
        result = namespace["HOTA"]().eval_sequence({"num_tracker_dets": 0, "num_gt_dets": 2})
        self.assertEqual(result["HOTA_FN"].dtype, np.dtype("float64"))
        np.testing.assert_array_equal(result["HOTA_FN"], np.full(19, 2.))

    def test_removed_numpy_aliases_are_not_used_in_owned_python_sources(self):
        removed = {"mat", "float_", "complex_", "float", "int", "object", "str", "asfarray"}
        for folder in ("tracker", "MCtracker", "preprocess", "evaluation"):
            for path in (ROOT / folder).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                        if node.value.id in {"np", "numpy"}:
                            self.assertNotIn(node.attr, removed, f"{path}:{node.lineno}")
