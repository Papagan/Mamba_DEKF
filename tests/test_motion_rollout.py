import ast
import pathlib
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_train_function(name):
    module_path = REPO_ROOT / "training" / "train.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = {"torch": torch}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]
    raise AssertionError(f"Missing function: {name}")


class MotionRolloutTest(unittest.TestCase):
    def test_future_velocity_is_derived_per_step(self):
        fn = _load_train_function("_derive_future_velocities")
        current = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        future = torch.tensor([[[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        delta_t = torch.tensor([[0.5, 0.5, 0.5]])

        velocity = fn(current, future, delta_t)

        self.assertTrue(
            torch.allclose(
                velocity[0, :, 0],
                torch.tensor([1.0, 2.0, 3.0]),
            )
        )

    def test_validation_rollout_is_fully_open_loop(self):
        fn = _load_train_function("_build_open_loop_mask")
        future_mask = torch.tensor([[True, True, False], [True, True, True]])

        coast = fn(
            future_mask,
            {"ENABLED": True, "VALIDATION_OPEN_LOOP": True},
            training=False,
        )

        self.assertTrue(torch.equal(coast, future_mask))

    def test_training_coast_mask_is_contiguous_from_first_step(self):
        fn = _load_train_function("_build_open_loop_mask")
        future_mask = torch.ones(8, 6, dtype=torch.bool)
        torch.manual_seed(7)

        coast = fn(
            future_mask,
            {
                "ENABLED": True,
                "TRAIN_OPEN_LOOP_PROB": 1.0,
                "TRAIN_MIN_COAST_STEPS": 2,
                "TRAIN_MAX_COAST_STEPS": 5,
            },
            training=True,
        )

        for row in coast:
            active_steps = torch.nonzero(row, as_tuple=False).flatten().tolist()
            self.assertGreaterEqual(len(active_steps), 2)
            self.assertEqual(active_steps, list(range(len(active_steps))))


if __name__ == "__main__":
    unittest.main()
