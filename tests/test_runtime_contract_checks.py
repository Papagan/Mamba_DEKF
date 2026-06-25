import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_functions(module_path, function_names):
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = {}
    loaded = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            fn_source = ast.get_source_segment(source, node)
            exec(fn_source, namespace)
            loaded[node.name] = namespace[node.name]

    missing = [name for name in function_names if name not in loaded]
    if missing:
        raise AssertionError(f"Missing functions in {module_path}: {missing}")
    return loaded


class _FakeTorch:
    device = object

    @staticmethod
    def no_grad():
        def _decorator(fn):
            return fn
        return _decorator


class RuntimeContractChecksTest(unittest.TestCase):
    def test_runtime_contract_warns_only_on_explicit_runtime_history_init_signals(self):
        helpers = _load_functions(
            REPO_ROOT / "tracker" / "base_tracker.py",
            [
                "build_runtime_contract_warnings",
            ],
        )
        build_runtime_contract_warnings = helpers["build_runtime_contract_warnings"]

        runtime_contract = {
            "tracker_compat_mode": "mctrack",
            "expected_bev_cost_mode": "geometric",
            "history_source": "fusion",
            "init_state_source": "fusion",
            "filter_mode": "mamba",
        }

        warnings = build_runtime_contract_warnings(
            runtime_contract=runtime_contract,
            tracker_compat_mode="default",
            filter_mode="pure_dekf",
            current_cost_mode="geometric",
            current_history_source=None,
            current_init_state_source=None,
        )

        self.assertTrue(any("TRACKER_COMPAT_MODE=mctrack" in warning for warning in warnings))
        self.assertTrue(any("FILTER_MODE=mamba" in warning for warning in warnings))
        self.assertFalse(any("history_source=fusion" in warning for warning in warnings))
        self.assertFalse(any("init_state_source=fusion" in warning for warning in warnings))

        explicit_warnings = build_runtime_contract_warnings(
            runtime_contract=runtime_contract,
            tracker_compat_mode="mctrack",
            filter_mode="mamba",
            current_cost_mode="geometric",
            current_history_source="det",
            current_init_state_source="det",
        )
        self.assertTrue(any("history_source=fusion" in warning for warning in explicit_warnings))
        self.assertTrue(any("init_state_source=fusion" in warning for warning in explicit_warnings))

    def test_training_runtime_contract_filter_mode_never_persists_invalid_mode(self):
        helpers = _load_functions(
            REPO_ROOT / "training" / "train.py",
            ["resolve_runtime_contract_filter_mode"],
        )
        resolve_runtime_contract_filter_mode = helpers["resolve_runtime_contract_filter_mode"]

        self.assertEqual(
            resolve_runtime_contract_filter_mode(
                cfg={"FILTER_MODE": "pure_dekf"},
                train_tracker_compat_mode="mctrack",
            ),
            "pure_dekf",
        )
        self.assertEqual(
            resolve_runtime_contract_filter_mode(
                cfg={"FILTER_MODE": "mamba_multihead_closure"},
                train_tracker_compat_mode="mctrack",
            ),
            "mamba_multihead_closure",
        )
        self.assertEqual(
            resolve_runtime_contract_filter_mode(
                cfg={},
                train_tracker_compat_mode="mctrack",
            ),
            "mamba",
        )
        self.assertEqual(
            resolve_runtime_contract_filter_mode(
                cfg={"FILTER_MODE": "default"},
                train_tracker_compat_mode="fusion",
            ),
            "fusion",
        )

    def test_validate_forwards_filter_mode_into_training_step(self):
        source = (REPO_ROOT / "training" / "train.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(REPO_ROOT / "training" / "train.py"))
        namespace = {
            "torch": _FakeTorch(),
            "DataLoader": object,
            "TemporalMamba": object,
            "JointLoss": object,
        }
        seen = {}

        def training_step(*args, **kwargs):
            seen["filter_mode"] = kwargs.get("filter_mode")
            return None, {"loss_total": 1.0}

        namespace["training_step"] = training_step
        validate = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "validate":
                fn_source = ast.get_source_segment(source, node)
                exec(fn_source, namespace)
                validate = namespace["validate"]
                break

        self.assertIsNotNone(validate)

        class _DummyMamba:
            def eval(self):
                return None

            def train(self):
                return None

        result = validate(
            _DummyMamba(),
            val_loader=[{"dummy": 1}],
            loss_fn=None,
            device=None,
            filter_mode="mamba_multihead_closure",
        )

        self.assertEqual(seen["filter_mode"], "mamba_multihead_closure")
        self.assertEqual(result["loss_total"], 1.0)


if __name__ == "__main__":
    unittest.main()
