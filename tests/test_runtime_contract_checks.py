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
    def test_closure_force_coast_prior_only_ast_wiring(self):
        module_path = REPO_ROOT / "kalmanfilter" / "mamba_adaptive_kf.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))

        temporal_forward = None
        predict_with_mamba = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "TemporalMamba":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "forward":
                        temporal_forward = child
            if isinstance(node, ast.ClassDef) and node.name == "MambaDecoupledEKF":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "predict_with_mamba":
                        predict_with_mamba = child

        self.assertIsNotNone(temporal_forward)
        self.assertIsNotNone(predict_with_mamba)
        self.assertIn("state_buckets", [arg.arg for arg in temporal_forward.args.args])
        self.assertIn("state_buckets", [arg.arg for arg in predict_with_mamba.args.args])

        forward_calls = [node for node in ast.walk(temporal_forward) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name) and call.func.id == "apply_force_coast_prior_only_to_ratios"
                for call in forward_calls
            )
        )

        predict_calls = [node for node in ast.walk(predict_with_mamba) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "mamba"
                and any(keyword.arg == "state_buckets" for keyword in call.keywords)
                for call in predict_calls
            )
        )

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

    def test_warns_on_closure_conditional_prior_mismatch(self):
        helpers = _load_functions(
            REPO_ROOT / "tracker" / "base_tracker.py",
            [
                "build_runtime_contract_warnings",
            ],
        )
        build_runtime_contract_warnings = helpers["build_runtime_contract_warnings"]

        warnings = build_runtime_contract_warnings(
            runtime_contract={
                "tracker_compat_mode": "mctrack",
                "filter_mode": "mamba_multihead_closure",
                "expected_bev_cost_mode": "geometric",
                "closure_use_conditional_prior": True,
                "closure_force_prior_states": ["matched"],
                "closure_active_class_states": {2: ["unmatched"]},
            },
            tracker_compat_mode="mctrack",
            filter_mode="mamba_multihead_closure",
            current_cost_mode="geometric",
            current_history_source="fusion",
            current_init_state_source="fusion",
            current_closure_cfg={
                "USE_CONDITIONAL_PRIOR": False,
                "FORCE_PRIOR_STATES": ["matched", "unmatched"],
                "ACTIVE_CLASS_STATES": {},
            },
        )
        self.assertTrue(any("closure_use_conditional_prior" in item for item in warnings))
        self.assertTrue(any("closure_force_prior_states" in item for item in warnings))
        self.assertTrue(any("closure_active_class_states" in item for item in warnings))

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

    def test_orientation_curriculum_schedule_transitions_from_state_to_wrapped(self):
        helpers = _load_functions(
            REPO_ROOT / "training" / "train.py",
            ["resolve_orientation_curriculum_weights"],
        )
        resolve_orientation_curriculum_weights = helpers["resolve_orientation_curriculum_weights"]

        early = resolve_orientation_curriculum_weights(
            epoch=1,
            closure_cfg={
                "ORI_WARMUP_EPOCHS": 4,
                "ORI_TRANSITION_EPOCHS": 4,
                "ORI_STATE_WEIGHT": 1.0,
                "ORI_WRAPPED_NLL_WEIGHT": 1.0,
            },
        )
        late = resolve_orientation_curriculum_weights(
            epoch=9,
            closure_cfg={
                "ORI_WARMUP_EPOCHS": 4,
                "ORI_TRANSITION_EPOCHS": 4,
                "ORI_STATE_WEIGHT": 1.0,
                "ORI_WRAPPED_NLL_WEIGHT": 1.0,
            },
        )

        self.assertGreater(early["state_weight"], early["wrapped_weight"])
        self.assertGreater(late["wrapped_weight"], late["state_weight"])

    def test_validate_forwards_filter_mode_into_training_step(self):
        source = (REPO_ROOT / "training" / "train.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(REPO_ROOT / "training" / "train.py"))
        namespace = {
            "torch": _FakeTorch(),
            "DataLoader": object,
            "TemporalMamba": object,
            "JointLoss": object,
            "init_class_state_metric_accumulator": lambda: {},
            "finalize_class_state_metric_accumulator": lambda acc: acc,
            "update_class_state_metric_accumulator": lambda acc, **kwargs: acc.update({"metrics": kwargs["metrics"]}),
        }
        seen = {}

        def training_step(*args, **kwargs):
            seen["filter_mode"] = kwargs.get("filter_mode")
            seen["emit_sample_metrics"] = kwargs.get("emit_sample_metrics")
            return None, {
                "loss_total": 1.0,
                "_class_ids": [2],
                "_state_buckets": ["matched"],
                "_sample_loss_real": [3.0],
            }

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
        self.assertIs(seen["emit_sample_metrics"], True)
        self.assertEqual(result["loss_total"], 1.0)
        self.assertEqual(result["class_state/metrics"], {"loss_real": [3.0]})

    def test_closure_training_step_logs_orientation_saturation_and_tensor_blend_inputs(self):
        source = (REPO_ROOT / "training" / "train.py").read_text(encoding="utf-8")

        self.assertIn("orientation_saturation_penalty(", source)
        self.assertIn('detail["loss_ori_saturation_reg"]', source)
        self.assertIn('detail_k["loss_ori_state_tensor"]', source)
        self.assertIn('detail_k["loss_ori_wrapped_tensor"]', source)

    def test_closure_inference_wires_separate_prior_conditioning_history(self):
        tracker_source = (REPO_ROOT / "tracker" / "base_tracker.py").read_text(encoding="utf-8")
        mamba_source = (REPO_ROOT / "kalmanfilter" / "mamba_adaptive_kf.py").read_text(encoding="utf-8")

        self.assertIn("prior_track_history", tracker_source)
        self.assertIn("prior_history_mask", tracker_source)
        self.assertIn("prior_history_match_mask", tracker_source)
        self.assertIn("prior_track_history", mamba_source)
        self.assertIn("conditioning_history", mamba_source)

    def test_closure_force_coast_prior_only_is_wired_through_train_and_infer(self):
        train_path = REPO_ROOT / "training" / "train.py"
        train_source = train_path.read_text(encoding="utf-8")
        train_tree = ast.parse(train_source, filename=str(train_path))

        training_step = None
        for node in train_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "training_step":
                training_step = node
                break
        self.assertIsNotNone(training_step)

        calls = [node for node in ast.walk(training_step) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name)
                and call.func.id == "mamba"
                and any(keyword.arg == "state_buckets" for keyword in call.keywords)
                for call in calls
            )
        )

    def test_closure_training_bypasses_inference_force_prior_for_learnable_matched_head(self):
        train_path = REPO_ROOT / "training" / "train.py"
        train_source = train_path.read_text(encoding="utf-8")
        train_tree = ast.parse(train_source, filename=str(train_path))

        training_step = None
        for node in train_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "training_step":
                training_step = node
                break
        self.assertIsNotNone(training_step)

        calls = [node for node in ast.walk(training_step) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name)
                and call.func.id == "mamba"
                and any(keyword.arg == "apply_force_prior" for keyword in call.keywords)
                for call in calls
            )
        )

    def test_main_reports_tracking_only_fps(self):
        source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(REPO_ROOT / "main.py"))

        run_fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_fn = node
                break

        self.assertIsNotNone(run_fn)
        self.assertIn("scene_perf_stats", [arg.arg for arg in run_fn.args.args])
        self.assertIn("time.perf_counter()", source)
        self.assertIn("tracking_fps", source)
        self.assertIn("[PERF] scene=", source)
        self.assertIn("[PERF] total_tracking_frames=", source)
        self.assertIn("scene_perf_stats = manager.dict()", source)
        self.assertIn("scene_perf_stats=scene_perf_stats", source)


if __name__ == "__main__":
    unittest.main()
