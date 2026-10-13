import unittest

import yaml

from kalmanfilter.mctrack_exact_update import validate_exact_update_contract


class MCTrackExactUpdateContractTest(unittest.TestCase):
    def test_current_training_and_inference_configs_agree(self):
        with open("config/train_nuscenes.yaml", encoding="utf-8") as file:
            train = yaml.safe_load(file)
        with open("config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml", encoding="utf-8") as file:
            inference = yaml.safe_load(file)
        validate_exact_update_contract(train, inference)

    def test_mismatched_exact_classes_fail_before_training(self):
        train = {
            "BASE_NOISE": {
                "DETECTION_UPDATE": {"ENABLED": True},
                "MCTRACK_EXACT_UPDATE": {
                    "MATCHED_CLASSES": [5, 6],
                    "UNMATCHED_CLASSES": [5, 6],
                },
            },
        }
        inference = {
            "TRACKER_COMPAT_MODE": "mctrack",
            "MCTRACK_EXACT_MATCHED_UPDATE": True,
            "MCTRACK_EXACT_MATCHED_UPDATE_CLASSES": [5],
            "MCTRACK_EXACT_UNMATCH_UPDATE": True,
            "MCTRACK_EXACT_UNMATCH_UPDATE_CLASSES": [5, 6],
        }
        with self.assertRaisesRegex(ValueError, "MATCHED_CLASSES"):
            validate_exact_update_contract(train, inference)


if __name__ == "__main__":
    unittest.main()
