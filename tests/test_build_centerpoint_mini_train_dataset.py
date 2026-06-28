import unittest

from tools.build_centerpoint_mini_train_dataset import (
    DEFAULT_CLASS_DIST_TH,
    parse_class_dist_thresholds,
    resolve_class_dist_thresholds,
    resolve_split_scene_names,
)


class BuildCenterPointMiniTrainDatasetTest(unittest.TestCase):
    def test_resolve_split_scene_names_selects_requested_split(self):
        splits = {
            "train": ["scene-0001", "scene-0002"],
            "val": ["scene-0101"],
        }

        self.assertEqual(
            resolve_split_scene_names("train", splits=splits),
            {"scene-0001", "scene-0002"},
        )
        self.assertEqual(
            resolve_split_scene_names("val", splits=splits),
            {"scene-0101"},
        )

    def test_resolve_split_scene_names_rejects_unknown_split(self):
        with self.assertRaises(ValueError):
            resolve_split_scene_names("mini_train", splits={"train": []})

    def test_parse_class_dist_thresholds(self):
        parsed = parse_class_dist_thresholds("bicycle=1.2,motorcycle=1.5,trailer=2.5")

        self.assertEqual(parsed["bicycle"], 1.2)
        self.assertEqual(parsed["motorcycle"], 1.5)
        self.assertEqual(parsed["trailer"], 2.5)

    def test_resolve_class_dist_thresholds_uses_default_table_and_overrides(self):
        resolved = resolve_class_dist_thresholds(
            fallback_dist_th=2.0,
            use_default_class_dist_th=True,
            class_dist_th="bicycle=1.1,truck=2.7",
        )

        self.assertEqual(resolved["pedestrian"], DEFAULT_CLASS_DIST_TH["pedestrian"])
        self.assertEqual(resolved["bicycle"], 1.1)
        self.assertEqual(resolved["truck"], 2.7)

    def test_parse_class_dist_thresholds_rejects_unknown_class(self):
        with self.assertRaises(ValueError):
            parse_class_dist_thresholds("animal=1.0")


if __name__ == "__main__":
    unittest.main()
