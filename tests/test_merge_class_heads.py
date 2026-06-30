import unittest

from tools.merge_class_heads import (
    is_class_head_key_for_class,
    merge_class_head_state_dicts,
)


class MergeClassHeadsTest(unittest.TestCase):
    def test_merge_replaces_only_requested_class_head_keys(self):
        base = {
            "backbone.weight": "base-backbone",
            "head_bank.family_heads.r_pos_xyz.0.proj.weight": "base-c0",
            "head_bank.family_heads.r_pos_xyz.1.proj.weight": "base-c1",
            "head_bank.family_heads.r_ori.1.proj.bias": "base-c1-ori",
            "head_bank.family_heads.r_ori.2.proj.bias": "base-c2-ori",
            "raw_q_siz.weight": "base-shared",
        }
        class_one = {
            "backbone.weight": "bad-backbone",
            "head_bank.family_heads.r_pos_xyz.0.proj.weight": "bad-c0",
            "head_bank.family_heads.r_pos_xyz.1.proj.weight": "class1-c1",
            "head_bank.family_heads.r_ori.1.proj.bias": "class1-c1-ori",
            "raw_q_siz.weight": "bad-shared",
        }

        merged, copied = merge_class_head_state_dicts(
            base,
            {1: class_one},
        )

        self.assertEqual(merged["backbone.weight"], "base-backbone")
        self.assertEqual(merged["raw_q_siz.weight"], "base-shared")
        self.assertEqual(merged["head_bank.family_heads.r_pos_xyz.0.proj.weight"], "base-c0")
        self.assertEqual(merged["head_bank.family_heads.r_pos_xyz.1.proj.weight"], "class1-c1")
        self.assertEqual(merged["head_bank.family_heads.r_ori.1.proj.bias"], "class1-c1-ori")
        self.assertEqual(merged["head_bank.family_heads.r_ori.2.proj.bias"], "base-c2-ori")
        self.assertEqual(
            copied,
            {
                1: [
                    "head_bank.family_heads.r_ori.1.proj.bias",
                    "head_bank.family_heads.r_pos_xyz.1.proj.weight",
                ]
            },
        )

    def test_class_head_match_uses_exact_class_segment(self):
        self.assertTrue(is_class_head_key_for_class("head_bank.family_heads.r_ori.2.proj.weight", 2))
        self.assertFalse(is_class_head_key_for_class("head_bank.family_heads.r_ori.20.proj.weight", 2))
        self.assertFalse(is_class_head_key_for_class("head_bank.other.r_ori.2.proj.weight", 2))


if __name__ == "__main__":
    unittest.main()
