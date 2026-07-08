import unittest

import torch

from training.association_loss import association_ranking_loss
from training.association_tokens import build_future_detection_history


class AssociationSupervisionLossTest(unittest.TestCase):
    def test_positive_similarity_beats_same_class_negative(self):
        track_embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        good_det_embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        bad_det_embeddings = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        class_ids = torch.tensor([3, 3, 5], dtype=torch.long)
        match_mask = torch.tensor([True, True, True])

        good_loss, good_detail = association_ranking_loss(
            track_embeddings,
            good_det_embeddings,
            class_ids,
            match_mask,
            margin=0.2,
        )
        bad_loss, bad_detail = association_ranking_loss(
            track_embeddings,
            bad_det_embeddings,
            class_ids,
            match_mask,
            margin=0.2,
        )

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))
        self.assertEqual(good_detail["association_valid_anchors"], 2)
        self.assertEqual(bad_detail["association_valid_anchors"], 2)

    def test_returns_zero_without_same_class_negatives(self):
        embeddings = torch.eye(3, dtype=torch.float32)
        loss, detail = association_ranking_loss(
            embeddings,
            embeddings,
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([True, True, True]),
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(detail["association_valid_anchors"], 0)

    def test_respects_match_mask(self):
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        loss, detail = association_ranking_loss(
            embeddings,
            embeddings,
            torch.tensor([3, 3], dtype=torch.long),
            torch.tensor([True, False]),
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(detail["association_valid_anchors"], 0)

    def test_future_detection_history_uses_detector_score_token(self):
        future = build_future_detection_history(
            obs_future_pos=torch.tensor([[[1.0, 2.0, 0.5, 0.1, 0.2]]], dtype=torch.float32),
            obs_future_siz=torch.tensor([[[4.0, 1.8, 1.5]]], dtype=torch.float32),
            obs_future_ori=torch.tensor([[[0.3]]], dtype=torch.float32),
            obs_future_score=torch.tensor([[0.73]], dtype=torch.float32),
            obs_future_match=torch.tensor([[True]]),
            class_ids=torch.tensor([0], dtype=torch.long),
            history_len=4,
            step_index=0,
        )

        self.assertAlmostEqual(float(future["history"][0, -1, 11].item()), 0.73, places=5)
        self.assertEqual(future["history_mask"].tolist(), [[False, False, False, True]])
        self.assertEqual(future["history_match_mask"].tolist(), [[False, False, False, True]])


if __name__ == "__main__":
    unittest.main()
