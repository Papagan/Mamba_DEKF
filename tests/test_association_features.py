import unittest

import torch

from training.association_features import PAIR_GEOMETRY_DIM, compose_pair_vectors


class ComposePairVectorsTest(unittest.TestCase):
    def test_composes_embeddings_interactions_and_geometry_in_exact_order(self):
        track_embeddings = torch.tensor([[1.0, 2.0]])
        detection_embeddings = torch.tensor([[3.0, 5.0]])
        pair_geometry = torch.tensor([[0.4, 0.2, 1.5, 0.8, 0.9, 1.0, 5.0]])

        pair_vectors = compose_pair_vectors(
            track_embeddings,
            detection_embeddings,
            pair_geometry,
        )

        expected = torch.tensor([[
            1.0,
            2.0,
            3.0,
            5.0,
            2.0,
            3.0,
            3.0,
            10.0,
            0.4,
            0.2,
            1.5,
            0.8,
            0.9,
            1.0,
            5.0,
        ]])
        self.assertTrue(torch.allclose(pair_vectors, expected))
        self.assertEqual(PAIR_GEOMETRY_DIM, 7)

    def test_preserves_dtype_and_autograd(self):
        track_embeddings = torch.tensor(
            [[1.0, 2.0]], dtype=torch.float64, requires_grad=True
        )
        detection_embeddings = torch.tensor(
            [[3.0, 5.0]], dtype=torch.float64, requires_grad=True
        )
        pair_geometry = torch.tensor(
            [[0.4, 0.2, 1.5, 0.8, 0.9, 1.0, 5.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        pair_vectors = compose_pair_vectors(
            track_embeddings,
            detection_embeddings,
            pair_geometry,
        )
        pair_vectors.sum().backward()

        self.assertEqual(pair_vectors.dtype, torch.float64)
        self.assertTrue(pair_vectors.requires_grad)
        self.assertTrue(
            torch.equal(
                track_embeddings.grad,
                torch.tensor([[3.0, 5.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                detection_embeddings.grad,
                torch.tensor([[3.0, 4.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(torch.equal(pair_geometry.grad, torch.ones_like(pair_geometry)))

    def test_rejects_mismatched_batch_dimensions(self):
        with self.assertRaisesRegex(
            ValueError,
            "batch dimension mismatch: track_embeddings=2, detection_embeddings=1, pair_geometry=2",
        ):
            compose_pair_vectors(
                torch.zeros(2, 4),
                torch.zeros(1, 4),
                torch.zeros(2, 7),
            )

    def test_rejects_mismatched_embedding_widths(self):
        with self.assertRaisesRegex(
            ValueError,
            "embedding width mismatch: track_embeddings=4, detection_embeddings=5",
        ):
            compose_pair_vectors(
                torch.zeros(2, 4),
                torch.zeros(2, 5),
                torch.zeros(2, 7),
            )

    def test_rejects_tensors_that_are_not_rank_two(self):
        valid_embedding = torch.zeros(2, 4)
        valid_geometry = torch.zeros(2, 7)
        cases = (
            ("track_embeddings", torch.zeros(2, 1, 4), valid_embedding, valid_geometry),
            ("detection_embeddings", valid_embedding, torch.zeros(2, 1, 4), valid_geometry),
            ("pair_geometry", valid_embedding, valid_embedding, torch.zeros(2, 1, 7)),
        )

        for name, track_embeddings, detection_embeddings, pair_geometry in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f"{name} must be rank 2"):
                    compose_pair_vectors(
                        track_embeddings,
                        detection_embeddings,
                        pair_geometry,
                    )

    def test_rejects_geometry_width_other_than_seven(self):
        with self.assertRaisesRegex(
            ValueError,
            "pair_geometry width mismatch: got 6, expected 7",
        ):
            compose_pair_vectors(
                torch.zeros(2, 4),
                torch.zeros(2, 4),
                torch.zeros(2, 6),
            )


if __name__ == "__main__":
    unittest.main()
