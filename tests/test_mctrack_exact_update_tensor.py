import unittest

import torch

from kalmanfilter.mctrack_exact_update import restore_exact_matched_components


class MCTrackExactUpdateTensorTest(unittest.TestCase):
    def test_only_selected_track_restores_z_vz_height_and_covariance(self):
        predicted_pos_x = torch.arange(12.0).view(2, 6, 1)
        predicted_pos_P = torch.arange(72.0).view(2, 6, 6)
        predicted_siz_x = torch.arange(6.0).view(2, 3, 1)
        predicted_siz_P = torch.arange(18.0).view(2, 3, 3)
        updated_pos_x = predicted_pos_x + 100.0
        updated_pos_P = predicted_pos_P + 100.0
        updated_siz_x = predicted_siz_x + 100.0
        updated_siz_P = predicted_siz_P + 100.0

        pos_x, pos_P, siz_x, siz_P = restore_exact_matched_components(
            predicted_pos_x, predicted_pos_P, predicted_siz_x, predicted_siz_P,
            updated_pos_x, updated_pos_P, updated_siz_x, updated_siz_P,
            torch.tensor([True, False]),
        )

        for index in (2, 5):
            self.assertTrue(torch.equal(pos_x[0, index], predicted_pos_x[0, index]))
            self.assertTrue(torch.equal(pos_P[0, index], predicted_pos_P[0, index]))
            self.assertTrue(torch.equal(pos_P[0, :, index], predicted_pos_P[0, :, index]))
        self.assertTrue(torch.equal(siz_x[0, 2], predicted_siz_x[0, 2]))
        self.assertTrue(torch.equal(siz_P[0, 2], predicted_siz_P[0, 2]))
        self.assertTrue(torch.equal(siz_P[0, :, 2], predicted_siz_P[0, :, 2]))
        self.assertTrue(torch.equal(pos_x[1], updated_pos_x[1]))
        self.assertTrue(torch.equal(siz_x[1], updated_siz_x[1]))


if __name__ == "__main__":
    unittest.main()
