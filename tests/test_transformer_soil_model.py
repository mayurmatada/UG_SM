import unittest

import torch

from transformer_soil_model import TransformerSoilNet


class TransformerSoilNetTests(unittest.TestCase):
    def test_forward_preserves_training_contract(self) -> None:
        model = TransformerSoilNet(
            in_channels=19,
            dynamic_idx=list(range(9)),
            static_idx=list(range(13, 19)),
            seq_input_size=4,
            spatial_layers=1,
            temporal_layers=1,
        )
        output = model(
            torch.randn(3, 19, 13, 13),
            torch.randn(3, 10, 4),
        )
        self.assertEqual(output.shape, (3, 1))

    def test_hybrid_contains_original_cnn_and_lstm_branches(self) -> None:
        model = TransformerSoilNet(
            in_channels=19,
            dynamic_idx=list(range(9)),
            static_idx=list(range(13, 19)),
            seq_input_size=4,
            spatial_layers=1,
            temporal_layers=1,
        )
        self.assertTrue(hasattr(model, "spatial_cnn"))
        self.assertTrue(hasattr(model, "temporal_lstm"))
        self.assertTrue(hasattr(model, "static_mlp"))

    def test_rejects_wrong_patch_size(self) -> None:
        model = TransformerSoilNet(
            in_channels=4,
            dynamic_idx=[0, 1],
            static_idx=[2, 3],
            seq_input_size=2,
            spatial_layers=1,
            temporal_layers=1,
        )
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 11, 11), torch.randn(2, 5, 2))


if __name__ == "__main__":
    unittest.main()
