"""Focused tests for the Phase 2 reference CNN."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.reference_cnn import ReferenceCNN, _train_epoch


class ReferenceCNNTest(unittest.TestCase):
    def test_architecture_has_ten_convolutions_and_two_logits(self) -> None:
        model = ReferenceCNN(channels=(2,) * 10)
        convolution_count = sum(isinstance(layer, nn.Conv2d) for layer in model.modules())
        self.assertEqual(convolution_count, 10)
        output = model(torch.zeros(2, 1, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 2))

    def test_rejects_non_grayscale_input(self) -> None:
        model = ReferenceCNN(channels=(2,) * 10)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.zeros(2, 3, 32, 32))

    def test_training_epoch_returns_finite_mean_loss(self) -> None:
        torch.manual_seed(3)
        model = ReferenceCNN(channels=(2,) * 10)
        images = torch.randn(4, 1, 32, 32)
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        loss = _train_epoch(
            model,
            loader,
            torch.optim.Adam(model.parameters(), lr=1e-4),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertGreater(loss, 0)


if __name__ == "__main__":
    unittest.main()
