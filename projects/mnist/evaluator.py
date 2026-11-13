"""Evaluator for MNIST: classification accuracy, in total and per digit."""

import torch
from torch import nn

from projects.mnist.data.loaders import Batch
from vizard.evaluation.evaluator import BaseEvaluator

NUM_DIGITS: int = 10


class MnistEvaluator(BaseEvaluator[Batch]):
    """Report the fraction of images with the correct predicted digit.

    Metrics:
    - accuracy: over all images.
    - accuracy_<d> for d in 0..9: over the images of digit d.
    """

    correct: torch.Tensor = torch.zeros(NUM_DIGITS, dtype=torch.int64)  # per true digit
    seen: torch.Tensor = torch.zeros(NUM_DIGITS, dtype=torch.int64)  # per true digit

    def predict(self, module: nn.Module, data: Batch) -> torch.Tensor:
        return module(data[0])

    def reset(self) -> None:
        self.correct = torch.zeros(NUM_DIGITS, dtype=torch.int64)
        self.seen = torch.zeros(NUM_DIGITS, dtype=torch.int64)

    def update(self, data: Batch, outputs: torch.Tensor) -> None:
        y: torch.Tensor = data[1].cpu()
        predicted: torch.Tensor = outputs.argmax(1).cpu()
        self.correct += torch.bincount(y[predicted == y], minlength=NUM_DIGITS)
        self.seen += torch.bincount(y, minlength=NUM_DIGITS)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "accuracy": self.correct.sum().item() / max(int(self.seen.sum().item()), 1)
        }
        for digit in range(NUM_DIGITS):
            seen: int = int(self.seen[digit].item())
            metrics[f"accuracy_{digit}"] = self.correct[digit].item() / max(seen, 1)
        return metrics
