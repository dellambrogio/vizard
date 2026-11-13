"""Evaluator for CIFAR-10: classification accuracy, in total and per class."""

import torch
from torch import nn

from projects.cifar10.data.io import CLASSES
from projects.cifar10.data.loaders import Batch
from vizard.evaluation.evaluator import BaseEvaluator

NUM_CLASSES: int = len(CLASSES)


class Cifar10Evaluator(BaseEvaluator[Batch]):
    """Report the fraction of images with the correct predicted class.

    Metrics:
    - accuracy: over all images.
    - accuracy_<class> for each class name (airplane, automobile, ...): over the images of
      that class.
    """

    correct: torch.Tensor = torch.zeros(NUM_CLASSES, dtype=torch.int64)  # per true class
    seen: torch.Tensor = torch.zeros(NUM_CLASSES, dtype=torch.int64)  # per true class

    def predict(self, module: nn.Module, data: Batch) -> torch.Tensor:
        return module(data[0])

    def reset(self) -> None:
        self.correct = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.seen = torch.zeros(NUM_CLASSES, dtype=torch.int64)

    def update(self, data: Batch, outputs: torch.Tensor) -> None:
        y: torch.Tensor = data[1].cpu()
        predicted: torch.Tensor = outputs.argmax(1).cpu()
        self.correct += torch.bincount(y[predicted == y], minlength=NUM_CLASSES)
        self.seen += torch.bincount(y, minlength=NUM_CLASSES)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "accuracy": self.correct.sum().item() / max(int(self.seen.sum().item()), 1)
        }
        for index, name in enumerate(CLASSES):
            seen: int = int(self.seen[index].item())
            metrics[f"accuracy_{name}"] = self.correct[index].item() / max(seen, 1)
        return metrics
