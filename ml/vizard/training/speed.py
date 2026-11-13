"""Train speed: steps per second and the time spent waiting for data."""

from dataclasses import dataclass


@dataclass
class TrainSpeed:
    """Train time of a number of train steps. Evaluation time is not in it."""

    seconds: float = 0.0
    steps: int = 0
    data_wait_seconds: float = 0.0  # time spent waiting for the next batch

    def add(self, seconds: float, data_wait_seconds: float) -> None:
        self.seconds += seconds
        self.steps += 1
        self.data_wait_seconds += data_wait_seconds

    @property
    def steps_per_sec(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    @property
    def data_wait_frac(self) -> float:
        return min(self.data_wait_seconds / self.seconds, 1.0) if self.seconds > 0 else 0.0
