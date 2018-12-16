# pylint: disable=unused-import
from typing import List, Optional
# pylint: enable=unused-import
import time

from neuralmonkey.logging import log, notice


class TrainingProfiler:
    """Training profiler class.

    This class is used for measuring inter-validation and validation times
    during training. It stores the training profile in the `inter_val_times`
    and `validation_times` lists. These can be accessed by the toolkit to
    provide the user with insight about the training and validation time ratio.

    Additionally, this class provides getters for last logging and validation
    times, which can be used for deciding whether to log training progress
    or validate the model.
    """

    def __init__(self) -> None:
        self._start_time = None  # type: Optional[float]
        self._epoch_starts = []  # type: List[float]

        self._last_val_time = None  # type: Optional[float]
        self._last_log_time = None  # type: Optional[float]
        self._current_validation_start = None  # type: Optional[float]

        self.inter_val_times = []  # type: List[float]
        self.validation_times = []  # type: List[float]

        self.time = time.process_time

    @property
    def start_time(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Training did not start yet")
        return self._start_time

    @property
    def last_log_time(self) -> float:
        if self._last_log_time is None:
            return self.start_time
        return self._last_log_time

    @property
    def last_val_time(self) -> float:
        if self._last_val_time is None:
            return self.start_time
        return self._last_val_time

    def training_start(self) -> None:
        self._start_time = self.time()

    def epoch_start(self) -> None:
        self._epoch_starts.append(self.time())

    def log_done(self) -> None:
        self._last_log_time = self.time()

    def validation_start(self) -> None:
        assert self._current_validation_start is None
        self._current_validation_start = self.time()
        self.inter_val_times.append(
            self._current_validation_start - self.last_val_time)

    def validation_done(self) -> None:
        assert self._current_validation_start is not None
        self._last_val_time = self.time()

        self.validation_times.append(
            self.last_val_time - self._current_validation_start)

        self._current_validation_start = None

    def log_after_validation(
            self, val_examples: int, train_examples: int) -> None:

        train_duration = self.inter_val_times[-1]
        val_duration = self.validation_times[-1]

        train_speed = train_examples / train_duration
        val_speed = val_examples / val_duration

        log("Validation time: {:.2f}s ({:.1f} instances/sec), "
            "inter-validation: {:.2f}s, ({:.1f} instances/sec)"
            .format(val_duration, val_speed, train_duration, train_speed),
            color="blue")

        if self.inter_val_times[-1] < 2 * self.validation_times[-1]:
            notice("Validation period setting is inefficient.")
