from typing import Any, List

from neuralmonkey.runners.base_runner import Executable
from neuralmonkey.trainers.generic_trainer import GenericTrainer


# pylint: disable=too-few-public-methods
class MultitaskTrainer:
    """Wrapper for scheduling multitask training.

    The wrapper contains a list of trainer objects. They are being
    called in the order defined by this list thus simulating a task
    switching schedule.
    """

    def __init__(self,
                 trainers: List[GenericTrainer]) -> None:
        assert trainers is not None

        self.trainers = trainers
        self.trainer_idx = len(trainers)

        self.var_list = list(set().union(*[t.var_list for t in trainers])) \
            # type: List[Any]
        self.all_coders = set.union(*[t.all_coders for t in self.trainers])

    def get_executable(
            self, compute_losses=True, summaries=True,
            num_sessions=1) -> Executable:

        self.trainer_idx = (self.trainer_idx + 1) % len(self.trainers)

        return self.trainers[self.trainer_idx].get_executable(
            compute_losses, summaries, num_sessions)
