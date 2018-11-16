from typing import List, Set

from typeguard import check_argument_types

from neuralmonkey.model.feedable import Feedable
from neuralmonkey.model.parameterized import Parameterized
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
        check_argument_types()

        self.trainers = trainers
        self.trainer_idx = 0

        self.var_list = list(set.union(*[set(t.var_list) for t in trainers]))

        self.feedables = set()  # type: Set[Feedable]
        self.parameterizeds = set()  # type: Set[Parameterized]

        for trainer in self.trainers:
            self.feedables |= trainer.feedables
            self.parameterizeds |= trainer.parameterizeds

    def get_executable(
            self, compute_losses: bool = True, summaries: bool = True,
            num_sessions: int = 1) -> Executable:

        focused_trainer = self.trainers[self.trainer_idx]
        self.trainer_idx = (self.trainer_idx + 1) % len(self.trainers)

        return focused_trainer.get_executable(
            compute_losses, summaries, num_sessions)
