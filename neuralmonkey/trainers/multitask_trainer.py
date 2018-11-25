from typing import List, Dict

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import GraphExecutor
from neuralmonkey.trainers.generic_trainer import GenericTrainer


# pylint: disable=too-few-public-methods
class MultitaskTrainer(GraphExecutor):
    """Wrapper for scheduling multitask training.

    The wrapper contains a list of trainer objects. They are being
    called in the order defined by this list thus simulating a task
    switching schedule.
    """

    def __init__(self,
                 trainers: List[GenericTrainer]) -> None:
        check_argument_types()
        GraphExecutor.__init__(self, set(trainers))

        self.trainers = trainers
        self.trainer_idx = 0

        self.var_list = list(set.union(*[set(t.var_list) for t in trainers]))

    def get_executable(
            self, compute_losses: bool = True, summaries: bool = True,
            num_sessions: int = 1) -> GraphExecutor.Executable:

        focused_trainer = self.trainers[self.trainer_idx]
        self.trainer_idx = (self.trainer_idx + 1) % len(self.trainers)

        return focused_trainer.get_executable(
            compute_losses, summaries, num_sessions)

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        fetches = {}
        for trainer in self.trainers:
            fetches.update(trainer.fetches)
        return fetches
