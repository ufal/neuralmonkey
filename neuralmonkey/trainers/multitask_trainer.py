from typing import List, Optional

import tensorflow as tf

from neuralmonkey.runners.base_runner import Executable
from neuralmonkey.trainers.generic_trainer import GenericTrainer, Objective
from neuralmonkey.trainers.regularizers import Regularizer


# pylint: disable=too-few-public-methods
class MultitaskTrainer(GenericTrainer):

    def __init__(self,
                 objectives: List[Objective] = None,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 regularizers: List[Regularizer] = None,
                 trainers: List[GenericTrainer] = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        self.trainers = trainers
        self.trainer_idx = len(self.trainers)

        # TODO: everything except trainers is ignored now,
        # find a better solution

        self.var_list = list(set([var for t in trainers for var in t.var_list]))
        self.all_coders = set.union(*[t.all_coders for t in self.trainers])

    def get_executable(
            self, compute_losses=True, summaries=True,
            num_sessions=1) -> Executable:

        self.trainer_idx = (self.trainer_idx + 1) % len(self.trainers)

        return self.trainers[self.trainer_idx].get_executable(
            compute_losses, summaries, num_sessions)

