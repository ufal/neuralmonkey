from typing import List, Optional, Any

import tensorflow as tf

from neuralmonkey.runners.base_runner import BaseRunner, Executable
from neuralmonkey.trainers.generic_trainer import GenericTrainer, Objective, ObjectiveWeight
from neuralmonkey.trainers.cross_entropy_trainer import CrossEntropyTrainer, xent_objective


# pylint: disable=too-few-public-methods
class ReconstructionTrainer(CrossEntropyTrainer):

    def __init__(self,
                 feeds,
                 fetches,
                 decoders: List[Any],
                 decoder_weights: List[ObjectiveWeight] = None,
                 l1_weight: float = 0.,
                 l2_weight: float = 0.,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()
        CrossEntropyTrainer.__init__(
            self, decoders, decoder_weigths, l1_weight, l2_weight, clip_norm,
            optimizer, var_scopes, var_collection)

        self.feeds = feeds
        self.fetches = fetches

    def get_executable(
            self, compute_losses=True, summaries=True,
            num_sessions=1) -> Executable:
        assert compute_losses

        return TrainExecutable(self.fetches,
                               self.feeds,
                               self.all_coders,
                               num_sessions,
                               self.train_op,
                               self.losses,
                               self.scalar_summaries if summaries else None,
                               self.histogram_summaries if summaries else None)


class ReconstructionTrainExecutable(TrainExecutable):

    def __init__(self,
                 to_fetch,
                 to_feed,
                 all_coders,
                 num_sessions,
                 train_op,
                 losses,
                 scalar_summaries,
                 histogram_summaries) -> None:
        TrainExecutable.__init__(self, all_coders, num_sessions, train_op,
                                 losses, scalar_summaries, histogram_summaries)

        self.fetches_list = to_fetch + [train_op]
        self.feeds_list = to_feed

        self.feed_dicts = [{} for _ in range(self.num_sessions)]
        self.result = None

    def next_to_execute(self) -> NextExecute:
        fetches = {}
        fetch = self.fetches_list[0]
        self.fetches_list = self.fetches_list[1:]
        if not self.fetches_list:
            if self.scalar_summaries is not None:
                fetches["scalar_summaries"] = self.scalar_summaries
                fetches["histogram_summaries"] = self.histogram_summaries
            fetches["loss"] = self.losses
        fetches["step_op"] = fetch

        return self.all_coders, fetches, self.feed_dicts

    def collect_results(self, results: List[Dict]) -> None:
        if self.feeds_list:
            for i, session_result in enumerate(results):
                next_feed = self.feeds_list[0]
                self.feeds_list = self.feeds_list[1:]
                self.feed_dicts[i][next_feed] = session_result["step_op"]
        else:
            self.result = super(TrainExecutable, self).collect_results(results)

        return self.result
