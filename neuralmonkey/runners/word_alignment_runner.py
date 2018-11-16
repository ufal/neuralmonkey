from typing import Dict, List, Set
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.base_attention import BaseAttention
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)


class WordAlignmentRunnerExecutable(Executable):

    def __init__(self,
                 feedables: Set[Feedable],
                 fetches: FeedDict) -> None:
        self._feedables = feedables
        self._fetches = fetches

        self._result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._feedables, self._fetches, []

    def collect_results(self, results: List[Dict]) -> None:
        self._result = ExecutionResult(
            outputs=results[0]["alignment"],
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class WordAlignmentRunner(BaseRunner[BaseAttention]):

    def __init__(self,
                 output_series: str,
                 attention: BaseAttention,
                 decoder: Decoder) -> None:
        check_argument_types()
        BaseRunner[BaseAttention].__init__(self, output_series, attention)

        self._key = "{}_run".format(decoder.name)

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1) -> WordAlignmentRunnerExecutable:
        if self._key not in self._decoder.histories:
            raise KeyError("Attention has no recorded histories under "
                           "key '{}'".format(self._key))

        att_histories = self._decoder.histories[self._key]
        alignment = tf.transpose(att_histories, perm=[1, 2, 0])
        fetches = {"alignment": alignment}

        return WordAlignmentRunnerExecutable(self.feedables, fetches)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return []
