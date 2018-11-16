from typing import Dict, List, Set, Optional, Callable, Union

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.classifier import Classifier

# pylint: disable=invalid-name
SupportedDecoder = Union[AutoregressiveDecoder, Classifier]
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class GreedyRunExecutable(Executable):

    def __init__(self,
                 feedables: Set[Feedable],
                 fetches: FeedDict,
                 vocabulary: Vocabulary,
                 postprocess: Optional[Postprocessor]) -> None:
        self._feedables = feedables
        self._fetches = fetches
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self._result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._feedables, self._fetches, []

    def collect_results(self, results: List[Dict]) -> None:
        train_loss = 0.
        runtime_loss = 0.
        summed_logprobs = [-np.inf for _ in range(
            results[0]["decoded_logprobs"].shape[0])]

        for sess_result in results:
            train_loss += sess_result["train_xent"]
            runtime_loss += sess_result["runtime_xent"]

            for i, logprob in enumerate(sess_result["decoded_logprobs"]):
                summed_logprobs[i] = np.logaddexp(summed_logprobs[i], logprob)

        argmaxes = [np.argmax(l, axis=1) for l in summed_logprobs]

        decoded_tokens = self._vocabulary.vectors_to_sentences(argmaxes)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess(decoded_tokens)

        image_summaries = results[0].get("image_summaries")

        self._result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=image_summaries)


class GreedyRunner(BaseRunner[SupportedDecoder]):

    def __init__(self,
                 output_series: str,
                 decoder: SupportedDecoder,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[AutoregressiveDecoder].__init__(
            self, output_series, decoder)

        self._postprocess = postprocess

        self.image_summaries = None
        att_plot_summaries = tf.get_collection("summary_att_plots")
        if att_plot_summaries:
            self.image_summaries = tf.summary.merge(att_plot_summaries)

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> GreedyRunExecutable:
        fetches = {"decoded_logprobs": self._decoder.runtime_logprobs,
                   "train_xent": tf.zeros([]),
                   "runtime_xent": tf.zeros([])}

        if compute_losses:
            fetches["train_xent"] = self._decoder.train_loss
            fetches["runtime_xent"] = self._decoder.runtime_loss

        if summaries and self.image_summaries is not None:
            fetches["image_summaries"] = self.image_summaries

        return GreedyRunExecutable(
            self.feedables,
            fetches,
            self._decoder.vocabulary,
            self._postprocess)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["train_xent", "runtime_xent"]
