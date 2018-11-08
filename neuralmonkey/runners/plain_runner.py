from typing import Dict, List, Set, Union, Callable, Optional

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.ctc_decoder import CTCDecoder
from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.decoders.sequence_labeler import SequenceLabeler
from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.vocabulary import Vocabulary

# pylint: disable=invalid-name
SupportedDecoder = Union[
    AutoregressiveDecoder, CTCDecoder, Classifier, SequenceLabeler]
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class PlainExecutable(Executable):

    def __init__(self,
                 all_coders: Set[GenericModelPart],
                 fetches: FeedDict,
                 num_sessions: int,
                 vocabulary: Vocabulary,
                 postprocess: Optional[Postprocessor]) -> None:
        self._all_coders = all_coders
        self._fetches = fetches
        self._num_sessions = num_sessions
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._all_coders, self._fetches, []

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) != 1:
            raise ValueError("PlainRunner needs exactly 1 execution result, "
                             "got {}".format(len(results)))

        train_loss = results[0]["train_loss"]
        runtime_loss = results[0]["runtime_loss"]
        decoded = results[0]["decoded"]

        decoded_tokens = self._vocabulary.vectors_to_sentences(decoded)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess(decoded_tokens)

        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class PlainRunner(BaseRunner[SupportedDecoder]):
    """A runner which takes the output from decoder.decoded."""

    def __init__(self,
                 output_series: str,
                 decoder: SupportedDecoder,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[SupportedDecoder].__init__(self, output_series, decoder)

        self._postprocess = postprocess

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int):
        fetches = {"decoded": self._decoder.decoded,
                   "train_loss": tf.zeros([]),
                   "runtime_loss": tf.zeros([])}

        if compute_losses:
            fetches["train_loss"] = self._decoder.train_loss
            fetches["runtime_loss"] = self._decoder.runtime_loss

        return PlainExecutable(
            self.all_coders, fetches, num_sessions, self._decoder.vocabulary,
            self._postprocess)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
