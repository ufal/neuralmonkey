from typing import List, Dict, Set, Optional, Callable
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.logging import log
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN_INDEX
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.decoders.sequence_labeler import SequenceLabeler

# pylint: disable=invalid-name
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class LabelRunExecutable(Executable):

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
        loss = results[0].get("loss", 0.)
        summed_logprobs = results[0]["label_logprobs"]
        input_mask = results[0]["input_mask"]

        for sess_result in results[1:]:
            loss += sess_result.get("loss", 0.)
            summed_logprobs = np.logaddexp(summed_logprobs,
                                           sess_result["label_logprobs"])
            assert input_mask == sess_result["input_mask"]

        argmaxes = np.argmax(summed_logprobs, axis=2)

        # CAUTION! FABULOUS HACK BELIEVE ME
        argmaxes -= END_TOKEN_INDEX
        argmaxes *= input_mask.astype(int)
        argmaxes += END_TOKEN_INDEX

        # must transpose argmaxes because vectors_to_sentences is time-major
        decoded_labels = self._vocabulary.vectors_to_sentences(argmaxes.T)

        if self._postprocess is not None:
            decoded_labels = self._postprocess(decoded_labels)

        self._result = ExecutionResult(
            outputs=decoded_labels,
            losses=[loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class LabelRunner(BaseRunner[SequenceLabeler]):

    def __init__(self,
                 output_series: str,
                 decoder: SequenceLabeler,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[SequenceLabeler].__init__(self, output_series, decoder)

        self._postprocess = postprocess

        # Make sure the lazy decoder creates its output tensor
        log("Decoder output tensor: {}".format(decoder.decoded))

    # pylint: disable=unused-argument
    # Don't know why it works in Attention.attention and not here.
    # Parameters are unused beacause they are inherited.
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> LabelRunExecutable:
        fetches = {
            "label_logprobs": self._decoder.logprobs,
            "input_mask": self._decoder.encoder.input_sequence.temporal_mask}

        if compute_losses:
            fetches["loss"] = self._decoder.cost

        return LabelRunExecutable(
            self.feedables, fetches, self._decoder.vocabulary,
            self._postprocess)
    # pylint: enable: unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["loss"]
