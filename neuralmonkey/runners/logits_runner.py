"""A runner outputing logits or normalized distriution from a decoder."""

from typing import Dict, List, Set, Optional
from typeguard import check_argument_types

import numpy as np
import tensorflow as tf

from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.vocabulary import Vocabulary


class LogitsExecutable(Executable):

    def __init__(self,
                 feedables: Set[Feedable],
                 fetches: FeedDict,
                 vocabulary: Vocabulary,
                 normalize: bool,
                 pick_index: Optional[int]) -> None:
        self._feedables = feedables
        self._fetches = fetches
        self._vocabulary = vocabulary
        self._normalize = normalize
        self._pick_index = pick_index

        self._result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._feedables, self._fetches, []

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) != 1:
            raise ValueError("LogitsRunner needs exactly 1 execution result, "
                             "got {}".format(len(results)))

        train_loss = results[0]["train_loss"]
        runtime_loss = results[0]["runtime_loss"]

        # logits_list in shape (time, batch, vocab)
        logits_list = results[0]["logits"]

        # outputs are lists of strings (batch, time)
        outputs = [[] for _ in logits_list[0]]  # type: List[List[str]]

        for time_step in logits_list:
            for logits, output_list in zip(time_step, outputs):

                if self._normalize:
                    logits = np.exp(logits) / np.sum(np.exp(logits), axis=0)
                if self._pick_index:
                    instance_logits = str(logits[self._pick_index])
                else:
                    instance_logits = ",".join(str(l) for l in logits)

                output_list.append(instance_logits)

        str_outputs = [["\t".join(l)] for l in outputs]

        self._result = ExecutionResult(
            outputs=str_outputs,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


# pylint: disable=too-few-public-methods
class LogitsRunner(BaseRunner[Classifier]):
    """A runner which takes the output from decoder.decoded_logits.

    The logits / normalized probabilities are outputted as tab-separates string
    values. If the decoder produces a list of logits (as the recurrent
    decoder), the tab separated arrays are separated with commas.
    Alternatively, we may be interested in a single distribution dimension.
    """

    def __init__(self,
                 output_series: str,
                 decoder: Classifier,
                 normalize: bool = True,
                 pick_index: int = None,
                 pick_value: str = None) -> None:
        """Initialize the logits runner.

        Args:
            output_series: Name of the series produced by the runner.
            decoder: A decoder having logits.
            normalize: Flag whether the logits should be normalized with
                softmax.
            pick_index: If not None, it specifies the index of the logit or the
                probability that should be on output.
            pick_value: If not None, it specifies a value from the decoder's
                vocabulary whose logit or probability should be on output.
        """
        check_argument_types()
        BaseRunner[Classifier].__init__(self, output_series, decoder)

        if pick_index is not None and pick_value is not None:
            raise ValueError("Either a pick index or a vocabulary value can "
                             "be specified, not both at the same time.")

        self._pick_index = None  # type: Optional[int]

        self._normalize = normalize
        if pick_value is not None:
            if pick_value in self._decoder.vocabulary:
                vocab_map = self._decoder.vocabulary.word_to_index
                self._pick_index = vocab_map[pick_value]
            else:
                raise ValueError(
                    "Value '{}' is not in vocabulary of decoder '{}'".format(
                        pick_value, decoder.name))
        else:
            self._pick_index = pick_index

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> LogitsExecutable:
        fetches = {"logits": self._decoder.decoded_logits,
                   "train_loss": tf.zeros([]),
                   "runtime_loss": tf.zeros([])}

        if compute_losses:
            fetches["train_loss"] = self._decoder.train_loss
            fetches["runtime_loss"] = self._decoder.runtime_loss

        return LogitsExecutable(
            self.feedables, fetches, self._decoder.vocabulary,
            self._normalize, self._pick_index)
    # pylint: enable: unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
