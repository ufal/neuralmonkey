"""A runner outputing logits or normalized distriution from a decoder."""

from typing import Dict, List, Any, Set
from typeguard import check_argument_types

import numpy as np
import tensorflow as tf

from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              FeedDict, ExecutionResult,
                                              NextExecute)
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.vocabulary import Vocabulary


class LogitsExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: FeedDict,
                 vocabulary: Vocabulary,
                 normalize: bool = True,
                 pick_index: int = None) -> None:
        self.all_coders = all_coders
        self._fetches = fetches
        self._vocabulary = vocabulary
        self._normalize = normalize
        self._pick_index = pick_index

        self.decoded_sentences = []  # type: List[List[str]]
        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) != 1:
            raise ValueError('LogitsRunner needs exactly 1 execution result, '
                             'got {}'.format(len(results)))

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

        self.result = ExecutionResult(
            outputs=str_outputs,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


# pylint: disable=too-few-public-methods
class LogitsRunner(BaseRunner):
    """A runner which takes the output from decoder.decoded_logits.

    The logits / normalized probabilities are outputted as tab-separates string
    values. If the decoder produces a list of logits (as the recurrent
    decoder), the tab separated arrays are separated with commas.
    Alternatively, we may be interested in a single distribution dimension.
    """

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 normalize: bool = True,
                 pick_index: int = None,
                 pick_value: str = None) -> None:
        """Initializes the logits runner.

        Args:
            output_series: Name of the series produces by the runner.
            decoder: A decoder having logits.
            normalize: Flag whether the logits should be normalized with
                softmax.
            pick_index: If not None, it specifies the index of the logit or the
                probability that should be on output.
            pick_value: If not None, it specifies a value from the decoder's
                vocabulary whose logit or probability should be on output.
        """
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

        if not hasattr(self._decoder, "vocabulary"):
            raise TypeError("Decoder for logits runner should have the "
                            "'vocabulary' attribute")

        if not hasattr(self._decoder, "decoded_logits"):
            raise TypeError("Decoder for logits runner should have the "
                            "'decoded_logits' attribute")

        # TODO why these two?
        if not hasattr(self._decoder, "train_loss"):
            raise TypeError("Decoder for logits runner should have the "
                            "'train_loss' attribute")

        if not hasattr(self._decoder, "vocabulary"):
            raise TypeError("Decoder for logits runner should have the "
                            "'runtime_loss' attribute")

        self._decoder_vocab = getattr(self._decoder, "vocabulary")
        self._decoder_train_loss = getattr(self._decoder, "train_loss")
        self._decoder_runtime_loss = getattr(self._decoder, "runtime_loss")
        self._decoder_logits = getattr(self._decoder, "decoded_logits")

        if pick_index is not None and pick_value is not None:
            raise ValueError("Either a pick index or a vocabulary value can "
                             "be specified, not both at the same time.")

        self._normalize = normalize
        if pick_value is not None:
            if pick_value in self._decoder_vocab:
                vocab_map = self._decoder_vocab.word_to_index
                self._pick_index = vocab_map[pick_value]
            else:
                raise ValueError(
                    "Value '{}' is not in vocabulary of decoder '{}'".format(
                        pick_value, decoder.name))
        else:
            self._pick_index = pick_index

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> LogitsExecutable:
        if compute_losses:
            fetches = {"train_loss": self._decoder_train_loss,
                       "runtime_loss": self._decoder_runtime_loss}
        else:
            fetches = {"train_loss": tf.zeros([]),
                       "runtime_loss": tf.zeros([])}

        fetches["logits"] = self._decoder_logits

        return LogitsExecutable(self.all_coders, fetches,
                                self._decoder_vocab,
                                self._normalize,
                                self._pick_index)

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
