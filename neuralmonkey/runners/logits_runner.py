"""A runner outputing logits or normalized distriution from a decoder."""

from typing import Dict, List, Any
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import
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
                 all_coders: List[ModelPart],
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
        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) != 1:
            raise ValueError('LogitsRunner needs exactly 1 execution result, '
                             'got {}'.format(len(results)))

        train_loss = results[0]["train_loss"]
        runtime_loss = results[0]["runtime_loss"]
        logits_list = results[0]["logits"]

        outputs = []
        for instance in logits_list:
            instace_logits = []
            for logits in instance:
                if self._normalize:
                    logits = np.exp(logits) / np.sum(np.exp(logits), axis=0)
                if self._pick_index:
                    instace_logits.append(str(logits[self._pick_index]))
                else:
                    instace_logits.append("\t".join(str(l) for l in logits))
            outputs.append(",".join(instace_logits))

        self.result = ExecutionResult(
            outputs=outputs,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


# pylint: disable=too-few-public-methods
class LogitsRunner(BaseRunner):
    """A runner which takes the output from decoder.decoded."""

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 normalize: bool = True,
                 pick_index: int = None) -> None:
        super(LogitsRunner, self).__init__(output_series, decoder)
        assert check_argument_types()
        self._normalize = normalize
        self._pick_index = pick_index

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> LogitsExecutable:
        if compute_losses:
            fetches = {"train_loss": self._decoder.train_loss,
                       "runtime_loss": self._decoder.runtime_loss}
        else:
            fetches = {"train_loss": tf.zeros([]),
                       "runtime_loss": tf.zeros([])}

        fetches["logits"] = self._decoder.decoded_logits

        return LogitsExecutable(self.all_coders, fetches,
                                self._decoder.vocabulary,
                                self._normalize,
                                self._pick_index)

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
