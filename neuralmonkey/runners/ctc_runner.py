from typing import Callable, Dict, List, Any
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf

from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
from neuralmonkey.vocabulary import END_TOKEN

# pylint: disable=too-few-public-methods


class CTCRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 postprocess: Callable[[List[str]], List[str]]=None) -> None:
        super(CTCRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess

    def get_executable(self, compute_losses=False, summaries=True):
        if compute_losses:
            fetches = {"train_loss": self._decoder.train_loss,
                       "runtime_loss": self._decoder.runtime_loss}
        else:
            fetches = {"train_loss": tf.zeros([]),
                       "runtime_loss": tf.zeros([])}

        fetches["decoded"] = self._decoder.decoded

        return CTCExecutable(self.all_coders, fetches,
                             self._decoder.vocabulary,
                             self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]


class CTCExecutable(Executable):

    def __init__(self, all_coders, fetches, vocabulary, postprocess) -> None:
        self.all_coders = all_coders
        self._fetches = fetches
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.decoded_sentences = []  # type: List[List[str]]
        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        train_loss = 0.
        runtime_loss = 0.

        for sess_result in results:
            train_loss += sess_result["train_loss"]
            runtime_loss += sess_result["runtime_loss"]

        decoded = sess_result["decoded"]

        decoded_tokens = self._vocabulary.vectors_to_sentences(decoded)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess(decoded_tokens)

        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
