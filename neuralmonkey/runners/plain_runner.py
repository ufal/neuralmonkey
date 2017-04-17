from typing import Callable, Dict, List, Any
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf

from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)

# pylint: disable=too-few-public-methods


class PlainRunner(BaseRunner):
    """A runner which takes the output from decoder.decoded."""

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 postprocess: Callable[[List[str]], List[str]] = None
                ) -> None:
        super(PlainRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess

    def get_executable(self, compute_losses=False, summaries=True):
        if compute_losses:
            fetches = {"train_loss": self._decoder.train_loss,
                       "runtime_loss": self._decoder.runtime_loss}
        else:
            fetches = {"train_loss": tf.zeros([]),
                       "runtime_loss": tf.zeros([])}

        fetches["decoded"] = self._decoder.decoded

        return PlainExecutable(self.all_coders, fetches,
                               self._decoder.vocabulary,
                               self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]


class PlainExecutable(Executable):

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
        if len(results) != 1:
            raise ValueError('PlainRunner needs exactly 1 execution result, '
                             'got {}'.format(len(results)))

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
