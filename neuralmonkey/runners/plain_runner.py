from typing import Callable, Dict, List, Any, Set
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf

from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.vocabulary import Vocabulary


# pylint: disable=too-few-public-methods


class PlainExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: FeedDict,
                 num_sessions: int,
                 vocabulary: Vocabulary,
                 postprocess: Optional[Callable]) -> None:
        self.all_coders = all_coders
        self._fetches = fetches
        self._num_sessions = num_sessions
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.decoded_sentences = []  # type: List[List[str]]
        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return (self.all_coders,
                self._fetches,
                None)

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


class PlainRunner(BaseRunner):
    """A runner which takes the output from decoder.decoded."""

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 postprocess: Callable[[List[str]], List[str]] = None
                ) -> None:
        super(PlainRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1):
        if compute_losses:
            if not hasattr(self._decoder, "train_loss"):
                raise TypeError("Decoder should have the 'train_loss' "
                                "attribute")

            if not hasattr(self._decoder, "runtime_loss"):
                raise TypeError("Decoder should have the 'runtime_loss'"
                                "attribute")
            fetches = {"train_loss": getattr(self._decoder, "train_loss"),
                       "runtime_loss": getattr(self._decoder, "runtime_loss")}
        else:
            fetches = {"train_loss": tf.zeros([]),
                       "runtime_loss": tf.zeros([])}

        if not hasattr(self._decoder, "decoded"):
            raise TypeError("Decoder should have the 'decoded' attribute")

        if not hasattr(self._decoder, "vocabulary"):
            raise TypeError("Decoder should have the 'vocabulary' attribute")

        fetches["decoded"] = getattr(self._decoder, "decoded")

        return PlainExecutable(self.all_coders, fetches,
                               num_sessions,
                               getattr(self._decoder, "vocabulary"),
                               self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
