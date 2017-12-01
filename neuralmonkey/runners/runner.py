from typing import Callable, Dict, List, Any, Set
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf

from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.vocabulary import Vocabulary

# pylint: disable=too-few-public-methods


class GreedyRunExecutable(Executable):

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

        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=image_summaries)


class GreedyRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 postprocess: Callable[[List[str]], List[str]] = None) -> None:
        super(GreedyRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess

        att_plot_summaries = tf.get_collection("summary_att_plots")
        if att_plot_summaries:
            self.image_summaries = tf.summary.merge(att_plot_summaries)
        else:
            self.image_summaries = None

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1) -> GreedyRunExecutable:
        if compute_losses:
            if not hasattr(self._decoder, "train_loss"):
                raise TypeError("Decoder should have the 'train_loss' "
                                "attribute")

            if not hasattr(self._decoder, "runtime_loss"):
                raise TypeError("Decoder should have the 'runtime_loss' "
                                "attribute")
            fetches = {"train_xent": getattr(self._decoder, "train_loss"),
                       "runtime_xent": getattr(self._decoder, "runtime_loss")}
        else:
            fetches = {"train_xent": tf.zeros([]),
                       "runtime_xent": tf.zeros([])}

        if not hasattr(self._decoder, "runtime_logprobs"):
            raise TypeError("Decoder should have the 'runtime_logprobs' "
                            "attribute")

        fetches["decoded_logprobs"] = getattr(self._decoder,
                                              "runtime_logprobs")

        if summaries and self.image_summaries is not None:
            fetches["image_summaries"] = self.image_summaries

        if not hasattr(self._decoder, "vocabulary"):
            raise TypeError("Decoder should have the 'vocabulary' attribute")

        return GreedyRunExecutable(self.all_coders, fetches,
                                   num_sessions,
                                   getattr(self._decoder, "vocabulary"),
                                   self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["train_xent", "runtime_xent"]
