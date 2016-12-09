from typing import Callable, Dict, List
import numpy as np
import tensorflow as tf

from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)

# tests: mypy,pylint
# pylint: disable=too-few-public-methods


class GreedyRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder,
                 postprocess: Callable[[List[str]], List[str]]=None) -> None:
        super(GreedyRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess

        val_plot_summaries = tf.get_collection("summary_val_plots")
        if val_plot_summaries:
            self.image_summaries = tf.merge_summary(val_plot_summaries)
        else:
            self.image_summaries = []

    def get_executable(self, train=False, summaries=True):
        if train:
            fecthes = {"train_xent": self._decoder.train_loss,
                       "runtime_xent": self._decoder.runtime_loss}
        else:
            fecthes = {"train_xent": tf.zeros([]),
                       "runtime_xent": tf.zeros([])}
        fecthes["decoded_logprobs"] = self._decoder.runtime_logprobs

        if summaries and self.image_summaries:
            fecthes['image_summaries'] = self.image_summaries

        return GreedyRunExecutable(self.all_coders, fecthes,
                                   self._decoder.vocabulary,
                                   self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["train_xent", "runtime_xent"]


class GreedyRunExecutable(Executable):

    def __init__(self, all_coders, fecthes, vocabulary, postprocess):
        self.all_coders = all_coders
        self._fetches = fecthes
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.decoded_sentences = []
        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        train_loss = 0.
        runtime_loss = 0.
        summed_logprobs = [-np.inf for _ in self._fetches["decoded_logprobs"]]

        for sess_result in results:
            train_loss += sess_result["train_xent"]
            runtime_loss += sess_result["runtime_xent"]

            for i, logprob in enumerate(sess_result["decoded_logprobs"]):
                summed_logprobs[i] = np.logaddexp(summed_logprobs[i], logprob)

        argmaxes = [np.argmax(l, axis=1) for l in summed_logprobs]

        decoded_tokens = self._vocabulary.vectors_to_sentences(argmaxes)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess(decoded_tokens)

        image_summaries = results[0].get('image_summaries')

        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=image_summaries)
