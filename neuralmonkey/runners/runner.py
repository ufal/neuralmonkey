from typing import List
import numpy as np
import tensorflow as tf

from neuralmonkey.tf_manager import RunResult
from neuralmonkey.runners.base_runner import BaseRunner, \
    Executable, ExecutionResult, NextExecute

# tests: mypy,pylint
# pylint: disable=too-few-public-methods


class GreedyRunner(BaseRunner):

    def __init__(self, output_series, decoder):
        super(GreedyRunner, self).__init__(output_series, decoder)

    def get_executable(self, train=False, summaries=True):
        if train:
            losses = [self._decoder.train_loss,
                      self._decoder.runtime_loss]
        else:
            losses = [tf.zeros([]), tf.zeros([])]
        to_run = losses + self._decoder.runtime_logprobs
        return GreedyRunExecutable(self._all_coders, to_run,
                                   self._decoder.vocabulary)

    @property
    def loss_names(self) -> List[str]:
        return ["train_xent", "runtime_xent"]


class GreedyRunExecutable(Executable):

    def __init__(self, all_coders, to_run, vocabulary):
        self.all_coders = all_coders
        self.to_run = to_run
        self.vocabulary = vocabulary

        self.decoded_sentences = []
        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self.to_run, {}

    def collect_results(self, results: List[List[RunResult]]) -> None:
        train_loss = 0.
        runtime_loss = 0.
        summed_logprobs = [-np.inf for _ in range(len(self.to_run) - 2)]

        for sess_result in results:
            train_loss += sess_result[0]
            runtime_loss += sess_result[1]

            for i, logprob in enumerate(sess_result[2:]):
                summed_logprobs[i] = np.logaddexp(summed_logprobs[i], logprob)

        argmaxes = [np.argmax(l, axis=1) for l in summed_logprobs]

        decoded_sentences_batch = \
            self.vocabulary.vectors_to_sentences(argmaxes)
        self.result = ExecutionResult(
            outputs=decoded_sentences_batch,
            losses=[train_loss, runtime_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None
        )
