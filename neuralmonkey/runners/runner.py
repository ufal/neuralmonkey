from typing import List, Tuple
import tensorflow as tf

from neuralmonkey.tf_manager import RunResult
from neuralmonkey.runners.base_runner import BaseRunner, \
        Executable, ExecutionResult

# tests: mypy,pylint
# pylint: disable=too-few-public-methods

class GreedyRunner(BaseRunner):

    def __init__(self, output_series, decoder):
        super(GreedyRunner, self).__init__(output_series, decoder)

    def get_executable(self, train=False, summaries=True):
        if train:
            losses = [self.decoder.train_loss,
                      self.decoder.runtime_loss]
        else:
            losses = [tf.zeros([]), tf.zeros([])]
        to_run = losses + self.decoder.decoded
        return GreedyRunExecutable(self.all_coders, to_run,
                                   self.decoder.vocabulary)


class GreedyRunExecutable(Executable):

    def __init__(self, all_coders, to_run, vocabulary):
        self.all_coders = all_coders
        self.to_run = to_run
        self.vocabulary = vocabulary

        self.loss_with_gt_ins = 0.0
        self.loss_with_decoded_ins = 0.0
        self.decoded_sentences = []
        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> Tuple[List[object], List[tf.Tensor]]:
        """Get the feedables and tensors to run."""
        return self.all_coders, self.to_run

    def collect_results(self, results: List[List[RunResult]]) -> None:
        # TODO do ensembles
        if len(results) > 1:
            raise Exception('Does not support ensembling of multiple sessions')

        sess_results = results[0]
        decoded_sentences_batch = \
            self.vocabulary.vectors_to_sentences(sess_results[2:])
        self.result = ExecutionResult(
            outputs=decoded_sentences_batch,
            losses=sess_results[:2],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None
        )

class ExecutionResult(object):

    def __init__(self, outputs, losses):
        self.outputs = outputs
        self.losses = losses
