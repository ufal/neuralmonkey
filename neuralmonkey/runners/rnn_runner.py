from typing import List
import numpy as np

from neuralmonkey.tf_manager import RunResult
from neuralmonkey.runners.base_runner import BaseRunner, \
    Executable, ExecutionResult, NextExecute

# tests: mypy,pylint

# TODO this module should be extenden to implement beamsearch


class RuntimeRnnRunner(BaseRunner):
    """Prepare running the RNN decoder step by step."""

    def __init__(self, output_series, decoder):
        super(RuntimeRnnRunner, self).__init__(output_series, decoder)

        self._initial_fetches = [decoder.runtime_rnn_states[0]]
        self._initial_fetches += [e.encoded for e in self._all_coders
                                  if hasattr(e, 'encoded')]

    def get_executable(self, train=False, summaries=True):

        return RuntimeRnnExecutable(self._all_coders, self._decoder,
                                    self._initial_fetches,
                                    self._decoder.vocabulary)

    @property
    def loss_names(self) -> List[str]:
        return ["runtime_xent"]


class RuntimeRnnExecutable(Executable):
    """Run and ensemble the RNN decoder step by step."""

    def __init__(self, all_coders, decoder, initial_fetches, vocabulary):
        self._all_coders = all_coders
        self._decoder = decoder
        self._vocabulary = vocabulary
        self._initial_fetches = initial_fetches

        self._decoded = []
        self._time_step = 0

        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""

        to_run = [self._decoder.train_logprobs[self._time_step]]
        additional_feed_dict = {t: index for t, index in zip(
            self._decoder.train_inputs[1:], self._decoded)}

        # at the end, we should compute loss
        if self._time_step == self._decoder.max_output - 1:
            to_run.append(self._decoder.train_loss)

        return self._all_coders, to_run, additional_feed_dict

    def collect_results(self, results: List[List[RunResult]]) -> None:
        summed_logprobs = -np.inf
        for sess_result in results:
            summed_logprobs = np.logaddexp(summed_logprobs, sess_result[0])

        ensembled_indices = np.argmax(summed_logprobs, axis=1)
        self._decoded.append(ensembled_indices)
        self._time_step += 1

        decoded_tokens = self._vocabulary.vectors_to_sentences(self._decoded)
        if self._time_step == self._decoder.max_output:
            loss = np.average([res[1] for res in results])
            self.result = ExecutionResult(
                outputs=decoded_tokens,
                losses=[loss],
                scalar_summaries=None,
                histogram_summaries=None,
                image_summaries=None
            )
