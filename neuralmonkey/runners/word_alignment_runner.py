from typing import Dict, List

import tensorflow as tf

from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


class WordAlignmentRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 encoder: ModelPart,
                 decoder: Decoder) -> None:
        super(WordAlignmentRunner, self).__init__(output_series, decoder)

        self._encoder = encoder

    def get_executable(self, compute_losses=False, summaries=True):
        att_object = self._decoder.get_attention_object(self._encoder,
                                                        train_mode=False)
        alignment = tf.transpose(
            tf.stack(att_object.attentions_in_time), perm=[1, 2, 0])
        fetches = {'alignment': alignment}

        return WordAlignmentRunnerExecutable(self.all_coders, fetches)

    @property
    def loss_names(self) -> List[str]:
        return []


class WordAlignmentRunnerExecutable(Executable):

    def __init__(self, all_coders, fetches):
        self.all_coders = all_coders
        self._fetches = fetches

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        self.result = ExecutionResult(
            outputs=results[0]['alignment'],
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
