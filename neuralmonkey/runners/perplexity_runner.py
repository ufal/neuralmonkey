# pylint: disable=unused-import
from typing import Dict, List, Optional
# pylint: enable=unused-import

import tensorflow as tf
import numpy as np

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


class PerplexityExecutable(Executable):
    def __init__(self, all_coders: List[ModelPart],
                 xent_op: tf.Tensor) -> None:
        self._all_coders = all_coders
        self._xent_op = xent_op
        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._all_coders, {'xents': self._xent_op}, {}

    def collect_results(self, results: List[Dict]) -> None:
        perplexities = np.mean([2 ** res['xents'] for res in results], axis=0)
        xent = float(np.mean([res['xents'] for res in results]))
        self.result = ExecutionResult(
            outputs=perplexities.tolist(),
            losses=[xent],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class PerplexityRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: Decoder) -> None:
        super(PerplexityRunner, self).__init__(output_series, decoder)

        self._decoder_xent = self._decoder.train_xents

    def get_executable(self, compute_losses=False,
                       summaries=True) -> PerplexityExecutable:
        return PerplexityExecutable(self.all_coders,
                                    self._decoder_xent)

    @property
    def loss_names(self) -> List[str]:
        return ["xent"]
