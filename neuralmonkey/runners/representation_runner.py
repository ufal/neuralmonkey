"""A runner that prints out the input representation from an encoder."""

# pylint: disable=unused-import
from typing import Dict, List, Optional
# pylint: enable=unused-import

import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


class RepresentationExecutable(Executable):
    def __init__(self, prev_coders: List[ModelPart],
                 encoded: tf.Tensor) -> None:
        self._prev_coders = prev_coders
        self._encoded = encoded

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        return self._prev_coders, {"encoded": self._encoded}, {}

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) > 1:
            raise ValueError(
                "It does not make sense to collect representation vectors from"
                "multiple sessions because they don't have to occupy the same"
                "vector space.")
        vectors = results[0]['encoded']

        self.result = ExecutionResult(
            outputs=vectors.tolist(),
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class RepresentationRunner(BaseRunner):
    """Runner printing out representation from a encoder.

    Using this runner is the way how to get input / other data representation
    out from Neural Monkey.
    """
    def __init__(self,
                 output_series: str,
                 encoder: ModelPart) -> None:
        super(RepresentationRunner, self).__init__(output_series, encoder)

        self._encoded = encoder.encoded  # type: ignore

    def get_executable(self, compute_losses=False,
                       summaries=True) -> RepresentationExecutable:
        return RepresentationExecutable(self.all_coders,
                                        self._encoded)

    @property
    def loss_names(self) -> List[str]:
        return []
