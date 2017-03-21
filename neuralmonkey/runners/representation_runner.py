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
                 encoded: tf.Tensor,
                 used_session: int) -> None:
        self._prev_coders = prev_coders
        self._encoded = encoded
        self._used_session = used_session

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        return self._prev_coders, {"encoded": self._encoded}, {}

    def collect_results(self, results: List[Dict]) -> None:
        if self._used_session > len(results):
            raise ValueError(("Session id {} is higher than number of used "
                              "TensorFlow session ({}).").format(
                                  self._used_session, len(results)))

        vectors = results[self._used_session]['encoded']

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
                 encoder: ModelPart,
                 used_session: int=0) -> None:
        """Initialize the representation runner.

        Args:
            output_series: Name of the output seriesi with vectors.
            encoder: Used encoder.
            used_session: Id of the TensorFlow session used in case of model
                ensembles.
        """
        super(RepresentationRunner, self).__init__(output_series, encoder)

        self._used_session = used_session
        self._encoded = encoder.encoded  # type: ignore

    def get_executable(self, compute_losses=False,
                       summaries=True) -> RepresentationExecutable:
        return RepresentationExecutable(self.all_coders,
                                        self._encoded,
                                        self._used_session)

    @property
    def loss_names(self) -> List[str]:
        return []
