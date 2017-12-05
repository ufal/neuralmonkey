from typing import Dict, List, cast, Set

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)


class RepresentationExecutable(Executable):

    def __init__(self,
                 prev_coders: Set[ModelPart],
                 encoded: tf.Tensor,
                 used_session: int) -> None:
        self._prev_coders = prev_coders
        self._encoded = encoded
        self._used_session = used_session

        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        return self._prev_coders, {"encoded": self._encoded}, None

    def collect_results(self, results: List[Dict]) -> None:
        if self._used_session > len(results):
            raise ValueError(("Session id {} is higher than number of used "
                              "TensorFlow session ({}).").format(
                                  self._used_session, len(results)))

        vectors = results[self._used_session]["encoded"]

        self.result = ExecutionResult(
            outputs=vectors,
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class RepresentationRunner(BaseRunner):
    """Runner printing out representation from an encoder.

    Using this runner is the way how to get input / other data representation
    out from Neural Monkey.
    """

    def __init__(self,
                 output_series: str,
                 encoder: Stateful,
                 used_session: int = 0) -> None:
        """Initialize the representation runner.

        Args:
            output_series: Name of the output seriesi with vectors.
            encoder: Used encoder.
            used_session: Id of the TensorFlow session used in case of model
                ensembles.
        """
        check_argument_types()

        if not isinstance(encoder, ModelPart):
            raise TypeError("The encoder of the representation runner has to "
                            "be an instance of 'ModelPart'")

        BaseRunner.__init__(self, output_series, cast(ModelPart, encoder))

        self._used_session = used_session  # type: int
        self._encoded = encoder.output  # type: tf.Tensor

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> RepresentationExecutable:
        return RepresentationExecutable(
            self.all_coders, self._encoded, self._used_session)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return []
