"""Module for reverting grandients when passing a model part."""

from typing import List
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.framework import ops
# pylint: disable=no-name-in-module
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.model.stateful import (
    Stateful, TemporalStateful, SpatialStateful)


def _reverse_gradient(x: tf.Tensor) -> tf.Tensor:
    """Flips the sign of the incoming gradient during training."""

    grad_name = "gradient_reversal_{}".format(x.name)

    # pylint: disable=unused-variable,invalid-name,unused-argument
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad)]
    # pylint: enable=unused-variable,invalid-name,unused-argument

    from neuralmonkey.experiment import Experiment
    graph = Experiment.get_current().graph
    with graph.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(x)

    return y


class StatefulView(Stateful):
    """Provides an adversarial view of a `Stateful` object."""

    def __init__(self, reversed_object: Stateful) -> None:
        check_argument_types()
        self._reversed_object = reversed_object

    @tensor
    def output(self) -> tf.Tensor:
        return _reverse_gradient(self._reversed_object.output)

    @property
    def dependencies(self) -> List[str]:
        return super().dependencies + ["_reversed_object"]


class TemporalStatefulView(TemporalStateful):
    """Provides an adversarial view of a `TemporalStateful` object."""

    def __init__(self, reversed_object: TemporalStateful) -> None:
        check_argument_types()
        self._reversed_object = reversed_object

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return _reverse_gradient(self._reversed_object.temporal_states)

    @property
    def temporal_mask(self) -> tf.Tensor:
        return self._reversed_object.temporal_mask

    @property
    def dependencies(self) -> List[str]:
        return super().dependencies + ["_reversed_object"]


class SpatialStatefulView(SpatialStateful):
    """Provides an adversarial view of a `SpatialStateful` object."""

    def __init__(self, reversed_object: SpatialStateful) -> None:
        check_argument_types()
        self._reversed_object = reversed_object

    @tensor
    def spatial_states(self) -> tf.Tensor:
        return _reverse_gradient(self._reversed_object.spatial_states)

    @property
    def spatial_mask(self) -> tf.Tensor:
        return self._reversed_object.spatial_mask

    @property
    def dependencies(self) -> List[str]:
        return super().dependencies + ["_reversed_object"]
