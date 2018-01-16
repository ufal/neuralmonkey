"""Module that blocks gradient propagation to model parts."""
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import (
    Stateful, TemporalStateful, SpatialStateful,
    TemporalStatefulWithOutput, SpatialStatefulWithOutput)


# pylint: disable=too-few-public-methods
class StatefulView(Stateful):

    def __init__(self, blocked_object: Stateful) -> None:
        check_argument_types()
        self._output = tf.stop_gradient(blocked_object.output)

    @property
    def output(self) -> tf.Tensor:
        return self._output
# pylint: enable=too-few-public-methods


class TemporalStatefulView(TemporalStateful):
    def __init__(self, blocked_object: TemporalStateful) -> None:
        check_argument_types()
        self._blocked_object = blocked_object
        self._states = tf.stop_gradient(blocked_object.temporal_states)

    @property
    def temporal_states(self) -> tf.Tensor:
        return self._states

    @property
    def temporal_mask(self) -> tf.Tensor:
        return self._blocked_object.temporal_mask


class SpatialStatefulView(SpatialStateful):
    def __init__(self, blocked_object: SpatialStateful) -> None:
        check_argument_types()
        self._blocked_object = blocked_object
        self._states = tf.stop_gradient(blocked_object.spatial_states)

    @property
    def spatial_states(self) -> tf.Tensor:
        return self._states

    @property
    def spatial_mask(self) -> tf.Tensor:
        return self._blocked_object.spatial_mask


# pylint: disable=abstract-method
class TemporalStatefulWithOutputView(TemporalStatefulWithOutput,
                                     TemporalStatefulView, StatefulView):
    def __init__(self, blocked_object: TemporalStatefulWithOutput) -> None:
        check_argument_types()
        TemporalStatefulView.__init__(self, blocked_object)
        StatefulView.__init__(self, blocked_object)


class SpatialStatefulWithOutputView(SpatialStatefulWithOutput,
                                    SpatialStatefulView, StatefulView):
    def __init__(self, blocked_object: SpatialStatefulWithOutput) -> None:
        check_argument_types()
        SpatialStatefulView.__init__(self, blocked_object)
        StatefulView.__init__(self, blocked_object)
