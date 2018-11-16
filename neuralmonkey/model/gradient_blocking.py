"""Module that blocks gradient propagation to model parts."""
from typing import List
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import (
    Stateful, TemporalStateful, SpatialStateful)


class StatefulView(Stateful):
    def __init__(self, blocked_object: Stateful) -> None:
        check_argument_types()
        self._blocked_object = blocked_object
        self._output = tf.stop_gradient(blocked_object.output)

    @property
    def output(self) -> tf.Tensor:
        return self._output

    @property
    def _singleton_dependencies(self) -> List[str]:
        return super()._singleton_dependencies + ["_blocked_object"]


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

    @property
    def _singleton_dependencies(self) -> List[str]:
        return super()._singleton_dependencies + ["_blocked_object"]


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

    @property
    def _singleton_dependencies(self) -> List[str]:
        return super()._singleton_dependencies + ["_blocked_object"]
