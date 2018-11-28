"""Module that blocks gradient propagation to model parts."""
from typing import List
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.model.stateful import (
    Stateful, TemporalStateful, SpatialStateful)


class StatefulView(Stateful):
    """Provides a gradient-blocking view of a `Stateful` object."""

    def __init__(self, blocked_object: Stateful) -> None:
        check_argument_types()
        self._blocked_object = blocked_object

    @tensor
    def output(self) -> tf.Tensor:
        return tf.stop_gradient(self._blocked_object.output)

    @property
    def dependencies(self) -> List[str]:
        return super().dependencies + ["_blocked_object"]


class TemporalStatefulView(TemporalStateful):
    """Provides a gradient-blocking view of a `TemporalStateful` object."""

    def __init__(self, blocked_object: TemporalStateful) -> None:
        check_argument_types()
        self._blocked_object = blocked_object

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return tf.stop_gradient(self._blocked_object.temporal_states)

    @property
    def temporal_mask(self) -> tf.Tensor:
        return self._blocked_object.temporal_mask

    @property
    def dependencies(self) -> List[str]:
        return super().dependencies + ["_blocked_object"]


class SpatialStatefulView(SpatialStateful):
    """Provides a gradient-blocking view of a `SpatialStateful` object."""

    def __init__(self, blocked_object: SpatialStateful) -> None:
        check_argument_types()
        self._blocked_object = blocked_object

    @tensor
    def spatial_states(self) -> tf.Tensor:
        return tf.stop_gradient(self._blocked_object.spatial_states)

    @property
    def spatial_mask(self) -> tf.Tensor:
        return self._blocked_object.spatial_mask

    @property
    def dependencies(self) -> List[str]:
        return super().dependencies + ["_blocked_object"]
