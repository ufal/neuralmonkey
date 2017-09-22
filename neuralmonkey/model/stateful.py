"""Module that provides classes that encapsulate model parts with states.

There are three classes: `Stateful`, `TemporalStateful`, and `SpatialStateful`.

Model parts that do not keep states in time but have a single tensor on the
output should be instances of `Stateful`. Model parts that keep their hidden
states in a time-oriented list (e.g. recurrent encoder) should be instances
of `TemporalStateful`. Model parts that keep the states in a 2D matrix (e.g.
image encoders) should be instances of `SpatialStateful`.

There are also classes that inherit from both stateful and temporal or spatial
stateful (e.g. `TemporalStatefulWithOutput`) that can be used for model parts
that satisfy more requirements (e.g. recurrent encoder).
"""
from abc import ABCMeta, abstractproperty
import tensorflow as tf


# pylint: disable=too-few-public-methods
class Stateful(metaclass=ABCMeta):
    @abstractproperty
    def output(self) -> tf.Tensor:
        """A 2D `Tensor` of shape (batch, state_size) which contains the
        resulting state of the object.
        """
        raise NotImplementedError("Abstract property")
# pylint: enable=too-few-public-methods


class TemporalStateful(metaclass=ABCMeta):
    @abstractproperty
    def temporal_states(self) -> tf.Tensor:
        """A 3D `Tensor` of shape (batch, time, state_size) which contains the
        states of the object in time (e.g. hidden states of a recurrent
        encoder.
        """
        raise NotImplementedError("Abstract property")

    @abstractproperty
    def temporal_mask(self) -> tf.Tensor:
        """A 2D `Tensor` of shape (batch, time) of type float32 which masks the
        temporal states so each sequence can have a different length. It should
        only contain ones or zeros.
        """
        raise NotImplementedError("Abstract property")


class SpatialStateful(metaclass=ABCMeta):
    @property
    def spatial_states(self) -> tf.Tensor:
        """A 4D `Tensor` of shape (batch, width, height, state_size) which
        contains the states of the object in space (e.g. final layer of a
        convolution network processing an image.
        """
        raise NotImplementedError("Abstract property")

    @abstractproperty
    def spatial_mask(self) -> tf.Tensor:
        """A 3D `Tensor` of shape (batch, width, height) of type float32
        which masks the spatial states that they can be of different shapes.
        The mask should only contain ones or zeros.
        """
        raise NotImplementedError("Abstract property")


# pylint: disable=abstract-method
class TemporalStatefulWithOutput(Stateful, TemporalStateful):
    pass


class SpatialStatefulWithOutput(Stateful, SpatialStateful):
    pass
