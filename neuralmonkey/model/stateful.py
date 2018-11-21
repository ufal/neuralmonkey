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
from abc import abstractproperty
import tensorflow as tf
from neuralmonkey.model.model_part import GenericModelPart


# pylint: disable=too-few-public-methods
# pydocstyle: disable=
class Stateful(GenericModelPart):
    @abstractproperty
    def output(self) -> tf.Tensor:
        """Return the object output.

        A 2D `Tensor` of shape (batch, state_size) which contains the
        resulting state of the object.
        """
        raise NotImplementedError("Abstract property")
# pylint: enable=too-few-public-methods


class TemporalStateful(GenericModelPart):
    @abstractproperty
    def temporal_states(self) -> tf.Tensor:
        """Return object states in time.

        A 3D `Tensor` of shape (batch, time, state_size) which contains the
        states of the object in time (e.g. hidden states of a recurrent
        encoder.
        """
        raise NotImplementedError("Abstract property")

    @abstractproperty
    def temporal_mask(self) -> tf.Tensor:
        """Return mask for the temporal_states.

        A 2D `Tensor` of shape (batch, time) of type float32 which masks the
        temporal states so each sequence can have a different length. It should
        only contain ones or zeros.
        """
        raise NotImplementedError("Abstract property")

    @property
    def lengths(self) -> tf.Tensor:
        """Return the sequence lengths.

        A 1D `Tensor` of type `int32` that stores the lengths of the
        state sequences in the batch.
        """
        return tf.to_int32(tf.reduce_sum(self.temporal_mask, 1))

    @property
    def dimension(self) -> int:
        """Return the dimension of the states."""
        return self.temporal_states.get_shape()[-1].value


class SpatialStateful(GenericModelPart):
    @property
    def spatial_states(self) -> tf.Tensor:
        """Return object states in space.

        A 4D `Tensor` of shape (batch, width, height, state_size) which
        contains the states of the object in space (e.g. final layer of a
        convolution network processing an image.
        """
        raise NotImplementedError("Abstract property")

    @abstractproperty
    def spatial_mask(self) -> tf.Tensor:
        """Return mask for the spatial_states.

        A 3D `Tensor` of shape (batch, width, height) of type float32
        which masks the spatial states that they can be of different shapes.
        The mask should only contain ones or zeros.
        """
        raise NotImplementedError("Abstract property")

    @property
    def dimension(self) -> int:
        """Return the dimension of the states."""
        return self.spatial_states.get_shape()[-1].value


# pylint: disable=abstract-method
class TemporalStatefulWithOutput(Stateful, TemporalStateful):
    pass


class SpatialStatefulWithOutput(Stateful, SpatialStateful):
    pass
