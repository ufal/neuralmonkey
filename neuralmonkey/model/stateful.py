from abc import ABCMeta


class Stateful(metaclass=ABCMeta):

    @property
    def output(self) -> tf.Tensor:
        """A 2D `Tensor` of shape (batch, state_size) which contains the
        resulting state of the object.
        """
        raise NotImplementedError("Abstract property")


class TemporalStateful(metaclass=ABCMeta):

    @property
    def temporal_states(self) -> tf.Tensor:
        """A 3D `Tensor` of shape (batch, time, state_size) which contains the
        states of the object in time (e.g. hidden states of a recurrent
        encoder.
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
