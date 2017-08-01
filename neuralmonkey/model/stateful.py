from abc import ABCMeta, abstractproperty
import tensorflow as tf

from neuralmonkey.encoders.attentive import Attentive


class Stateful(metaclass=ABCMeta):

    @abstractproperty
    def output(self) -> tf.Tensor:
        """A 2D `Tensor` of shape (batch, state_size) which contains the
        resulting state of the object.
        """
        raise NotImplementedError("Abstract property")


class TemporalStateful(Attentive):

    # TODO remove when attentions become parts of ini
    def __init__(self, *args, **kwargs) -> None:
        Attentive.__init__(self, *args, **kwargs)

    @abstractproperty
    def temporal_states(self) -> tf.Tensor:
        """A 3D `Tensor` of shape (batch, time, state_size) which contains the
        states of the object in time (e.g. hidden states of a recurrent
        encoder.
        """
        raise NotImplementedError("Abstract property")


class SpatialStateful(Attentive):

    # TODO remove when attentions become parts of ini
    def __init__(self, *args, **kwargs) -> None:
        Attentive.__init__(self, *args, **kwargs)

    @property
    def spatial_states(self) -> tf.Tensor:
        """A 4D `Tensor` of shape (batch, width, height, state_size) which
        contains the states of the object in space (e.g. final layer of a
        convolution network processing an image.
        """
        raise NotImplementedError("Abstract property")


# TODO rename these
class TemporalStatefulWithOutput(Stateful, TemporalStateful):
    # TODO remove when attentions become parts of ini
    def __init__(self, *args, **kwargs) -> None:
        TemporalStateful.__init__(self, *args, **kwargs)


class SpatialStatefulWithOutput(Stateful, SpatialStateful):
    # TODO remove when attentions become parts of ini
    def __init__(self, *args, **kwargs) -> None:
        SpatialStateful.__init__(self, *args, **kwargs)
