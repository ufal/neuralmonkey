"""Split temporal states such that the sequence is n-times longer."""
from typing import Callable, List
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import TemporalStateful


Activation = Callable[[tf.Tensor], tf.Tensor]


class SequenceSplitter(TemporalStateful, ModelPart):
    def __init__(
            self,
            name: str,
            parent: TemporalStateful,
            factor: int,
            projection_size: int = None,
            projection_activation: Activation = None) -> None:
        """Initialize SentenceSplitter.

        Args:
            parent: TemporalStateful whose states will be split.
            factor: Factor by which the states will be split - the  resulting
                sequence will be longer by this factor.
            projection_size: If not None, specifies dimensionality of a
                projection before state splitting.
            projection_activation: Non-linearity function for the optional
                projection.
        """
        check_argument_types()

        ModelPart.__init__(
            self, name=name, save_checkpoint=None, load_checkpoint=None,
            initializers=None)
        self.parent = parent
        self.factor = factor
        self.projection_size = projection_size
        self.activation = projection_activation

        state_dim = parent.dimension
        if state_dim % factor != 0 and projection_size is None:
            raise ValueError((
                "Dimension of the parent temporal stateful ({}) must be "
                "dividable by the given factor ({}).").format(
                    state_dim, factor))

        if projection_size is not None and projection_size % factor != 0:
            raise ValueError((
                "Dimension of projection ({}) must be "
                "dividable by the given factor ({}).").format(
                    projection_size, factor))

    @tensor
    def temporal_states(self) -> tf.Tensor:
        states = self.parent.temporal_states
        if self.projection_size:
            states = tf.layers.dense(
                states, self.projection_size, activation=self.activation)

        return split_by_factor(states, self.batch_size, self.factor)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        double_mask = tf.stack(
            self.factor * [tf.expand_dims(self.parent.temporal_mask, 2)],
            axis=2)
        return tf.squeeze(
            split_by_factor(double_mask, self.batch_size, self.factor), axis=2)

    def feed_dict(self, dataset: Dataset, train: bool = True) -> FeedDict:
        return ModelPart.feed_dict(self, dataset, train)

    @property
    def singleton_dependencies(self) -> List[str]:
        return super().singleton_dependencies + ["parent"]


def split_by_factor(
        tensor_3d: tf.Tensor, batch_size: tf.Tensor, factor: int) -> tf.Tensor:
    max_time = tf.shape(tensor_3d)[1]
    state_dim = tensor_3d.get_shape()[2].value
    return tf.reshape(
        tensor_3d, [batch_size, max_time * factor, state_dim // factor])
