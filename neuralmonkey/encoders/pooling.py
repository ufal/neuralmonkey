from abc import ABCMeta
from typing import Set, cast

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import Stateful, TemporalStateful
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor


# pylint: disable=abstract-method
class SequencePooling(ModelPart, Stateful, metaclass=ABCMeta):
    """An abstract pooling layer over a sequence."""

    def __init__(self,
                 name: str,
                 input_sequence: TemporalStateful,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize an instance of the pooling layer."""
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.input_sequence = input_sequence

        with self.use_scope():
            self.train_mode = tf.placeholder(tf.bool, [], "train_mode")
            self._input_mask = tf.expand_dims(
                self.input_sequence.temporal_mask, -1)
            self._masked_input = (
                self.input_sequence.temporal_states * self._input_mask)

    def get_dependencies(self) -> Set[ModelPart]:
        deps = ModelPart.get_dependencies(self)

        # feed only if needed
        if isinstance(self.input_sequence, ModelPart):
            feedable = cast(ModelPart, self.input_sequence)
            deps |= feedable.get_dependencies()

        return deps

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.train_mode: train}
# pylint: enable=abstract-method


class SequenceMaxPooling(SequencePooling):
    """A max pooling layer over a sequence.

    Takes the maximum of a sequence over time to produce a single state.
    """

    @tensor
    def output(self) -> tf.Tensor:
        # Pad the sequence with a large negative value, but make sure it has
        # non-zero length.
        length = tf.reduce_sum(self._input_mask)
        with tf.control_dependencies([tf.assert_greater(length, 0.5)]):
            padded_input = self._masked_input + 1e-15 * (1 - self._input_mask)
        return tf.reduce_max(padded_input, axis=1)


class SequenceAveragePooling(SequencePooling):
    """An average pooling layer over a sequence.

    Averages a sequence over time to produce a single state.
    """

    @tensor
    def output(self) -> tf.Tensor:
        return (tf.reduce_sum(self._masked_input, axis=1) /
                (tf.reduce_sum(self._input_mask, axis=1) + 1e-8))
