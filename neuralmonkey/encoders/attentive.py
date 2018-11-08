import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import TemporalStatefulWithOutput
from neuralmonkey.model.model_part import ModelPart, InitializerSpecs
from neuralmonkey.nn.utils import dropout
from neuralmonkey.decorators import tensor
from neuralmonkey.attention.base_attention import (
    get_attention_states, get_attention_mask, Attendable)


class AttentiveEncoder(ModelPart, TemporalStatefulWithOutput):
    """An encoder with attention over the input and a fixed-dimension output.

    Based on "A Structured Self-attentive Sentence Embedding",
    https://arxiv.org/abs/1703.03130.

    The encoder combines a sequence of vectors into a fixed-size matrix where
    each row of the matrix is computed using a different attention head. This
    matrix is exposed as the ``temporal_states`` property (the time dimension
    corresponds to the different attention heads). The ``output`` property
    provides a flattened and, optionally, projected representation of this
    matrix.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: Attendable,
                 hidden_size: int,
                 num_heads: int,
                 output_size: int = None,
                 state_proj_size: int = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize an instance of the encoder."""
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.input_sequence = input_sequence
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.state_proj_size = state_proj_size
        self.dropout_keep_prob = dropout_keep_prob

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        with self.use_scope():
            self._attention_states_dropped = dropout(
                get_attention_states(self.input_sequence),
                self.dropout_keep_prob,
                self.train_mode)
    # pylint: enable=too-many-arguments

    @tensor
    def attention_weights(self) -> tf.Tensor:
        mask = get_attention_mask(self.input_sequence)
        hidden = tf.layers.dense(self._attention_states_dropped,
                                 units=self.hidden_size,
                                 activation=tf.tanh, use_bias=False,
                                 name="S1")
        energies = tf.layers.dense(hidden, units=self.num_heads,
                                   use_bias=False, name="S2")
        # shape: [batch_size, max_time, num_heads]
        weights = tf.nn.softmax(energies, dim=1)
        if mask is not None:
            weights *= tf.expand_dims(mask, -1)
            weights /= tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-8

        return weights

    @tensor
    def temporal_states(self) -> tf.Tensor:
        states = self._attention_states_dropped
        if self.state_proj_size is not None:
            states = tf.layers.dense(states, units=self.state_proj_size,
                                     name="state_projection")

        return tf.matmul(a=self.attention_weights, b=states, transpose_a=True)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return tf.ones(tf.shape(self.temporal_states)[:2], tf.float32)

    @tensor
    def output(self) -> tf.Tensor:
        # pylint: disable=no-member
        state_size = self.temporal_states.get_shape()[2].value
        # pylint: enable=no-member
        output = tf.reshape(self.temporal_states,
                            [-1, self.num_heads * state_size])
        if self.output_size is not None:
            output = tf.layers.dense(output, self.output_size,
                                     name="output_projection")

        return output
