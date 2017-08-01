"""Module which implements decoding functions using multiple attentions
for RNN decoders.

See http://arxiv.org/abs/1606.07481

The attention mechanisms used in Neural Monkey are inherited from the
``BaseAttention`` class defined in this module.

Each attention object has the ``attention`` function which operates on the
``attention_states`` tensor.  The attention function receives the query tensor,
the decoder previous state and input, and its inner state, which could bear an
arbitrary structure of information. The default structure for this is the
``AttentionLoopState``, which contains a growing array of attention
distributions and context vectors in time. That's why there is the
``initial_loop_state`` function in the ``BaseAttention`` class.

Mainly for illustration purposes, the attention objects can keep their
*histories*, which is a dictionary populated with attention distributions in
time for every decoder, that used this attention object. This is because for
example the recurrent decoder is can be run twice for each sentence - once in
the *training* mode, in which the decoder gets the reference tokens on the
input, and once in the *running* mode, in which it gets its own outputs. The
histories object is constructed *after* the decoding and its construction
should be triggered manually from the decoder by calling the ``finalize_loop``
method.
"""
from typing import NamedTuple, Dict, Optional, Any, Tuple

import tensorflow as tf

from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset
from neuralmonkey.nn.utils import dropout

# pylint: disable=invalid-name
AttentionLoopState = NamedTuple("AttentionLoopState",
                                [("contexts", tf.TensorArray),
                                 ("weights", tf.TensorArray)])
# pylint: enable=invalid-name


def empty_attention_loop_state() -> AttentionLoopState:
    """Create an empty attention loop state.

    The attention loop state is a technical object for storing the attention
    distributions and the context vectors in time. It is used with the
    ``tf.while_loop`` dynamic implementation of the decoder.

    This function returns an empty attention loop state which means there are
    two empty arrays, one for attention distributions in time, and one for
    the attention context vectors in time.
    """
    return AttentionLoopState(
        contexts=tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True,
            name="contexts"),
        weights=tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True,
            name="distributions"))


class BaseAttention(ModelPart):
    def __init__(self,
                 name: str,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        # TODO is the next TODO still valid?
        # TODO create context vector size property
        self._histories = {}  # type: Dict[str, tf.Tensor]

    @property
    def histories(self) -> Dict[str, tf.Tensor]:
        return self._histories

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: Any,
                  step: tf.Tensor) -> Tuple[tf.Tensor, Any]:
        """Get context vector for a given query."""
        raise NotImplementedError("Abstract method")

    def initial_loop_state(self) -> Any:
        """Get initial loop state for the attention object."""
        raise NotImplementedError("Abstract method")

    def finalize_loop(self, key: str, last_loop_state: Any) -> None:
        raise NotImplementedError("Abstract method")

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, [], "train_mode")
    # pylint: enable=no-self-use

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.train_mode: train}


class Attention(BaseAttention):

    def __init__(self,
                 name: str,
                 # TODO not just temporal
                 encoder: TemporalStateful,
                 dropout_keep_prob: float = 1.0,
                 state_size: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        BaseAttention.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoder = encoder
        self.dropout_keep_prob = dropout_keep_prob
        self._state_size = state_size

    @tensor
    def attention_states(self) -> tf.Tensor:
        return dropout(self.encoder.temporal_states,
                       self.dropout_keep_prob,
                       self.train_mode)

    # pylint: disable=no-member
    # Pylint fault from resolving tensor decoration
    @property
    def input_state_size(self) -> int:
        return self.attention_states.get_shape()[2].value
    # pylint: disable=no-member

    @property
    def state_size(self) -> int:
        if self._state_size is not None:
            return self._state_size
        return self.input_state_size

    @tensor
    def query_projection_matrix(self) -> tf.Variable:
        with tf.variable_scope("Attention"):
            return tf.get_variable(
                name="attn_query_projection",
                shape=[self.query_state_size, self.state_size],
                initializer=tf.random_normal_initializer(stddev=0.001))

    @tensor
    def key_projection_matrix(self) -> tf.Variable:
        return tf.get_variable(
            name="attn_key_projection",
            shape=[self.input_state_size, self.state_size],
            initializer=tf.random_normal_initializer(stddev=0.001))

    @tensor
    def similarity_bias_vector(self) -> tf.Variable:
        return tf.get_variable(
            name="attn_similarity_v", shape=[self.state_size],
            initializer=tf.random_normal_initializer(stddev=.001))

    @tensor
    def projection_bias_vector(self) -> tf.Variable:
        return tf.get_variable(
            name="attn_projection_bias", shape=[self.state_size],
            initializer=tf.zeros_initializer())

    # pylint: disable=no-self-use
    # Implicit self use in tensor annotation
    @tensor
    def bias_term(self) -> tf.Variable:
        return tf.get_variable(
            name="attn_bias", shape=[],
            initializer=tf.constant_initializer(0))
    # pylint: enable=no-self-use

    @tensor
    def _att_states_reshaped(self) -> tf.Tensor:
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to
        # reshape before.
        return tf.expand_dims(self.attention_states, 2)

    @tensor
    def hidden_features(self) -> tf.Tensor:
        # This variable corresponds to Bahdanau's U_a in the paper
        key_proj_reshaped = tf.expand_dims(
            tf.expand_dims(self.key_projection_matrix, 0), 0)

        return tf.nn.conv2d(
            self._att_states_reshaped, key_proj_reshaped, [1, 1, 1, 1], "SAME")

    def get_energies(self, y, _):
        return tf.reduce_sum(
            self.similarity_bias_vector * tf.tanh(self.hidden_features + y),
            [2, 3]) + self.bias_term

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: AttentionLoopState,
                  step: tf.Tensor) -> Tuple[tf.Tensor, AttentionLoopState]:
        self.query_state_size = query.get_shape()[-1].value

        y = tf.matmul(query, self.query_projection_matrix)
        y = y + self.projection_bias_vector
        y = tf.reshape(y, [-1, 1, 1, self.state_size])

        energies = self.get_energies(y, loop_state.weights)

        if self.encoder.temporal_mask is None:
            weights = tf.nn.softmax(energies)
        else:
            weights_all = tf.nn.softmax(energies) * self.encoder.temporal_mask
            norm = tf.reduce_sum(weights_all, 1, keep_dims=True) + 1e-8
            weights = weights_all / norm

        # Now calculate the attention-weighted vector d.
        context = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(weights, -1), -1)
            * self._att_states_reshaped, [1, 2])
        context = tf.reshape(context, [-1, self.input_state_size])

        next_contexts = loop_state.contexts.write(step, context)
        next_weights = loop_state.weights.write(step, weights)

        next_loop_state = AttentionLoopState(
            contexts=next_contexts,
            weights=next_weights)

        return context, next_loop_state

    def initial_loop_state(self) -> AttentionLoopState:
        return empty_attention_loop_state()

    def finalize_loop(self, key: str,
                      last_loop_state: AttentionLoopState) -> None:
        self.histories[key] = last_loop_state.weights.stack()
