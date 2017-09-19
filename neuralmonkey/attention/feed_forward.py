"""The feed-forward attention mechanism. This is the attention mechanism
used in Bahdanau et al. (2015)

See arxiv.org/abs/1409.0473
"""
from typing import Union, Optional, Tuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.base_attention import (
    BaseAttention, AttentionLoopState, empty_attention_loop_state,
    get_attention_states, get_attention_mask)
from neuralmonkey.model.stateful import TemporalStateful, SpatialStateful
from neuralmonkey.decorators import tensor
from neuralmonkey.nn.utils import dropout
from neuralmonkey.logging import log


class Attention(BaseAttention):

    def __init__(self,
                 name: str,
                 encoder: Union[TemporalStateful, SpatialStateful],
                 dropout_keep_prob: float = 1.0,
                 state_size: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        BaseAttention.__init__(self, name, save_checkpoint, load_checkpoint)

        self.encoder = encoder
        self.dropout_keep_prob = dropout_keep_prob
        self._state_size = state_size

        # TODO blessing
        log("Hidden features: {}".format(self.hidden_features))
        log("Attention mask: {}".format(self.attention_mask))

    @tensor
    def attention_states(self) -> tf.Tensor:
        return dropout(get_attention_states(self.encoder),
                       self.dropout_keep_prob,
                       self.train_mode)

    @tensor
    def attention_mask(self) -> Optional[tf.Tensor]:
        return get_attention_mask(self.encoder)

    # pylint: disable=no-member
    # Pylint fault from resolving tensor decoration
    @property
    def context_vector_size(self) -> int:
        return self.attention_states.get_shape()[2].value
    # pylint: disable=no-member

    @property
    def state_size(self) -> int:
        if self._state_size is not None:
            return self._state_size
        return self.context_vector_size

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
            # TODO tohle neni spravne
            shape=[self.context_vector_size, self.state_size],
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

        energies = self.get_energies(y, loop_state.weights.identity())

        if self.attention_mask is None:
            weights = tf.nn.softmax(energies)
        else:
            weights_all = tf.nn.softmax(energies) * self.attention_mask
            norm = tf.reduce_sum(weights_all, 1, keep_dims=True) + 1e-8
            weights = weights_all / norm

            # condition = tf.equal(self.attention_mask, 1)
            # masked_logits = tf.where(
            #     tf.tile(condition, [tf.shape(energies)[0], 1]),
            #     energies, -np.inf * tf.ones_like(energies))
            # weights = tf.nn.softmax(masked_logits)

        # Now calculate the attention-weighted vector d.
        context = tf.reduce_sum(
            tf.expand_dims(tf.expand_dims(weights, -1), -1)
            * self._att_states_reshaped, [1, 2])
        context = tf.reshape(context, [-1, self.context_vector_size])

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
