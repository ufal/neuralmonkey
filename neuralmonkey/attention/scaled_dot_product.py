"""The scaled dot-product attention mechanism, defined in Vaswani et al. (2017)

See arxiv.org/abs/1706.03762
"""
import math
from typing import Tuple, List, NamedTuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.nn.utils import dropout
from neuralmonkey.nn.projection import linear
from neuralmonkey.attention.base_attention import (
    BaseAttention, Attendable, get_attention_states, get_attention_mask)

# pylint: disable=invalid-name
MultiHeadLoopStateTA = NamedTuple("MultiHeadLoopStateTA",
                                  [("contexts", tf.TensorArray),
                                   ("head_weights", List[tf.TensorArray])])
# pylint: enable=invalid-name


class MultiHeadAttention(BaseAttention):

    def __init__(self,
                 name: str,
                 n_heads: int,
                 keys_encoder: Attendable,
                 values_encoder: Attendable = None,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        BaseAttention.__init__(self, name, save_checkpoint, load_checkpoint)

        self.n_heads = n_heads
        self.dropout_keep_prob = dropout_keep_prob

        if self.n_heads <= 0:
            raise ValueError("Number of heads must be greater than zero.")

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        if values_encoder is None:
            values_encoder = keys_encoder

        self.attention_keys = get_attention_states(keys_encoder)
        self.attention_values = get_attention_states(values_encoder)
        self.attention_mask = get_attention_mask(keys_encoder)

        self._dimension = self.attention_keys.get_shape()[-1].value

        if self._dimension % self.n_heads != 0:
            raise ValueError("Model dimension ({}) must be divisible by the "
                             "number of attention heads ({})"
                             .format(self._dimension, self.n_heads))

        self._head_dim = int(self._dimension / self.n_heads)
        self._scaling_factor = 1 / math.sqrt(self._head_dim)

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: MultiHeadLoopStateTA,
                  step: tf.Tensor) -> Tuple[tf.Tensor, MultiHeadLoopStateTA]:

        if self.n_heads == 1:
            context, weights = self.attention_single_head(
                query, self.attention_keys, self.attention_values)
            head_weights = [weights]
        else:
            # project query, keys and vals: [batch, rnn] to [batch, rnn2]
            head_contexts, head_weights = zip(*[
                self.attention_single_head(
                    linear(query, self._head_dim,
                           scope="query_proj_head{}".format(i)),
                    linear(self.attention_keys, self._head_dim,
                           scope="keys_proj_head{}".format(i)),
                    linear(self.attention_values, self._head_dim,
                           scope="values_proj_head{}".format(i)))
                for i in range(self.n_heads)])

            context = linear(tf.concat(head_contexts, -1), self._dimension,
                             scope="output_proj")

        next_contexts = loop_state.contexts.write(step, context)
        next_head_weights = [loop_state.head_weights[i].write(step,
                                                              head_weights[i])
                             for i in range(self.n_heads)]

        next_loop_state = MultiHeadLoopStateTA(
            contexts=next_contexts,
            head_weights=next_head_weights)

        return context, next_loop_state

    def attention_single_head(self, query: tf.Tensor,
                              keys: tf.Tensor,
                              values: tf.Tensor) -> Tuple[tf.Tensor,
                                                          tf.Tensor]:
        # shape: batch, time (similarities of attention keys in batch and time
        # to the queries in the batch)
        dot_product = tf.reduce_sum(
            tf.expand_dims(query, 1) * keys, [-1])
        energies = dot_product * self._scaling_factor

        weights = tf.nn.softmax(energies)

        if self.attention_mask is not None:
            weights_all = weights * self.attention_mask
            norm = tf.reduce_sum(weights_all, 1, keep_dims=True) + 1e-8
            weights = weights_all / norm

        # apply dropout to the weights (Attention Dropout)
        weights = dropout(weights, self.dropout_keep_prob, self.train_mode)

        # sum up along the time axis
        context = tf.reduce_sum(
            tf.expand_dims(weights, -1) * values, [1])

        return context, weights

    def initial_loop_state(self) -> MultiHeadLoopStateTA:
        return MultiHeadLoopStateTA(
            contexts=tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True,
                name="contexts"),
            head_weights=[tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True,
                name="distributions_head{}".format(i), clear_after_read=False)
                          for i in range(self.n_heads)])

    def finalize_loop(self, key: str,
                      last_loop_state: MultiHeadLoopStateTA) -> None:
        for i in range(self.n_heads):
            head_weights = last_loop_state.head_weights[i].stack()
            self.histories["{}_head{}".format(key, i)] = head_weights

    @property
    def context_vector_size(self) -> int:
        return self.attention_values.get_shape()[-1].value

    def visualize_attention(self, key: str) -> None:
        for i in range(self.n_heads):
            head_key = "{}_head{}".format(key, i)
            if head_key not in self.histories:
                raise ValueError(
                    "Key {} not among attention histories".format(head_key))

            alignments = tf.expand_dims(
                tf.transpose(self.histories[head_key], perm=[1, 2, 0]), -1)

            tf.summary.image("{}_head{}".format(self.name, i), alignments,
                             collections=["summary_att_plots"],
                             max_outputs=256)


class ScaledDotProdAttention(MultiHeadAttention):

    def __init__(self,
                 name: str,
                 keys_encoder: Attendable,
                 values_encoder: Attendable = None,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        MultiHeadAttention.__init__(
            self, name, 1, keys_encoder, values_encoder, dropout_keep_prob,
            save_checkpoint, load_checkpoint)
