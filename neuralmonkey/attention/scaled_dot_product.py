"""The scaled dot-product attention mechanism defined in Vaswani et al. (2017).

The attention energies are computed as dot products between the query vector
and the key vector. The query vector is scaled down by the square root of its
dimensionality. This attention function has no trainable parameters.

See arxiv.org/abs/1706.03762
"""
import math
from typing import Tuple, List, NamedTuple

import tensorflow as tf
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.nn.utils import dropout
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

    # pylint: disable=too-many-locals
    # TODO improve this code
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
            query_proj = tf.layers.dense(
                query, self._dimension, name="query_proj")
            keys_proj = tf.layers.dense(
                self.attention_keys, self._dimension, name="keys_proj")
            vals_proj = tf.layers.dense(
                self.attention_values, self._dimension, name="vals_proj")

            query_heads = tf.split(query_proj, self.n_heads, axis=1)
            keys_heads = tf.split(keys_proj, self.n_heads, axis=2)
            vals_heads = tf.split(vals_proj, self.n_heads, axis=2)

            head_contexts, head_weights = zip(*[
                self.attention_single_head(q, k, v)
                for q, k, v in zip(query_heads, keys_heads, vals_heads)])

            context = tf.layers.dense(
                tf.concat(head_contexts, -1), self._dimension,
                name="output_proj")

        next_contexts = loop_state.contexts.write(step, context)
        next_head_weights = [loop_state.head_weights[i].write(step,
                                                              head_weights[i])
                             for i in range(self.n_heads)]

        next_loop_state = MultiHeadLoopStateTA(
            contexts=next_contexts,
            head_weights=next_head_weights)

        return context, next_loop_state
    # pylint: enable=too-many-locals

    def attention_3d(self, query_3d: tf.Tensor,
                     masked: bool = False) -> tf.Tensor:

        if self.n_heads == 1:
            query_heads = [query_3d]
            keys_heads = [self.attention_keys]
            vals_heads = [self.attention_values]
        else:
            # Linearly project queries, keys and vals, then split
            # query_proj of shape batch, time(q), self._dimension (=q_channels)
            query_proj = tf.layers.dense(
                query_3d, self._dimension, name="query_proj")
            keys_proj = tf.layers.dense(
                self.attention_keys, self._dimension, name="keys_proj")
            vals_proj = tf.layers.dense(
                self.attention_values, self._dimension, name="vals_proj")

            query_heads = tf.split(query_proj, self.n_heads, axis=2)
            keys_heads = tf.split(keys_proj, self.n_heads, axis=2)
            vals_heads = tf.split(vals_proj, self.n_heads, axis=2)

        # head_contexts_3d, head_weights_3d = zip(*[
        head_contexts_3d, _ = zip(*[
            self.attention_single_head_3d(q, k, v, masked=masked)
            for q, k, v in zip(query_heads, keys_heads, vals_heads)])

        context_3d = tf.layers.dense(tf.concat(head_contexts_3d, -1),
                                     self._dimension, name="output_proj")

        # next_contexts = loop_state.contexts.write(step, context_3d)
        # next_head_weights = [
        #     loop_state.head_weights[i].write(step, head_weights_3d[i])
        #     for i in range(self.n_heads)]

        # next_loop_state = MultiHeadLoopState3DTA(
        #     contexts=next_contexts,
        #     head_weights=next_head_weights)

        return context_3d

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

    def attention_single_head_3d(self, query: tf.Tensor,
                                 keys: tf.Tensor,
                                 values: tf.Tensor,
                                 masked: bool = False) -> Tuple[tf.Tensor,
                                                                tf.Tensor]:
        # Shapes:
        # query:  batch, time(q), k_channels
        # keys:   batch, time(k), k_channels
        # values: batch, time(k), v_channels
        # Outputs:
        # context: batch, time(q), v_channels
        # weights: batch, time(q), time(k)

        # Scale first:
        query_scaled = query * self._scaling_factor

        # For dot-product, we use matrix multiplication
        # shape: batch, time(q), time(k) (k_channels is the matmul axis)
        energies = tf.matmul(query_scaled, keys, transpose_b=True)

        # To protect the attention from looking ahead of time, we must
        # replace the energies of future keys with negative infinity
        # We use lower triangular matrix and basic tf where tricks
        if masked:
            triangular_mask = tf.matrix_band_part(
                tf.ones_like(energies), -1, 0)
            energies = tf.where(
                tf.equal(triangular_mask, 1),
                energies, tf.fill(energies.shape, -np.inf))

        # Softmax along the last axis
        # shape: batch, time(q), time(k)
        weights_3d = tf.nn.softmax(energies)

        if self.attention_mask is not None:
            # attention mask shape: batch, time(k)
            # weights_all shape: batch, time(q), time(k)
            weights_all = weights_3d * tf.expand_dims(self.attention_mask, 1)
            # normalization along time(k)
            # norm shape: batch, time(q), 1
            norm = tf.reduce_sum(weights_all, 2, keep_dims=True) + 1e-8
            weights_3d = weights_all / norm

        # apply dropout to the weights (Attention Dropout)
        weights_3d = dropout(
            weights_3d, self.dropout_keep_prob, self.train_mode)

        # sum up along the time(k) axis, weigh values along the v_channels axis

        # 1. expand weights_3d to shape batch, time(q), time(k), 1
        # 2. expand values to shape     batch, 1, time(k), v_channels
        # 3. element-wise multiplication broadcasts that to
        #    shape: batch, time(q), time(k), v_channels
        # 4. sum along the time(k) axis
        context_3d = tf.reduce_sum(
            tf.expand_dims(weights_3d, 3) * tf.expand_dims(values, 1), 2)

        return context_3d, weights_3d

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
