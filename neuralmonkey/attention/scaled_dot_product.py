"""The scaled dot-product attention mechanism, defined in Vaswani et al. (2017)

See arxiv.org/abs/1706.03762
"""
import math
from typing import Tuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.nn.utils import dropout
from neuralmonkey.attention.base_attention import (
    BaseAttention, Attendable, get_attention_states, get_attention_mask,
    AttentionLoopStateTA, empty_attention_loop_state)


class ScaledDotProdAttention(BaseAttention):

    def __init__(self,
                 name: str,
                 keys_encoder: Attendable,
                 values_encoder: Attendable = None,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        BaseAttention.__init__(self, name, save_checkpoint, load_checkpoint)

        if values_encoder is None:
            values_encoder = keys_encoder

        self.attention_keys = get_attention_states(keys_encoder)
        self.attention_values = get_attention_states(values_encoder)
        self.attention_mask = get_attention_mask(keys_encoder)

        self.dropout_keep_prob = dropout_keep_prob

        self._dimension = self.attention_keys.get_shape()[-1].value
        self._scaling_factor = 1 / math.sqrt(self._dimension)

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: AttentionLoopStateTA,
                  step: tf.Tensor) -> Tuple[tf.Tensor, AttentionLoopStateTA]:

        # shape: batch, time (similarities of attention keys in batch and time
        # to the queries in the batch)
        dot_product = tf.reduce_sum(
            tf.expand_dims(query, 1) * self.attention_keys, [-1])
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
            tf.expand_dims(weights, -1) * self.attention_values, [1])

        next_contexts = loop_state.contexts.write(step, context)
        next_weights = loop_state.weights.write(step, weights)

        next_loop_state = AttentionLoopStateTA(
            contexts=next_contexts,
            weights=next_weights)

        return context, next_loop_state

    def initial_loop_state(self) -> AttentionLoopStateTA:
        return empty_attention_loop_state()

    def finalize_loop(self, key: str,
                      last_loop_state: AttentionLoopStateTA) -> None:
        self.histories[key] = last_loop_state.weights.stack()

    @property
    def context_vector_size(self) -> int:
        return self.attention_values.get_shape()[-1].value
