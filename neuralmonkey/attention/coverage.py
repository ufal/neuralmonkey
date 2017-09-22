"""Coverage attention introduced in Tu et al. (2016)
See arxiv.org/abs/1601.04811

The CoverageAttention class inherites from the basic feed-forward attention
introduced by Bahdanau et al. (2015)
"""
from typing import Union

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.feed_forward import Attention
from neuralmonkey.model.stateful import TemporalStateful, SpatialStateful


class CoverageAttention(Attention):
    def __init__(self,
                 name: str,
                 encoder: Union[TemporalStateful, SpatialStateful],
                 dropout_keep_prob: float = 1.0,
                 state_size: int = None,
                 max_fertility: int = 5,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        Attention.__init__(self, name, encoder, dropout_keep_prob, state_size,
                           save_checkpoint, load_checkpoint)

        self.max_fertility = max_fertility

        self.coverage_weights = tf.get_variable(
            "coverage_matrix", [1, 1, 1, self.state_size])
        self.fertility_weights = tf.get_variable(
            "fertility_matrix", [1, 1, self.context_vector_size])

        self.fertility = 1e-8 + self.max_fertility * tf.sigmoid(
            tf.reduce_sum(self.fertility_weights * self.attention_states, [2]))

    def get_energies(self, y: tf.Tensor, weights_in_time: tf.TensorArray):
        weight_sum = tf.cond(
            tf.greater(weights_in_time.size(), 0),
            lambda: tf.reduce_sum(weights_in_time.stack(), axis=0),
            lambda: 0.0)

        coverage = weight_sum / self.fertility * self.attention_mask
        logits = tf.reduce_sum(
            self.similarity_bias_vector * tf.tanh(
                self.hidden_features + y + self.coverage_weights *
                tf.expand_dims(tf.expand_dims(coverage, -1), -1)),
            [2, 3])

        return logits
