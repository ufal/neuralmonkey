"""Implementation of the encoder of the Transformer model.

Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762
"""
# pylint: disable=unused-import
from typing import Set, Optional, List
# pylint: enable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.attention.scaled_dot_product import attention
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import FeedDict, ModelPart
from neuralmonkey.model.stateful import (TemporalStateful,
                                         TemporalStatefulWithOutput)
from neuralmonkey.nn.utils import dropout


def position_signal(dimension: int, length: tf.Tensor) -> tf.Tensor:
    # code simplified and copied from github.com/tensorflow/tensor2tensor

    # TODO write this down on a piece of paper and understand the code and
    # compare it to the paper
    positions = tf.to_float(tf.range(length))

    num_timescales = dimension // 2
    log_timescale_increment = 4 / (tf.to_float(num_timescales) - 1)

    inv_timescales = tf.exp(tf.to_float(tf.range(num_timescales))
                            * -log_timescale_increment)

    scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(
        inv_timescales, 0)

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(dimension, 2)]])
    signal = tf.reshape(signal, [1, length, dimension])

    return signal


class TransformerLayer(TemporalStateful):
    def __init__(self, states: tf.Tensor, mask: tf.Tensor) -> None:
        self._states = states
        self._mask = mask

    @property
    def temporal_states(self) -> tf.Tensor:
        return self._states

    @property
    def temporal_mask(self) -> tf.Tensor:
        return self._mask


class TransformerEncoder(ModelPart, TemporalStatefulWithOutput):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: TemporalStateful,
                 ff_hidden_size: int,
                 depth: int,
                 n_heads: int,
                 dropout_keep_prob: float = 1.0,
                 attention_dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create an encoder of the Transformer model.

        Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762

        Arguments:
            input_sequence: Embedded input sequence.
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects.
            dropout_keep_prob: Probability of keeping a value during dropout.

        Keyword arguments:
            ff_hidden_size: Size of the feedforward sublayers.
            n_heads: Number of the self-attention heads.
            depth: Number of sublayers.
            attention_dropout_keep_prob: Probability of keeping a value
                during dropout on the attention output.
        """
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.input_sequence = input_sequence
        self.model_dimension = self.input_sequence.dimension
        self.ff_hidden_size = ff_hidden_size
        self.depth = depth
        self.n_heads = n_heads
        self.dropout_keep_prob = dropout_keep_prob
        self.attention_dropout_keep_prob = attention_dropout_keep_prob

        if self.depth <= 0:
            raise ValueError("Depth must be a positive integer.")

        if self.ff_hidden_size <= 0:
            raise ValueError("Feed forward hidden size must be a "
                             "positive integer.")

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        if (self.attention_dropout_keep_prob <= 0.0
                or self.attention_dropout_keep_prob > 1.0):
            raise ValueError("Dropout keep prob for attn must be in (0,1].")

        self.train_mode = tf.placeholder(tf.bool, [], "train_mode")
        log("Output op: {}".format(self.output))
    # pylint: enable=too-many-arguments

    @tensor
    def output(self) -> tf.Tensor:
        return tf.reduce_sum(self.temporal_states, axis=1)

    @tensor
    def encoder_inputs(self) -> tf.Tensor:
        length = tf.shape(self.input_sequence.temporal_states)[1]
        signal = position_signal(self.model_dimension, length)
        return dropout(self.input_sequence.temporal_states + signal,
                       self.dropout_keep_prob, self.train_mode)

    def self_attention(
            self, level: int, prev_layer: TransformerLayer) -> tf.Tensor:

        with tf.variable_scope("self_attention_{}".format(level)):
            self_context, _ = attention(
                queries=prev_layer.temporal_states,
                keys=prev_layer.temporal_states,
                values=prev_layer.temporal_states,
                keys_mask=prev_layer.temporal_mask,
                num_heads=self.n_heads,
                dropout_callback=lambda x: dropout(
                    x, self.attention_dropout_keep_prob, self.train_mode))

            return dropout(
                self_context, self.dropout_keep_prob, self.train_mode)

    def layer(self, level: int) -> TransformerLayer:
        # Recursive implementation. Outputs of the zeroth layer are normalized
        # inputs.
        if level == 0:
            norm_inputs = tf.contrib.layers.layer_norm(
                self.encoder_inputs, begin_norm_axis=2)
            return TransformerLayer(norm_inputs, self.temporal_mask)

        # Compute the outputs of the previous layer
        prev_layer = self.layer(level - 1)

        # Run self-attention
        self_context = self.self_attention(level, prev_layer)

        # Residual connections + layer normalization
        ff_input = tf.contrib.layers.layer_norm(
            self_context + prev_layer.temporal_states, begin_norm_axis=2)

        # Feed-forward network hidden layer + ReLU + dropout
        ff_hidden = tf.layers.dense(
            ff_input, self.ff_hidden_size, activation=tf.nn.relu,
            name="ff_hidden_{}".format(level))
        ff_hidden_drop = dropout(
            ff_hidden, self.dropout_keep_prob, self.train_mode)

        # Feed-forward output projection + dropout
        ff_output = tf.layers.dense(
            ff_hidden_drop, self.model_dimension,
            name="ff_out_{}".format(level))
        ff_output = dropout(ff_output, self.dropout_keep_prob, self.train_mode)

        # Residual connections + layer normalization
        output_states = tf.contrib.layers.layer_norm(
            ff_input + ff_output, begin_norm_axis=2)

        return TransformerLayer(states=output_states, mask=self.temporal_mask)

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return self.layer(self.depth).temporal_states

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.input_sequence.temporal_mask

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.train_mode: train}
