"""Implementation of the encoder of the Transformer model as described in
Vaswani et al. (2017).

See arxiv.org/abs/1706.03762
"""
# pylint: disable=unused-import
from typing import Set, Optional, List
# pylint: enable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.attention.scaled_dot_product import MultiHeadAttention
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import FeedDict, ModelPart
from neuralmonkey.model.sequence import Sequence
from neuralmonkey.model.stateful import (TemporalStateful,
                                         TemporalStatefulWithOutput)


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
                 input_sequence: Sequence,
                 ff_hidden_size: int,
                 depth: int,
                 n_heads: int,
                 dropout_keep_prob: float = 1.0,
                 attention_dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.input_sequence = input_sequence
        self.dimension = self.input_sequence.dimension
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
        self.self_attentions = [None for _ in range(self.depth)] \
            # type: List[Optional[MultiHeadAttention]]

        log("Output op: {}".format(self.output))
    # pylint: enable=too-many-arguments

    @tensor
    def output(self) -> tf.Tensor:
        return tf.reduce_sum(self.temporal_states, axis=1)

    @tensor
    def encoder_inputs(self) -> tf.Tensor:
        length = tf.shape(self.input_sequence.data)[1]
        signal = position_signal(self.dimension, length)
        return self.input_sequence.data + signal

    def layer(self, level: int) -> TransformerLayer:

        # Recursive implementation. Outputs of the zeroth layer are the inputs
        if level == 0:
            return TransformerLayer(self.encoder_inputs,
                                    self.temporal_mask)

        # Compute the outputs of the previous layer
        prev_layer = self.layer(level - 1)

        # Compute the outputs of this layer
        s_ckp = "enc_self_att_{}_{}".format(
            level, self._save_checkpoint) if self._save_checkpoint else None
        l_ckp = "enc_self_att_{}_{}".format(
            level, self._load_checkpoint) if self._load_checkpoint else None

        att = MultiHeadAttention(
            name="self_att_{}".format(level),
            n_heads=self.n_heads,
            keys_encoder=prev_layer,
            values_encoder=prev_layer,
            dropout_keep_prob=self.attention_dropout_keep_prob,
            save_checkpoint=s_ckp,
            load_checkpoint=l_ckp)

        # TODO generalize att work with 3D queries as default
        with tf.variable_scope("att_level_{}".format(level)):
            self_att_result = att.attention_3d(prev_layer.temporal_states)
            self.self_attentions[level - 1] = att

        ff_input = tf.contrib.layers.layer_norm(
            self_att_result + prev_layer.temporal_states)

        ff_hidden = tf.layers.dense(ff_input, self.ff_hidden_size,
                                    activation=tf.nn.relu,
                                    name="ff_hidden_{}".format(level))

        ff_output = tf.layers.dense(ff_hidden, self.dimension,
                                    name="ff_out_{}".format(level))

        output_states = tf.contrib.layers.layer_norm(ff_output + ff_input)

        return TransformerLayer(states=output_states, mask=self.temporal_mask)

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return self.layer(self.depth).temporal_states

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.input_sequence.mask

    def get_dependencies(self) -> Set[ModelPart]:
        assert all(self.self_attentions)

        dependencies = ModelPart.get_dependencies(self)

        for att in self.self_attentions:
            dependencies = dependencies.union(att.get_dependencies())

        return dependencies

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.train_mode: train}
