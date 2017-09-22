"""From the paper Convolutional Sequence to Sequence Learning

http://arxiv.org/abs/1705.03122
"""

import tensorflow as tf
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.nn.projection import glu, linear
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.model.stateful import TemporalStatefulWithOutput


class SentenceEncoder(ModelPart, TemporalStatefulWithOutput):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: EmbeddedSequence,
                 conv_features: int,
                 encoder_layers: int,
                 kernel_width: int = 5,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.input_sequence = input_sequence
        self.encoder_layers = encoder_layers
        self.conv_features = conv_features
        self.kernel_width = kernel_width
        self.dropout_keep_prob = dropout_keep_prob

        if conv_features <= 0:
            raise ValueError("Number of features must be a positive integer.")
        if encoder_layers <= 0:
            raise ValueError(
                "Number of encoder layers must be a positive integer.")

        log("Initializing convolutional seq2seq encoder, name {}"
            .format(self.name))
    # pylint: enable=too-many-arguments

    @tensor
    def temporal_states(self) -> tf.Tensor:
        convolutions = linear(self.ordered_embedded_inputs,
                              self.conv_features,
                              scope="order_and_embed")
        for layer in range(self.encoder_layers):
            convolutions = self._residual_conv(
                convolutions, "encoder_conv_{}".format(layer))

        return convolutions + linear(self.ordered_embedded_inputs,
                                     self.conv_features,
                                     scope="input_to_final_state")

    @tensor
    def output(self) -> tf.Tensor:
        # This state concatenation is not based on any paper, but was
        # tested empirically
        return tf.reduce_max(self.temporal_states, axis=1)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.input_sequence.mask

    @tensor
    def order_embeddings(self) -> tf.Tensor:
        # initialization in the same way as in original CS2S implementation
        with tf.variable_scope("input_projection"):
            return tf.get_variable(
                "order_embeddings", [self.input_sequence.max_length,
                                     self.input_sequence.embedding_sizes[0]],
                initializer=tf.random_normal_initializer(stddev=0.1))

    @tensor
    def ordered_embedded_inputs(self) -> tf.Tensor:
        # shape (batch, time, embedding size)
        ordering_additive = tf.expand_dims(self.order_embeddings, 0)
        batch_max_len = tf.shape(self.input_sequence.data)[1]
        clipped_ordering_embed = ordering_additive[:, :batch_max_len, :]

        return self.input_sequence.data + clipped_ordering_embed

    def _residual_conv(self, input_signals: tf.Tensor, name: str):
        with tf.variable_scope(name):
            # initialized as described in the paper
            init_deviat = np.sqrt(4 / self.conv_features)
            convolution_filters = tf.get_variable(
                "convolution_filters",
                [self.kernel_width, self.conv_features,
                 2 * self.conv_features],
                initializer=tf.random_normal_initializer(stddev=init_deviat))

            bias = tf.get_variable(
                name="conv_bias",
                shape=[2 * self.conv_features],
                initializer=tf.zeros_initializer())

            conv = (tf.nn.conv1d(input_signals, convolution_filters, 1, "SAME")
                    + bias)

            return glu(conv) + input_signals

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self):
        # scalar tensor
        return tf.placeholder(tf.bool, shape=[], name="mode_placeholder")
    # pylint: enable=no-self-use

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = self.input_sequence.feed_dict(dataset, train)
        fd[self.train_mode] = train

        return fd
