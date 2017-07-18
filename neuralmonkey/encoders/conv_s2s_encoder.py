"""From the paper Convolutional Sequence to Sequence Learning

http://arxiv.org/abs/1705.03122
"""

import tensorflow as tf
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.nn.projection import glu, linear
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.sequence import (EmbeddedSequence)


class ConvolutionalSentenceEncoder(ModelPart, Attentive):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: EmbeddedSequence,
                 conv_features: int,
                 encoder_layers: int,
                 kernel_width: int = 5,
                 dropout_keep_prob: float = 1.0,
                 attention_type: type = None,
                 attention_state_size: int = None,
                 attention_fertility: int = 3,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:

        assert check_argument_types()

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type,
                           attention_state_size=attention_state_size,
                           attention_fertility=attention_fertility)

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
    def states(self) -> tf.Tensor:
        convolutions = linear(self.ordered_embedded_inputs,
                              self.conv_features)
        for layer in range(self.encoder_layers):
            convolutions = self._residual_conv(
                convolutions, "encoder_conv_{}".format(layer))

        return convolutions + linear(self.ordered_embedded_inputs,
                                     self.conv_features)

    @tensor
    def encoded(self) -> tf.Tensor:
        # This state concatenation is not based on any paper, but was
        # tested empirically
        return tf.reduce_max(self.states, axis=1)

    @tensor
    def _attention_tensor(self) -> tf.Tensor:
        return dropout(self.states, self.dropout_keep_prob, self.train_mode)

    @tensor
    def _attention_mask(self) -> tf.Tensor:
        # TODO tohle je proti OOP prirode
        return self.input_sequence.mask

    @tensor
    def states_mask(self) -> tf.Tensor:
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


class SentenceEncoder(ConvolutionalSentenceEncoder):
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 conv_features: int,
                 encoder_layers: int,
                 kernel_width: int = 5,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 attention_type: type = None,
                 attention_fertility: int = 3,
                 attention_state_size: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a new instance of the sentence encoder. """

        # TODO Think this through.
        s_ckp = "input_{}".format(save_checkpoint) if save_checkpoint else None
        l_ckp = "input_{}".format(load_checkpoint) if load_checkpoint else None

        # TODO! Representation runner needs this. It is not simple to do it in
        # recurrent encoder since there may be more source data series. The
        # best way could be to enter the data_id parameter manually to the
        # representation runner
        self.data_id = data_id

        input_sequence = EmbeddedSequence(
            name="{}_input".format(name),
            vocabulary=vocabulary,
            data_id=data_id,
            embedding_size=embedding_size,
            max_length=max_input_len,
            save_checkpoint=s_ckp,
            load_checkpoint=l_ckp)

        ConvolutionalSentenceEncoder.__init__(
            self,
            name=name,
            input_sequence=input_sequence,
            conv_features=conv_features,
            encoder_layers=encoder_layers,
            kernel_width=kernel_width,
            dropout_keep_prob=dropout_keep_prob,
            attention_type=attention_type,
            attention_fertility=attention_fertility,
            attention_state_size=attention_state_size,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)
    # pylint: enable=too-many-arguments,too-many-locals
