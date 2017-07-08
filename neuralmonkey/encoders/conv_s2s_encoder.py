"""From a paper Convolutional Sequence to Sequence Learning

http://arxiv.org/abs/1705.03122
"""

import tensorflow as tf
import numpy as np
from typing import Any, List, Union, Type
from typeguard import check_argument_types

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.sequence import EmbeddedFactorSequence
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.nn.projection import glu, linear

# todo remove ipdb
import ipdb


class ConvolutionalSentenceEncoder(ModelPart, Attentive):

    def __init__(self,
                 name: str,
                 input_sequence: EmbeddedFactorSequence,
                 conv_features: int,
                 encoder_layers: int,
                 kernel_width: int = 5,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        #Attentive.__init__(self, None) # TODO attention

        assert check_argument_types()

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
        # TODO make this better
        if len(self.input_sequence.embedding_sizes) != 1:            
            raise ValueError(
                "Embedded sequence must have only one sequence.")

        log("Initializing convolutional seq2seq encoder, name {}"
            .format(self.name))

        with self.use_scope():
            convolutions = linear(self.ordered_embedded_inputs,
                                  self.conv_features)
            for layer in range(self.encoder_layers):
                convolutions = self.residual_conv(
                    convolutions, "encoder_conv_{}".format(layer))

            self.states = convolutions + linear(self.ordered_embedded_inputs,
                                                self.conv_features)
            #todo this is not based on any article
            self.encoded = tf.reduce_max(self.states, axis=1)

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
    def inputs(self):
        # shape (batch, time)
        return tf.placeholder(tf.int32, shape=[None, None],
                              name="conv_s2s_encoder_inputs")

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
        return self.input_sequence.data + ordering_additive

    def residual_conv(self, input, name):
        with tf.variable_scope(name):
            # initialized as described in the paper
            init_deviat = np.sqrt(4/self.conv_features)
            convolution_filters = tf.get_variable(
                "convolution_filters",
                [self.kernel_width, self.conv_features, 2*self.conv_features],
                initializer=tf.random_normal_initializer(stddev=init_deviat))

            bias = tf.get_variable(
                name="conv_bias",
                shape=[2 * self.conv_features],
                initializer=tf.zeros_initializer())

            conv = tf.nn.conv1d(input, convolution_filters, 1, "SAME") + bias

            return glu(conv) + input

    @tensor
    def train_mode(self):
        # scalar tensor
        return tf.placeholder(tf.bool, shape=[], name="mode_placeholder")

    @tensor
    def input_mask(self):
        # shape (batch, time)
        return tf.placeholder(tf.float32, shape=[None, None],
            name="conv_s2s_encoder_input_mask")

    @tensor
    def sentence_lengths(self) -> tf.Tensor:
        # shape (batch)
        return tf.to_int32(tf.reduce_sum(self.input_mask, 0))

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = self.input_sequence.feed_dict(dataset, train)
        fd[self.train_mode] = train

        return fd
