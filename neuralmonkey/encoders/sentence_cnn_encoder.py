"""Encoder for sentences withou explicit segmentation."""

from typing import Optional, Tuple, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.encoders.recurrent import RNNCellTuple
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.sequence import Sequence
from neuralmonkey.model.stateful import TemporalStatefulWithOutput
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.nn.highway import highway
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor


# pylint: disable=too-many-instance-attributes
class SentenceCNNEncoder(ModelPart, TemporalStatefulWithOutput):
    """Encoder processing a sentence using a CNN then
    running a bidirectional RNN on the result.

    Based on: Jason Lee, Kyunghyun Cho, Thomas Hofmann: Fully
    Character-Level Neural Machine Translation without Explicit
    Segmentation (https://arxiv.org/pdf/1610.03017.pdf)
    """

    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-statements
    def __init__(self,
                 name: str,
                 input_sequence: Sequence,
                 segment_size: int,
                 highway_depth: int,
                 rnn_size: int,
                 filters: List[Tuple[int, int]],
                 dropout_keep_prob: float = 1.0,
                 use_noisy_activations: bool = False,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        """Create a new instance of the sentence encoder.

        Arguments:
            name: An unique identifier for this encoder
            segment_size: The size of the segments over which we apply
                max-pooling.
            highway_depth: Depth of the highway layer.
            rnn_size: The size of the encoder's hidden state. Note
                that the actual encoder output state size will be
                twice as long because it is the result of
                concatenation of forward and backward hidden states.
            filters: Specification of CNN filters. It is a list of tuples
                specifying the filter size and number of channels.

        Keyword arguments:
            dropout_keep_prob: The dropout keep probability
                (default 1.0)
            attention_type: The class that is used for creating
                attention mechanism (default None)
            attention_fertility: Fertility parameter used with
                CoverageAttention (default 3).
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        self.input_sequence = input_sequence
        self.segment_size = segment_size
        self.highway_depth = highway_depth
        self.rnn_size = rnn_size
        self.filters = filters
        self.dropout_keep_prob = dropout_keep_prob
        self.use_noisy_activations = use_noisy_activations

        if dropout_keep_prob <= 0. or dropout_keep_prob > 1.:
            raise ValueError(
                ("Dropout keep probability must be "
                 "in (0; 1], was {}").format(dropout_keep_prob))

        if rnn_size <= 0:
            raise ValueError("RNN size must be a positive integer.")

        if highway_depth <= 0:
            raise ValueError("Highway depth must be a positive integer.")

        if segment_size <= 0:
            raise ValueError("Segment size be a positive integer.")

        if not filters:
            raise ValueError("You must specify convolutional filters.")

        for filter_size, num_filters in self.filters:
            if filter_size <= 0:
                raise ValueError("Filter size must be a positive integer.")
            if num_filters <= 0:
                raise ValueError("Number of filters must be a positive int.")

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, shape=[], name="train_mode")
    # pylint: enable=no-self-use

    @tensor
    def cnn_encoded(self) -> tf.Tensor:
        """1D convolution with max-pool that processing characters."""
        dropped_inputs = dropout(self.input_sequence.data,
                                 self.dropout_keep_prob, self.train_mode)

        pooled_outputs = []
        for filter_size, num_filters in self.filters:
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.input_sequence.dimension,
                                num_filters]
                w_filter = tf.get_variable(
                    "conv_W", filter_shape,
                    initializer=tf.random_uniform_initializer(-0.5, 0.5))
                b_filter = tf.get_variable(
                    "conv_bias", [num_filters],
                    initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv1d(
                    dropped_inputs,
                    w_filter,
                    stride=1,
                    padding="SAME",
                    name="conv")

                # Apply nonlinearity
                conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b_filter))

                # Max-pooling over the output segments
                expanded_conv_relu = tf.expand_dims(conv_relu, -1)
                pooled = tf.nn.max_pool(
                    expanded_conv_relu,
                    ksize=[1, self.segment_size, 1, 1],
                    strides=[1, self.segment_size, 1, 1],
                    padding="SAME",
                    name="maxpool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        concat = tf.concat(pooled_outputs, axis=2)
        return tf.squeeze(concat, [3])

    @tensor
    def highway_layer(self) -> tf.Tensor:
        """Highway net projection following the CNN."""
        batch_size = tf.shape(self.cnn_encoded)[0]
        # pylint: disable=no-member
        cnn_out_size = self.cnn_encoded.get_shape().as_list()[-1]
        highway_layer = tf.reshape(self.cnn_encoded, [-1, cnn_out_size])
        for i in range(self.highway_depth):
            highway_layer = highway(
                highway_layer,
                scope=("highway_layer_%s" % i))
        return tf.reshape(
            highway_layer,
            [batch_size, -1, cnn_out_size])

    @tensor
    def bidirectional_rnn(self) -> Tuple[Tuple[tf.Tensor, tf.Tensor],
                                         Tuple[tf.Tensor, tf.Tensor]]:
        # BiRNN Network
        fw_cell, bw_cell = self.rnn_cells()  # type: RNNCellTuple
        seq_lens = tf.ceil(tf.divide(
            self.input_sequence.lengths,
            self.segment_size))
        seq_lens = tf.cast(seq_lens, tf.int32)
        return tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, self.highway_layer,
            sequence_length=seq_lens,
            dtype=tf.float32)

    @tensor
    def temporal_states(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[0], 2)
        # pylint: enable=unsubscriptable-object

    @tensor
    def output(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[1], 1)
        # pylint: enable=unsubscriptable-object

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        expanded = tf.expand_dims(
            tf.expand_dims(self.input_sequence.mask, -1),
            -1)
        pooled = tf.nn.max_pool(
            expanded,
            ksize=[1, self.segment_size, 1, 1],
            strides=[1, self.segment_size, 1, 1],
            padding="SAME")
        return tf.squeeze(pooled, [2, 3])

    def rnn_cells(self) -> RNNCellTuple:
        """Return the graph template to for creating RNN memory cells"""

        if self.use_noisy_activations:
            return(NoisyGRUCell(self.rnn_size, self.train_mode),
                   NoisyGRUCell(self.rnn_size, self.train_mode))

        return (OrthoGRUCell(self.rnn_size),
                OrthoGRUCell(self.rnn_size))

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary with the encoder inputs.

        Arguments:
            dataset: The dataset to use
            train: Boolean flag telling whether it is training time
        """
        fd = self.input_sequence.feed_dict(dataset, train)
        fd[self.train_mode] = train

        return fd
