from typing import Optional, Tuple, List, Any

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.encoders.sentence_encoder import RNNCellTuple
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.nn.highway import highway
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary


class SentenceCNNEncoder(ModelPart, Attentive):
    """Encoder processing a sentence using a CNN then
    running a bidirectional RNN on the result.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-statements
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 segment_size: int,
                 highway_depth: int,
                 rnn_size: int,
                 filters: List[Tuple[int, int]],
                 max_input_len: Optional[int]=None,
                 dropout_keep_prob: float=1.0,
                 attention_type: Optional[Any]=None,
                 attention_fertility: int=3,
                 use_noisy_activations: bool=False,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        """Create a new instance of the sentence encoder.

        Based on: Jason Lee, Kyunghyun Cho, Thomas Hofmann: Fully
        Character-Level Neural Machine Translation without Explicit
        Segmentation (https://arxiv.org/pdf/1610.03017.pdf)

        Arguments:
            vocabulary: Input vocabulary
            data_id: Identifier of the data series fed to this encoder
            name: An unique identifier for this encoder
            max_input_len: Maximum length of an encoded sequence
            embedding_size: The size of the embedding vector assigned
                to each word
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
        Attentive.__init__(
            self, attention_type, attention_fertility=attention_fertility)

        assert check_argument_types()

        self.vocabulary = vocabulary
        self.data_id = data_id

        self.max_input_len = max_input_len
        self.embedding_size = embedding_size
        self.segment_size = segment_size
        self.highway_depth = highway_depth
        self.rnn_size = rnn_size
        self.filters = filters
        self.dropout_keep_p = dropout_keep_prob
        self.use_noisy_activations = use_noisy_activations

        if max_input_len is not None and max_input_len <= 0:
            raise ValueError("Input length must be a positive integer.")

        log("Initializing sentence encoder, name: '{}'"
            .format(self.name))

        with tf.variable_scope(self.name):
            self._create_input_placeholders()
            with tf.variable_scope('input_projection'):
                self._create_embedding_matrix()
                embedded_inputs = self._embed(self.inputs)  # type: tf.Tensor
                self.embedded_inputs = embedded_inputs

            # CNN Network
            pooled_outputs = []
            for filter_size, num_filters in self.filters:
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, num_filters]
                    w_filter = tf.get_variable(
                        "conv_W", filter_shape,
                        initializer=tf.random_uniform_initializer(-0.5, 0.5))
                    b_filter = tf.get_variable(
                        "conv_bias", [num_filters],
                        initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv1d(
                        embedded_inputs,
                        w_filter,
                        stride=1,
                        padding="VALID",
                        name="conv")

                    # Apply nonlinearity
                    conv_relu = tf.nn.relu(tf.nn.bias_add(conv, b_filter))

                    # Max-pooling over the output segments
                    expanded_conv_relu = tf.expand_dims(conv_relu, -1)
                    pooled = tf.nn.max_pool(
                        expanded_conv_relu,
                        ksize=[1, self.segment_size, 1, 1],
                        strides=[1, self.segment_size, 1, 1],
                        padding="SAME")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            self.cnn_encoded = tf.concat(pooled_outputs, axis=2)
            self.cnn_encoded = tf.squeeze(self.cnn_encoded, [3])

            # Highway Network
            batch_size = tf.shape(self.cnn_encoded)[0]
            # pylint: disable=no-member
            cnn_out_size = self.cnn_encoded.get_shape().as_list()[-1]
            highway_layer = tf.reshape(self.cnn_encoded, [-1, cnn_out_size])
            for i in range(self.highway_depth):
                highway_layer = highway(
                    highway_layer,
                    scope=("highway_layer_%s" % i))
            highway_layer = tf.reshape(
                highway_layer,
                [batch_size, -1, cnn_out_size])

            # BiRNN Network
            fw_cell, bw_cell = self.rnn_cells()  # type: RNNCellTuple
            seq_lens = tf.ceil(tf.divide(
                self.sentence_lengths,
                self.segment_size))
            seq_lens = tf.cast(seq_lens, tf.int32)
            outputs_bidi_tup, encoded_tup = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, highway_layer,
                sequence_length=seq_lens,
                dtype=tf.float32)

            self.hidden_states = tf.concat(outputs_bidi_tup, 2)

            with tf.variable_scope('attention_tensor'):
                self.__attention_tensor = self._dropout(
                    self.hidden_states)

            self.encoded = tf.concat(encoded_tup, 1)

        log("Sentence encoder initialized")

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    @property
    def _attention_mask(self):
        # TODO tohle je proti OOP prirode
        return self.input_mask

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, shape=[],
                                         name="mode_placeholder")

        self.inputs = tf.placeholder(tf.int32,
                                     shape=[None, None],
                                     name="encoder_input")

        self.input_mask = tf.placeholder(
            tf.float32, shape=[None, None],
            name="encoder_padding")

        self.sentence_lengths = tf.to_int32(
            tf.reduce_sum(self.input_mask, 1))

    def _create_embedding_matrix(self):
        """Create variables and operations for embedding the input words.

        If parent encoder is specified, we reuse its embedding matrix
        """
        # NOTE the note from the decoder's embedding matrix function applies
        # here also
        self.embedding_matrix = tf.get_variable(
            "word_embeddings", [self.vocabulary_size, self.embedding_size],
            initializer=tf.random_normal_initializer(stddev=0.01))

    def _dropout(self, variable: tf.Tensor) -> tf.Tensor:
        """Perform dropout on a variable

        Arguments:
            variable: The variable to be dropped out

        Returns:
            The dropped value of the variable
        """
        if self.dropout_keep_p == 1.0:
            return variable

        # TODO as soon as TF.12 is out, remove this line and use train_mode
        # directly
        train_mode_batch = tf.fill(tf.shape(variable)[:1], self.train_mode)
        dropped_value = tf.nn.dropout(variable, self.dropout_keep_p)
        return tf.where(train_mode_batch, dropped_value, variable)

    def _embed(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self._dropout(embedded)

    def rnn_cells(self) -> RNNCellTuple:
        """Return the graph template to for creating RNN memory cells"""

        if self.use_noisy_activations:
            return(NoisyGRUCell(self.rnn_size, self.train_mode),
                   NoisyGRUCell(self.rnn_size, self.train_mode))

        return (OrthoGRUCell(self.rnn_size),
                OrthoGRUCell(self.rnn_size))

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary with the encoder inputs.

        Encoder input placeholders:
            ``encoder_input``: Stores indices to the vocabulary,
                shape (batch, time)
            ``encoder_padding``: Stores the padding (ones and zeros,
                indicating valid words and positions after the end
                of sentence, shape (batch, time)
            ``train_mode``: Boolean scalar specifying the mode (train
                vs runtime)

        Arguments:
            dataset: The dataset to use
            train: Boolean flag telling whether it is training time
        """
        # pylint: disable=invalid-name
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train
        sentences = dataset.get_series(self.data_id)

        vectors, paddings = self.vocabulary.sentences_to_tensor(
            list(sentences), self.max_input_len, pad_to_max_len=False,
            train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self.input_mask] = list(zip(*paddings))

        return fd
