# tests: lint, mypy

from typing import Optional, Dict, Any, Tuple

import tensorflow as tf

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.logging import log
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary

# pylint: disable=invalid-name
AttType = Any  # Type[] or union of types do not work here
FeedDict = Dict[tf.Tensor, Any]
RNNCellTuple = Tuple[tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.RNNCell]
# pylint: enable=invalid-name


# pylint: disable=too-many-instance-attributes
class SentenceEncoder(Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences. It uses a bidirectional RNN.

    This version of the encoder does not support factors. Should you
    want to use them, use FactoredEncoder instead.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 vocabulary: Vocabulary,
                 data_id: str,
                 name: str,
                 max_input_len: int,
                 embedding_size: int,
                 rnn_size: int,
                 dropout_keep_prob: float=1.0,
                 attention_type: Optional[AttType]=None,
                 attention_fertility: int=3,
                 use_noisy_activations: bool=False,
                 parent_encoder: Optional["SentenceEncoder"]=None) -> None:
        """Createes a new instance of the sentence encoder

        Arguments:
            vocabulary: Input vocabulary
            data_id: Identifier of the data series fed to this encoder
            name: An unique identifier for this encoder
            max_input_len: Maximum length of an encoded sequence
            embedding_size: The size of the embedding vector assigned
                to each word
            rnn_size: The size of the encoder's hidden state. Note
                that the actual encoder output state size will be
                twice as long because it is the result of
                concatenation of forward and backward hidden states.

        Keyword arguments:
            dropout_keep_prob: The dropout keep probability
                (default 1.0)
            attention_type: The class that is used for creating
                attention mechanism (default None)
            attention_fertility: Fertility parameter used with
                CoverageAttention (default 3).
        """
        super().__init__(
            attention_type, attention_fertility=attention_fertility)

        self.vocabulary = vocabulary
        self.data_id = data_id
        self.name = name

        self.max_input_len = max_input_len
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.dropout_keep_p = dropout_keep_prob
        self.use_noisy_activations = use_noisy_activations
        self.parent_encoder = parent_encoder

        log("Initializing sentence encoder, name: '{}'"
            .format(self.name))

        with tf.variable_scope(self.name):
            self._create_input_placeholders()
            self._create_embedding_matrix()

            embedded_inputs = self._embed(self.inputs)  # type: tf.Tensor
            fw_cell, bw_cell = self.rnn_cells()  # type: RNNCellTuple

            outputs_bidi_tup, encoded_tup = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, embedded_inputs, self.sentence_lengths,
                dtype=tf.float32)

            self.__attention_tensor = tf.concat(2, outputs_bidi_tup)
            self.__attention_tensor = self._dropout(self.__attention_tensor)

            self.encoded = tf.concat(1, encoded_tup)

        log("Sentence encoder initialized")

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    @property
    def _attention_mask(self):
        return self._input_mask

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, shape=[],
                                         name="mode_placeholder")

        self.inputs = tf.placeholder(tf.int32,
                                     shape=[None, self.max_input_len],
                                     name="encoder_input")

        self._input_mask = tf.placeholder(
            tf.float32, shape=[None, self.max_input_len],
            name="encoder_padding")

        self.sentence_lengths = tf.to_int32(
            tf.reduce_sum(self._input_mask, 1))

    def _create_embedding_matrix(self):
        """Create variables and operations for embedding the input words.

        If parent encoder is specified, we reuse its embedding matrix
        """
        # NOTE the note from the decoder's embedding matrix function applies
        # here also
        if self.parent_encoder is not None:
            self.embedding_matrix = self.parent_encoder.embedding_matrix
        else:
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
        return tf.select(train_mode_batch, dropped_value, variable)

    def _embed(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self._dropout(embedded)

    def rnn_cells(self) -> RNNCellTuple:
        """Return the graph template to for creating RNN memory cells"""

        if self.parent_encoder is not None:
            return self.parent_encoder.rnn_cells()

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
            list(sentences), self.max_input_len, train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self._input_mask] = list(zip(*paddings))

        return fd
