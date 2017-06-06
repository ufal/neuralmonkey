from typing import Tuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor

# pylint: disable=invalid-name
RNNCellTuple = Tuple[tf.contrib.rnn.RNNCell, tf.contrib.rnn.RNNCell]
# pylint: enable=invalid-name

RNN_CELL_TYPES = {
    "GRU": OrthoGRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell
}


# pylint: disable=too-many-instance-attributes
class SentenceEncoder(ModelPart, Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences. It uses a bidirectional RNN.

    This version of the encoder does not support factors. Should you
    want to use them, use FactoredEncoder instead.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 rnn_size: int,
                 attention_state_size: int = None,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 rnn_cell: str = "GRU",
                 attention_type: type = None,
                 attention_fertility: int = 3,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a new instance of the sentence encoder.

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
            attention_state_size: The size of the attention inner state. If
                None, use the size of the encoder hidden state. (defalult None)
            attention_fertility: Fertility parameter used with
                CoverageAttention (default 3).
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type,
                           attention_state_size=attention_state_size,
                           attention_fertility=attention_fertility)

        check_argument_types()

        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.data_id = data_id
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size

        self.max_input_len = max_input_len
        self.dropout_keep_prob = dropout_keep_prob
        self.rnn_cell_str = rnn_cell

        if self.max_input_len is not None and self.max_input_len <= 0:
            raise ValueError("Input length must be a positive integer.")

        if self.embedding_size <= 0:
            raise ValueError("Embedding size must be a positive integer.")

        if self.rnn_size <= 0:
            raise ValueError("RNN size must be a positive integer.")

        if self.rnn_cell_str not in RNN_CELL_TYPES:
            raise ValueError("RNN cell must be a either 'GRU' or 'LSTM'")
    # pylint: enable=too-many-arguments,too-many-locals

    # pylint: disable=no-self-use
    @tensor
    def inputs(self) -> tf.Tensor:
        return tf.placeholder(tf.int32, [None, None], "encoder_input")

    @tensor
    def input_mask(self) -> tf.Tensor:
        return tf.placeholder(tf.float32, [None, None], "encoder_padding")

    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, [], "train_mode")
    # pylint: enable=no-self-use

    @tensor
    def embedding_matrix(self) -> tf.Tensor:
        """A variable for embedding the input words.
        If parent encoder is specified, we reuse its embedding matrix
        """
        # NOTE the note from the decoder's embedding matrix function applies
        # here also:

        with tf.variable_scope("input_projection"):
            return tf.get_variable(
                "word_embeddings",
                [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

    @tensor
    def bidirectional_rnn(self) -> Tuple[Tuple[tf.Tensor, tf.Tensor],
                                         Tuple[tf.Tensor, tf.Tensor]]:
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
        embedded = dropout(embedded, self.dropout_keep_prob, self.train_mode)

        sequence_lengths = tf.to_int32(tf.reduce_sum(self.input_mask, 1))

        fw_cell, bw_cell = self._rnn_cells()  # type: RNNCellTuple
        return tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, embedded, sequence_length=sequence_lengths,
            dtype=tf.float32)

    @tensor
    def hidden_states(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[0], 2)
        # pylint: enable=unsubscriptable-object

    @tensor
    def encoded(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[1], 1)
        # pylint: enable=unsubscriptable-object

    @tensor
    def _attention_tensor(self) -> tf.Tensor:
        return dropout(self.hidden_states, self.dropout_keep_prob,
                       self.train_mode)

    @tensor
    def _attention_mask(self) -> tf.Tensor:
        # TODO tohle je proti OOP prirode
        return self.input_mask

    @tensor
    def states_mask(self) -> tf.Tensor:
        return self.input_mask

    def _rnn_cells(self) -> RNNCellTuple:
        """Return the graph template to for creating RNN memory cells"""
        return(RNN_CELL_TYPES[self.rnn_cell_str](self.rnn_size),
               RNN_CELL_TYPES[self.rnn_cell_str](self.rnn_size))

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
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
