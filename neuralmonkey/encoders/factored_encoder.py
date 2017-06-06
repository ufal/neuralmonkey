from typing import Tuple, List, Dict

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
class FactoredEncoder(ModelPart, Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences. It uses a bidirectional RNN.

    This version of the encoder supports factors.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 rnn_size: int,
                 attention_state_size: int = None,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 rnn_cell: str = "GRU",
                 attention_type: type = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Construct a new instance of the factored encoder.

        Args:
            vocabularies: List of vocabularies indexed
            data_ids: List of data series IDs
            embedding_sizes: List of embedding sizes for each data series
            name: The name for this encoder.
            rnn_size: The size of the hidden state

        Keyword arguments:
            attention_state_size: The size of the attention hidden state
            max_input_len: Maximum input length (longer sequences are trimmed)
            attention_type: The attention to use.
            dropout_keep_prob: Dropout keep probability
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type,
                           attention_state_size=attention_state_size)

        check_argument_types()

        self.vocabularies = vocabularies
        self.vocabulary_sizes = [len(voc) for voc in self.vocabularies]
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes
        self.rnn_size = rnn_size

        self.max_input_len = max_input_len
        self.dropout_keep_prob = dropout_keep_prob
        self.rnn_cell_str = rnn_cell

        if not (len(self.data_ids)
                == len(self.vocabularies)
                == len(self.embedding_sizes)):
            raise ValueError("data_ids, vocabularies, and embedding_sizes "
                             "lists need to have the same length")

        if max_input_len is not None and max_input_len <= 0:
            raise ValueError("Input length must be a positive integer.")

        if any([esize <= 0 for esize in embedding_sizes]):
            raise ValueError("Embedding size must be a positive integer.")

        if rnn_size <= 0:
            raise ValueError("RNN size must be a positive integer.")

        if self.rnn_cell_str not in RNN_CELL_TYPES:
            raise ValueError("RNN cell must be a either 'GRU' or 'LSTM'")
    # pylint: enable=too-many-arguments,too-many-locals

    @tensor
    def input_factors(self) -> Dict[str, tf.Tensor]:
        return {name: tf.placeholder(tf.int32, [None, None],
                                     "encoder_factor_{}".format(name))
                for name in self.data_ids}

    # pylint: disable=no-self-use
    @tensor
    def input_mask(self) -> tf.Tensor:
        return tf.placeholder(tf.float32, [None, None], "encoder_padding")

    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, [], "train_mode")
    # pylint: enable=no-self-use

    @tensor
    def embedding_matrices(self) -> Dict[str, tf.Tensor]:
        # NOTE the note from the decoder's embedding matrix function applies
        # here also:
        with tf.variable_scope("input_projection"):
            return {name:
                    tf.get_variable(
                        "embeddings_factor_{}".format(name),
                        [voc_size, emb_size],
                        initializer=tf.random_normal_initializer(stddev=0.01))
                    for name, emb_size, voc_size
                    in zip(self.data_ids,
                           self.embedding_sizes,
                           self.vocabulary_sizes)}

    @tensor
    def bidirectional_rnn(self) -> Tuple[Tuple[tf.Tensor, tf.Tensor],
                                         Tuple[tf.Tensor, tf.Tensor]]:
        embedded_factors = []
        for name in self.data_ids:
            # pylint: disable=unsubscriptable-object
            embedded_f = tf.nn.embedding_lookup(self.embedding_matrices[name],
                                                self.input_factors[name])
            # pylint: enable=unsubscriptable-object

            embedded_f = dropout(embedded_f, self.dropout_keep_prob,
                                 self.train_mode)
            embedded_factors.append(embedded_f)

        birnn_input = tf.concat(embedded_factors, 2)
        sequence_lengths = tf.to_int32(tf.reduce_sum(self.input_mask, 1))

        fw_cell, bw_cell = self._rnn_cells()  # type: RNNCellTuple
        return tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, birnn_input, sequence_length=sequence_lengths,
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

    # pylint: disable=too-many-locals
    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train

        # for checking the lengths of individual factors
        arr_strings = []
        last_paddings = None

        for name, vocabulary in zip(self.data_ids, self.vocabularies):
            factors = dataset.get_series(name)
            vectors, paddings = vocabulary.sentences_to_tensor(
                list(factors), self.max_input_len, pad_to_max_len=False,
                train_mode=train)

            # pylint: disable=unsubscriptable-object
            fd[self.input_factors[name]] = list(zip(*vectors))
            # pylint: enable=unsubscriptable-object

            arr_strings.append(paddings.tostring())
            last_paddings = paddings

        if len(set(arr_strings)) > 1:
            raise ValueError("The lenghts of factors do not match")

        fd[self.input_mask] = list(zip(*last_paddings))

        return fd
