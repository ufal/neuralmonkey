from typing import Tuple, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import TemporalStatefulWithOutput, Stateful
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.sequence import (Sequence, EmbeddedSequence,
                                         EmbeddedFactorSequence)

# pylint: disable=invalid-name
RNNCellTuple = Tuple[tf.contrib.rnn.RNNCell, tf.contrib.rnn.RNNCell]
# pylint: enable=invalid-name

RNN_CELL_TYPES = {
    "GRU": OrthoGRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell
}


class RecurrentEncoder(ModelPart, TemporalStatefulWithOutput):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: Sequence,
                 rnn_size: int,
                 dropout_keep_prob: float = 1.0,
                 rnn_cell: str = "GRU",
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a new instance of a recurrent encoder."""
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        TemporalStatefulWithOutput.__init__(self)
        check_argument_types()

        self.input_sequence = input_sequence
        self.rnn_size = rnn_size
        self.dropout_keep_prob = dropout_keep_prob
        self.rnn_cell_str = rnn_cell

        if self.rnn_size <= 0:
            raise ValueError("RNN size must be a positive integer.")

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        if self.rnn_cell_str not in RNN_CELL_TYPES:
            raise ValueError("RNN cell must be a either 'GRU' or 'LSTM'")
    # pylint: enable=too-many-arguments

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, [], "train_mode")
    # pylint: enable=no-self-use

    @tensor
    def bidirectional_rnn(self) -> Tuple[Tuple[tf.Tensor, tf.Tensor],
                                         Tuple[tf.Tensor, tf.Tensor]]:
        embedded = dropout(self.input_sequence.data, self.dropout_keep_prob,
                           self.train_mode)

        fw_cell, bw_cell = self._rnn_cells()  # type: RNNCellTuple
        return tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, embedded,
            sequence_length=self.input_sequence.lengths,
            dtype=tf.float32)

    @tensor
    def states(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[0], 2)
        # pylint: enable=unsubscriptable-object

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return self.states

    @tensor
    def encoded(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        if self.rnn_cell_str == "GRU":
            return tf.concat(self.bidirectional_rnn[1], 1)
        elif self.rnn_cell_str == "LSTM":
            # TODO is "h" what we want?
            final_states = [state.h for state in self.bidirectional_rnn[1]]
            return tf.concat(final_states, 1)
        # pylint: enable=unsubscriptable-object

    @tensor
    def output(self) -> tf.Tensor:
        return self.encoded

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.input_sequence.mask

    @tensor
    def states_mask(self) -> tf.Tensor:
        return self.input_sequence.mask

    def _rnn_cells(self) -> RNNCellTuple:
        """Return the graph template to for creating RNN memory cells"""
        return(RNN_CELL_TYPES[self.rnn_cell_str](self.rnn_size),
               RNN_CELL_TYPES[self.rnn_cell_str](self.rnn_size))

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = self.input_sequence.feed_dict(dataset, train)
        fd[self.train_mode] = train

        return fd


class SentenceEncoder(RecurrentEncoder):
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 rnn_size: int,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 rnn_cell: str = "GRU",
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

        RecurrentEncoder.__init__(
            self,
            name=name,
            input_sequence=input_sequence,
            rnn_size=rnn_size,
            dropout_keep_prob=dropout_keep_prob,
            rnn_cell=rnn_cell,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)
    # pylint: enable=too-many-arguments,too-many-locals


class FactoredEncoder(RecurrentEncoder):
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 rnn_size: int,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 rnn_cell: str = "GRU",
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a new instance of the sentence encoder. """
        s_ckp = "input_{}".format(save_checkpoint) if save_checkpoint else None
        l_ckp = "input_{}".format(load_checkpoint) if load_checkpoint else None

        input_sequence = EmbeddedFactorSequence(
            name="{}_input".format(name),
            vocabularies=vocabularies,
            data_ids=data_ids,
            embedding_sizes=embedding_sizes,
            max_length=max_input_len,
            save_checkpoint=s_ckp,
            load_checkpoint=l_ckp)

        RecurrentEncoder.__init__(
            self,
            name=name,
            input_sequence=input_sequence,
            rnn_size=rnn_size,
            dropout_keep_prob=dropout_keep_prob,
            rnn_cell=rnn_cell,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)
    # pylint: enable=too-many-arguments,too-many-locals
