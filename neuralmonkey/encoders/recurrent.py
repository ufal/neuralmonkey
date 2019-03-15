from typing import Tuple, List, Union, Callable, NamedTuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import (
    TemporalStatefulWithOutput, TemporalStateful)
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell, NematusGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.model.sequence import (
    EmbeddedSequence, EmbeddedFactorSequence)
from neuralmonkey.tf_utils import layer_norm

RNN_CELL_TYPES = {
    "NematusGRU": NematusGRUCell,
    "GRU": OrthoGRUCell,
    "LSTM": tf.nn.rnn_cell.LSTMCell
}

RNN_DIRECTIONS = ["forward", "backward", "bidirectional"]

# pylint: disable=invalid-name
RNNSpecTuple = Union[Tuple[int], Tuple[int, str], Tuple[int, str, str]]
RNNCellTuple = Tuple[tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.RNNCell]
# pylint: enable=invalid-name


class RNNSpec(NamedTuple(
        "RNNSpec",
        [("size", int),
         ("direction", str),
         ("cell_type", str)])):
    """Recurrent neural network specifications.

    Attributes:
        size: The state size.
        direction: The RNN processing direction. One of ``forward``,
            ``backward``, and ``bidirectional``.
        cell_type: The recurrent cell type to use. Refer to
            ``encoders.recurrent.RNN_CELL_TYPES`` for possible values.
    """


def _make_rnn_spec(size: int,
                   direction: str = "bidirectional",
                   cell_type: str = "GRU") -> RNNSpec:
    if size <= 0:
        raise ValueError(
            "RNN size must be a positive integer. {} given.".format(size))

    if direction not in RNN_DIRECTIONS:
        raise ValueError("RNN direction must be one of {}. {} given."
                         .format(str(RNN_DIRECTIONS), direction))

    if cell_type not in RNN_CELL_TYPES:
        raise ValueError("RNN cell type must be one of {}. {} given."
                         .format(str(RNN_CELL_TYPES), cell_type))

    return RNNSpec(size, direction, cell_type)


def _make_rnn_cell(spec: RNNSpec) -> Callable[[], tf.nn.rnn_cell.RNNCell]:
    """Return the graph template for creating RNN cells."""
    return RNN_CELL_TYPES[spec.cell_type](spec.size)


def rnn_layer(rnn_input: tf.Tensor,
              lengths: tf.Tensor,
              rnn_spec: RNNSpec) -> Tuple[tf.Tensor, tf.Tensor]:
    """Construct a RNN layer given its inputs and specs.

    Arguments:
        rnn_inputs: The input sequence to the RNN.
        lengths: Lengths of input sequences.
        rnn_spec: A valid RNNSpec tuple specifying the network architecture.
        add_residual: Add residual connections to the layer output.
    """
    if rnn_spec.direction == "bidirectional":
        fw_cell = _make_rnn_cell(rnn_spec)
        bw_cell = _make_rnn_cell(rnn_spec)

        outputs_tup, states_tup = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, rnn_input, sequence_length=lengths,
            dtype=tf.float32)

        outputs = tf.concat(outputs_tup, 2)

        if rnn_spec.cell_type == "LSTM":
            states_tup = (state.h for state in states_tup)

        final_state = tf.concat(list(states_tup), 1)
    else:
        if rnn_spec.direction == "backward":
            rnn_input = tf.reverse_sequence(rnn_input, lengths, seq_axis=1)

        cell = _make_rnn_cell(rnn_spec)
        outputs, final_state = tf.nn.dynamic_rnn(
            cell, rnn_input, sequence_length=lengths, dtype=tf.float32)

        if rnn_spec.direction == "backward":
            outputs = tf.reverse_sequence(outputs, lengths, seq_axis=1)

        if rnn_spec.cell_type == "LSTM":
            final_state = final_state.h

    return outputs, final_state


class RecurrentEncoder(ModelPart, TemporalStatefulWithOutput):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: TemporalStateful,
                 rnn_layers: List[RNNSpecTuple],
                 add_residual: bool = False,
                 add_layer_norm: bool = False,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Create a new instance of a recurrent encoder.

        Arguments:
            name: ModelPart name.
            input_seqeunce: The input sequence for the encoder.
            rnn_size: The dimension of the RNN hidden state vector.
            rnn_cell: One of "GRU", "NematusGRU", "LSTM". Which kind of memory
                cell to use.
            rnn_direction: One of "forward", "backward", "bidirectional". In
                what order to process the input sequence. Note that choosing
                "bidirectional" will double the resulting vector dimension as
                well as the number of encoder parameters.
            add_residual: Add residual connections to the RNN layer output.
            add_layer_norm: Add layer normalization after each RNN layer.
            dropout_keep_prob: 1 - dropout probability.
            save_checkpoint: ModelPart save checkpoint file.
            load_checkpoint: ModelPart load checkpoint file.
        """
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)
        TemporalStatefulWithOutput.__init__(self)

        self.input_sequence = input_sequence
        self.dropout_keep_prob = dropout_keep_prob
        self.rnn_specs = [_make_rnn_spec(*r) for r in rnn_layers]
        self.add_residual = add_residual
        self.add_layer_norm = add_layer_norm

        if self.dropout_keep_prob <= 0.0 or self.dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep prob must be inside (0,1].")

        self._variable_scope.set_initializer(
            tf.random_normal_initializer(stddev=0.001))
    # pylint: enable=too-many-arguments

    @tensor
    def rnn_input(self) -> tf.Tensor:
        return dropout(self.input_sequence.temporal_states,
                       self.dropout_keep_prob, self.train_mode)

    @tensor
    def rnn(self) -> Tuple[tf.Tensor, tf.Tensor]:
        layer_input = self.rnn_input  # type: tf.Tensor
        layer_final = None

        for i, rnn_spec in enumerate(self.rnn_specs):
            with tf.variable_scope("rnn_{}_{}".format(i, rnn_spec.direction),
                                   reuse=tf.AUTO_REUSE):

                if self.add_layer_norm:
                    layer_input = layer_norm(layer_input)

                layer_output, layer_final_output = rnn_layer(
                    layer_input, self.input_sequence.lengths, rnn_spec)

                layer_output = dropout(
                    layer_output, self.dropout_keep_prob, self.train_mode)
                layer_final_output = dropout(
                    layer_final_output, self.dropout_keep_prob,
                    self.train_mode)

                in_dim = layer_input.get_shape()[-1]
                out_dim = layer_output.get_shape()[-1]

                if self.add_residual and in_dim == out_dim:
                    assert layer_final is not None
                    layer_input += layer_output
                    layer_final += layer_final_output
                else:
                    # pylint: disable=redefined-variable-type
                    layer_input = layer_output
                    layer_final = layer_final_output
                    # pylint: enable=redefined-variable-type

        assert layer_final is not None
        return layer_input, layer_final

    @tensor
    def temporal_states(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return self.rnn[0]
        # pylint: enable=unsubscriptable-object

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.input_sequence.temporal_mask

    @tensor
    def output(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return self.rnn[1]
        # pylint: enable=unsubscriptable-object


class SentenceEncoder(RecurrentEncoder):
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 rnn_size: int,
                 rnn_cell: str = "GRU",
                 rnn_direction: str = "bidirectional",
                 add_residual: bool = False,
                 add_layer_norm: bool = False,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None,
                 embedding_initializer: Callable = None) -> None:
        """Create a new instance of the sentence encoder.

        Arguments:
            name: ModelPart name.
            vocabulary: The input vocabulary.
            data_id: The input sequence data ID.
            embedding_size: The dimension of the embedding vectors in the input
                sequence.
            max_input_len: Maximum length of the input sequence (disregard
                tokens after this position).
            rnn_size: The dimension of the RNN hidden state vector.
            rnn_cell: One of "GRU", "NematusGRU", "LSTM". Which kind of memory
                cell to use.
            rnn_direction: One of "forward", "backward", "bidirectional". In
                what order to process the input sequence. Note that choosing
                "bidirectional" will double the resulting vector dimension as
                well as the number of encoder parameters.
            add_residual: Add residual connections to the RNN layer output.
            add_layer_norm: Add layer normalization after each RNN layer.
            dropout_keep_prob: 1 - dropout probability.
            save_checkpoint: ModelPart save checkpoint file.
            load_checkpoint: ModelPart load checkpoint file.
        """
        check_argument_types()
        s_ckp = "input_{}".format(save_checkpoint) if save_checkpoint else None
        l_ckp = "input_{}".format(load_checkpoint) if load_checkpoint else None
        input_initializers = []
        if embedding_initializer is not None:
            input_initializers.append(
                ("embedding_matrix_0", embedding_initializer))

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
            load_checkpoint=l_ckp,
            initializers=input_initializers)

        RecurrentEncoder.__init__(
            self,
            name=name,
            input_sequence=input_sequence,
            rnn_layers=[(rnn_size, rnn_direction, rnn_cell)],
            add_residual=add_residual,
            add_layer_norm=add_layer_norm,
            dropout_keep_prob=dropout_keep_prob,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)
    # pylint: enable=too-many-arguments,too-many-locals


class FactoredEncoder(RecurrentEncoder):
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 rnn_size: int,
                 rnn_cell: str = "GRU",
                 rnn_direction: str = "bidirectional",
                 add_residual: bool = False,
                 add_layer_norm: bool = False,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None,
                 input_initializers: InitializerSpecs = None) -> None:
        """Create a new instance of the factored encoder.

        Arguments:
            name: ModelPart name.
            vocabularies: The vocabularies for each factor.
            data_ids: The input sequence data ID for each factor.
            embedding_sizes: The dimension of the embedding vectors in the
                input sequence for each factor.
            max_input_len: Maximum length of the input sequence (disregard
                tokens after this position).
            rnn_size: The dimension of the RNN hidden state vector.
            rnn_cell: One of "GRU", "NematusGRU", "LSTM". Which kind of memory
                cell to use.
            rnn_direction: One of "forward", "backward", "bidirectional". In
                what order to process the input sequence. Note that choosing
                "bidirectional" will double the resulting vector dimension as
                well as the number of encoder parameters.
            add_residual: Add residual connections to the RNN layer output.
            add_layer_norm: Add layer normalization after each RNN layer.
            dropout_keep_prob: 1 - dropout probability.
            save_checkpoint: ModelPart save checkpoint file.
            load_checkpoint: ModelPart load checkpoint file.
        """
        check_argument_types()
        s_ckp = "input_{}".format(save_checkpoint) if save_checkpoint else None
        l_ckp = "input_{}".format(load_checkpoint) if load_checkpoint else None

        input_sequence = EmbeddedFactorSequence(
            name="{}_input".format(name),
            vocabularies=vocabularies,
            data_ids=data_ids,
            embedding_sizes=embedding_sizes,
            max_length=max_input_len,
            save_checkpoint=s_ckp,
            load_checkpoint=l_ckp,
            initializers=input_initializers)

        RecurrentEncoder.__init__(
            self,
            name=name,
            input_sequence=input_sequence,
            rnn_layers=[(rnn_size, rnn_cell, rnn_direction)],
            add_residual=add_residual,
            add_layer_norm=add_layer_norm,
            dropout_keep_prob=dropout_keep_prob,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)
    # pylint: enable=too-many-arguments,too-many-locals
