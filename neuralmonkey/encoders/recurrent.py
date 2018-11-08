from typing import Tuple, List, Union, Callable, NamedTuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import (
    TemporalStatefulWithOutput, TemporalStateful)
from neuralmonkey.model.model_part import ModelPart, InitializerSpecs
from neuralmonkey.logging import warn
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell, NematusGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decorators import tensor
from neuralmonkey.model.sequence import (
    EmbeddedSequence, EmbeddedFactorSequence)

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
              rnn_spec: RNNSpec,
              add_residual: bool) -> Tuple[tf.Tensor, tf.Tensor]:
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

    if add_residual:
        if outputs.get_shape()[-1].value != rnn_input.get_shape()[-1].value:
            warn("Size of the RNN layer input ({}) and layer output ({}) "
                 "must match when applying residual connection. Reshaping "
                 "the rnn output using linear projection.".format(
                     outputs.get_shape(), rnn_input.get_shape()))
            outputs = tf.layers.dense(outputs, rnn_input.shape.as_list()[-1])
        outputs += rnn_input

    return outputs, final_state


class RecurrentEncoder(ModelPart, TemporalStatefulWithOutput):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_sequence: TemporalStateful,
                 rnn_size: int,
                 rnn_cell: str = "GRU",
                 rnn_direction: str = "bidirectional",
                 add_residual: bool = False,
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
        self.rnn_spec = _make_rnn_spec(rnn_size, rnn_direction, rnn_cell)
        self.add_residual = add_residual

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
        return rnn_layer(self.rnn_input, self.input_sequence.lengths,
                         self.rnn_spec, self.add_residual)

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
            rnn_size=rnn_size,
            rnn_cell=rnn_cell,
            rnn_direction=rnn_direction,
            add_residual=add_residual,
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
            rnn_size=rnn_size,
            rnn_cell=rnn_cell,
            rnn_direction=rnn_direction,
            add_residual=add_residual,
            dropout_keep_prob=dropout_keep_prob,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)
    # pylint: enable=too-many-arguments,too-many-locals


class DeepSentenceEncoder(SentenceEncoder):
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 embedding_size: int,
                 rnn_sizes: List[int],
                 rnn_directions: List[str],
                 rnn_cell: str = "GRU",
                 add_residual: bool = False,
                 max_input_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None,
                 embedding_initializer: Callable = None) -> None:
        """Create a new instance of the deep sentence encoder.

        Arguments:
            name: ModelPart name.
            vocabulary: The input vocabulary.
            data_id: The input sequence data ID.
            embedding_size: The dimension of the embedding vectors in the input
                sequence.
            max_input_len: Maximum length of the input sequence (disregard
                tokens after this position).
            rnn_sizes: The list of dimensions of the RNN hidden state vectors
                in respective layers.
            rnn_cell: One of "GRU", "NematusGRU", "LSTM". Which kind of memory
                cell to use.
            rnn_directions: The list of rnn directions in the respective
                layers. Should be equally long as `rnn_sizes`. Each item must
                be one of "forward", "backward", "bidirectional". Determines in
                what order to process the input sequence. Note that choosing
                "bidirectional" will double the resulting vector dimension as
                well as the number of the parameters in the given layer.
            add_residual: Add residual connections to each RNN layer output.
            dropout_keep_prob: 1 - dropout probability.
            save_checkpoint: ModelPart save checkpoint file.
            load_checkpoint: ModelPart load checkpoint file.
        """
        check_argument_types()

        if len(rnn_sizes) != len(rnn_directions):
            raise ValueError("Different number of rnn sizes and directions.")

        self.rnn_sizes = rnn_sizes
        self.rnn_directions = rnn_directions
        self.rnn_cell = rnn_cell

        SentenceEncoder.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            embedding_size=embedding_size,
            rnn_size=rnn_sizes[-1],
            rnn_direction=rnn_directions[-1],
            rnn_cell=rnn_cell,
            add_residual=add_residual,
            max_input_len=max_input_len,
            dropout_keep_prob=dropout_keep_prob,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers,
            embedding_initializer=embedding_initializer)

    @tensor
    def rnn(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Run stacked RNN given sizes and directions.

        Inputs of the first RNN are the RNN inputs to the encoder. Outputs from
        each layer are used as inputs to the next one. As a final state of the
        stacked RNN, the final state of the final layer is used.
        """
        rnn_input_local = self.rnn_input

        for level, (rnn_size, rnn_dir) in enumerate(
                zip(self.rnn_sizes, self.rnn_directions)):
            rnn_spec = _make_rnn_spec(rnn_size, rnn_dir, self.rnn_cell)

            with tf.variable_scope("layer_{}".format(level)):
                outputs, state = rnn_layer(
                    rnn_input_local, self.input_sequence.lengths,
                    rnn_spec, self.add_residual)

            # pylint - redefinition from instancemethod to list
            # pylint: disable=redefined-variable-type
            rnn_input_local = outputs
            # pylint: enable=redefined-variable-type

        return outputs, state
