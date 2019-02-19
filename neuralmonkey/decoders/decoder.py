from typing import Any, List, Tuple, cast, NamedTuple

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.autoregressive import (
    AutoregressiveDecoder, LoopState)
from neuralmonkey.attention.base_attention import BaseAttention
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.logging import log
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell, NematusGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.tf_utils import append_tensor
from neuralmonkey.decoders.encoder_projection import (
    linear_encoder_projection, concat_encoder_projection, empty_initial_state,
    EncoderProjection)
from neuralmonkey.decoders.output_projection import (
    OutputProjectionSpec, OutputProjection, nonlinear_output)
from neuralmonkey.decorators import tensor


RNN_CELL_TYPES = {
    "NematusGRU": NematusGRUCell,
    "GRU": OrthoGRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell
}


class RNNFeedables(NamedTuple(
        "RNNFeedables", [
            ("prev_rnn_state", tf.Tensor),
            ("prev_rnn_output", tf.Tensor),
            ("prev_contexts", List[tf.Tensor])])):
    """Additional feedables used only by the RNN-based decoder.

    Attributes:
        prev_rnn_state: The recurrent state from the previous step. A tensor
            of shape ``(batch, rnn_size)``
        prev_rnn_output: The output of the recurrent network from the previous
            step. A tensor of shape ``(batch, output_size)``
        prev_contexts: A list of context vectors returned from attention
            mechanisms. Tensors of shape ``(batch, encoder_state_size)`` for
            each attended encoder.
    """


class RNNHistories(NamedTuple(
        "RNNHistories", [
            ("rnn_outputs", tf.Tensor),
            ("attention_histories", List[Tuple])])):
    """The loop state histories specific for RNN-based decoders.

    Attributes:
        rnn_outputs: History of outputs produced by RNN cell itself (before
            applying output projections).
        attention_histories: A list of ``AttentionLoopState`` objects (or
            similar) populated by values from the attention mechanisms used in
            the decoder.
    """


# pylint: disable=too-many-instance-attributes
class Decoder(AutoregressiveDecoder):
    """A class managing parts of the computation graph used during decoding."""

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    def __init__(self,
                 encoders: List[Stateful],
                 vocabulary: Vocabulary,
                 data_id: str,
                 name: str,
                 max_output_len: int,
                 dropout_keep_prob: float = 1.0,
                 embedding_size: int = None,
                 embeddings_source: EmbeddedSequence = None,
                 tie_embeddings: bool = False,
                 label_smoothing: float = None,
                 rnn_size: int = None,
                 output_projection: OutputProjectionSpec = None,
                 encoder_projection: EncoderProjection = None,
                 attentions: List[BaseAttention] = None,
                 attention_on_input: bool = False,
                 rnn_cell: str = "GRU",
                 conditional_gru: bool = False,
                 supress_unk: bool = False,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Create a refactored version of monster decoder.

        Arguments:
            encoders: Input encoders of the decoder.
            vocabulary: Target vocabulary.
            data_id: Target data series.
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects.
            max_output_len: Maximum length of an output sequence.
            dropout_keep_prob: Probability of keeping a value during dropout.
            embedding_size: Size of embedding vectors for target words.
            embeddings_source: Embedded sequence to take embeddings from.
            tie_embeddings: Use decoder.embedding_matrix also in place
                of the output decoding matrix.
            rnn_size: Size of the decoder hidden state, if None set
                according to encoders.
            output_projection: How to generate distribution over vocabulary
                from decoder_outputs.
            encoder_projection: How to construct initial state from encoders.
            attention: The attention object to use. Optional.
            rnn_cell: RNN Cell used by the decoder (GRU or LSTM).
            conditional_gru: Flag whether to use the Conditional GRU
                architecture.
            attention_on_input: Flag whether attention from previous decoding
                step should be combined with the input in the next step.
            supress_unk: If true, decoder will not produce symbols for unknown
                tokens.
            reuse: Reuse the model variables from the given model part.
        """
        check_argument_types()
        AutoregressiveDecoder.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            max_output_len=max_output_len,
            dropout_keep_prob=dropout_keep_prob,
            embedding_size=embedding_size,
            embeddings_source=embeddings_source,
            tie_embeddings=tie_embeddings,
            label_smoothing=label_smoothing,
            supress_unk=supress_unk,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)

        self.encoders = encoders
        self._output_projection_spec = output_projection
        self._conditional_gru = conditional_gru
        self._attention_on_input = attention_on_input
        self._rnn_cell_str = rnn_cell
        self._rnn_size = rnn_size
        self._encoder_projection = encoder_projection

        self.attentions = []  # type: List[BaseAttention]
        if attentions is not None:
            self.attentions = attentions

        if not rnn_size and not encoder_projection and not encoders:
            raise ValueError(
                "No RNN size, no encoders and no encoder_projection specified")

        if self._rnn_cell_str not in RNN_CELL_TYPES:
            raise ValueError("RNN cell must be a either 'GRU', 'LSTM', or "
                             "'NematusGRU'. Not {}".format(self._rnn_cell_str))

        if self._attention_on_input:
            self.input_projection = self.input_plus_attention
        else:
            self.input_projection = (
                lambda *args: LoopState(*args).feedables.embedded_input)

        with self.use_scope():
            with tf.variable_scope("attention_decoder") as self.step_scope:
                pass

        self._variable_scope.set_initializer(
            tf.random_normal_initializer(stddev=0.001))
    # pylint: enable=too-many-arguments,too-many-branches,too-many-statements

    @property
    def encoder_projection(self) -> EncoderProjection:
        if self._encoder_projection is not None:
            return self._encoder_projection

        if not self.encoders:
            log("No direct encoder input. Using empty initial state")
            return empty_initial_state

        if self._rnn_size is None:
            log("No rnn_size or encoder_projection: Using concatenation of "
                "encoded states")
            return concat_encoder_projection

        log("Using linear projection of encoders as the initial state")
        return linear_encoder_projection(self.dropout_keep_prob)

    @property
    def rnn_size(self) -> int:
        if self._rnn_size is not None:
            return self._rnn_size

        if self._encoder_projection is None:
            assert self.encoders
            return sum(e.output.get_shape()[1].value for e in self.encoders)

        raise ValueError("Cannot infer RNN size.")

    @tensor
    def output_projection_spec(self) -> Tuple[OutputProjection, int]:
        if self._output_projection_spec is None:
            log("No output projection specified - using tanh projection")
            return (nonlinear_output(self.rnn_size, tf.tanh)[0], self.rnn_size)

        if isinstance(self._output_projection_spec, tuple):
            return self._output_projection_spec

        return cast(OutputProjection,
                    self._output_projection_spec), self.rnn_size

    # pylint: disable=unsubscriptable-object
    @property
    def output_projection(self) -> OutputProjection:
        return self.output_projection_spec[0]

    @property
    def output_dimension(self) -> int:
        return self.output_projection_spec[1]
    # pylint: enable=unsubscriptable-object

    @tensor
    def initial_state(self) -> tf.Tensor:
        """Compute initial decoder state.

        The part of the computation graph that computes
        the initial state of the decoder.
        """
        with tf.variable_scope("initial_state"):
            # pylint: disable=not-callable
            initial_state = dropout(
                self.encoder_projection(self.train_mode,
                                        self.rnn_size,
                                        self.encoders),
                self.dropout_keep_prob,
                self.train_mode)
            # pylint: enable=not-callable

            init_state_shape = initial_state.get_shape()

            # Broadcast the initial state to the whole batch if needed
            if len(init_state_shape) == 1:
                assert init_state_shape[0].value == self.rnn_size
                tiles = tf.tile(initial_state,
                                tf.expand_dims(self.batch_size, 0))
                initial_state = tf.reshape(tiles, [-1, self.rnn_size])

        return initial_state

    def _get_rnn_cell(self) -> tf.contrib.rnn.RNNCell:
        return RNN_CELL_TYPES[self._rnn_cell_str](self.rnn_size)

    def _get_conditional_gru_cell(self) -> tf.contrib.rnn.GRUCell:
        if self._rnn_cell_str == "NematusGRU":
            return NematusGRUCell(
                self.rnn_size, use_state_bias=True, use_input_bias=False)

        return RNN_CELL_TYPES[self._rnn_cell_str](self.rnn_size)

    def input_plus_attention(self, *args) -> tf.Tensor:
        """Merge input and previous attentions.

        Input and previous attentions are merged into a single vector
        of the size fo embedding.
        """
        loop_state = LoopState(*args)
        feedables = loop_state.feedables
        emb_with_ctx = tf.concat(
            [feedables.embedded_input] + feedables.prev_contexts, 1)

        return dropout(
            tf.layers.dense(emb_with_ctx, self.embedding_size),
            self.dropout_keep_prob, self.train_mode)

    def next_state(self, loop_state: LoopState) -> Tuple[tf.Tensor, Any, Any]:
        rnn_feedables = loop_state.feedables.other
        rnn_histories = loop_state.histories.other

        with tf.variable_scope(self.step_scope):
            rnn_input = self.input_projection(*loop_state)

            cell = self._get_rnn_cell()
            if self._rnn_cell_str in ["GRU", "NematusGRU"]:
                cell_output, next_state = cell(
                    rnn_input, rnn_feedables.prev_rnn_output)

                attns = [
                    a.attention(
                        cell_output, rnn_feedables.prev_rnn_output,
                        rnn_input, att_loop_state)
                    for a, att_loop_state in zip(
                        self.attentions,
                        rnn_histories.attention_histories)]
                if self.attentions:
                    contexts, att_loop_states = zip(*attns)
                else:
                    contexts, att_loop_states = [], []

                if self._conditional_gru:
                    cell_cond = self._get_conditional_gru_cell()
                    cond_input = tf.concat(contexts, -1)
                    cell_output, next_state = cell_cond(
                        cond_input, next_state, scope="cond_gru_2_cell")

            elif self._rnn_cell_str == "LSTM":
                prev_state = tf.contrib.rnn.LSTMStateTuple(
                    rnn_feedables.prev_rnn_state,
                    rnn_feedables.prev_rnn_output)
                cell_output, state = cell(rnn_input, prev_state)
                next_state = state.c
                attns = [
                    a.attention(
                        cell_output, rnn_feedables.prev_rnn_output,
                        rnn_input, att_loop_state)
                    for a, att_loop_state in zip(
                        self.attentions,
                        rnn_histories.attention_histories)]
                if self.attentions:
                    contexts, att_loop_states = zip(*attns)
                else:
                    contexts, att_loop_states = [], []
            else:
                raise ValueError("Unknown RNN cell.")

            # TODO: attention functions should apply dropout on output
            #       themselves before returning the tensors
            contexts = [dropout(ctx, self.dropout_keep_prob, self.train_mode)
                        for ctx in list(contexts)]
            cell_output = dropout(
                cell_output, self.dropout_keep_prob, self.train_mode)

            with tf.name_scope("rnn_output_projection"):
                # pylint: disable=not-callable
                output = self.output_projection(
                    cell_output, loop_state.feedables.embedded_input,
                    list(contexts), self.train_mode)
                # pylint: enable=not-callable

        new_feedables = RNNFeedables(
            prev_rnn_state=next_state,
            prev_rnn_output=cell_output,
            prev_contexts=list(contexts))

        new_histories = RNNHistories(
            rnn_outputs=append_tensor(rnn_histories.rnn_outputs, cell_output),
            attention_histories=list(att_loop_states))

        return (output, new_feedables, new_histories)

    def get_initial_loop_state(self) -> LoopState:
        default_ls = AutoregressiveDecoder.get_initial_loop_state(self)
        feedables = default_ls.feedables
        histories = default_ls.histories

        rnn_feedables = RNNFeedables(
            prev_contexts=[tf.zeros([self.batch_size, a.context_vector_size])
                           for a in self.attentions],
            prev_rnn_state=self.initial_state,
            prev_rnn_output=self.initial_state)

        rnn_histories = RNNHistories(
            rnn_outputs=tf.zeros(
                shape=[0, self.batch_size, self.rnn_size],
                dtype=tf.float32,
                name="hist_rnn_output_states"),
            attention_histories=[a.initial_loop_state()
                                 for a in self.attentions if a is not None])

        return LoopState(
            histories=histories._replace(other=rnn_histories),
            constants=default_ls.constants,
            feedables=feedables._replace(other=rnn_feedables))

    def finalize_loop(self, final_loop_state: LoopState,
                      train_mode: bool) -> None:
        for att_state, attn_obj in zip(
                final_loop_state.histories.other.attention_histories,
                self.attentions):

            att_history_key = "{}_{}".format(
                self.name, "train" if train_mode else "run")

            attn_obj.finalize_loop(att_history_key, att_state)

            if not train_mode:
                attn_obj.visualize_attention(att_history_key)
