# pylint: disable=too-many-lines
import math
from typing import (cast, Iterable, List, Callable, Optional,
                    Any, Tuple, NamedTuple, Union)

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.base_attention import Attention
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import (Vocabulary, START_TOKEN, END_TOKEN_INDEX,
                                     PAD_TOKEN_INDEX)
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.model.stateful import (TemporalStatefulWithOutput,
                                         SpatialStatefulWithOutput)
from neuralmonkey.logging import log, warn
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.nn.projection import linear
from neuralmonkey.decoders.encoder_projection import (
    linear_encoder_projection, concat_encoder_projection, empty_initial_state)
from neuralmonkey.decoders.output_projection import (OutputProjectionSpec,
                                                     nonlinear_output)
from neuralmonkey.decorators import tensor


RNN_CELL_TYPES = {
    "GRU": OrthoGRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell
}


# The LoopState is a structure that works with the tf.while_loop function
# the decoder loop state stores all the information that is not invariant
# for the decoder run.
LoopState = NamedTuple("LoopState",
                       [("step", tf.Tensor),  # 1D int, number of the step
                        ("input_symbol", tf.Tensor),  # batch of ints to vocab
                        ("train_inputs", Optional[tf.Tensor]),
                        ("prev_rnn_state", tf.Tensor),
                        ("prev_rnn_output", tf.Tensor),
                        ("rnn_outputs", tf.TensorArray),
                        ("prev_logits", tf.Tensor),
                        ("logits", tf.TensorArray),
                        ("prev_contexts", List[tf.Tensor]),
                        ("mask", tf.TensorArray),  # float matrix, 0s and 1s
                        ("finished", tf.Tensor),  # batch-sized, bool
                        ("attention_loop_states", List[Any])])  # see att docs


# pylint: disable=too-many-public-methods,too-many-instance-attributes
# Big decoder cannot be simpler. Not sure if refactoring
# it into smaller units would be helpful
class Decoder(ModelPart):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    def __init__(self,
                 # TODO only stateful, attention will need temporal or spat.
                 encoders: List[Union[TemporalStatefulWithOutput,
                                      SpatialStatefulWithOutput]],
                 vocabulary: Vocabulary,
                 data_id: str,
                 name: str,
                 max_output_len: int,
                 dropout_keep_prob: float = 1.0,
                 rnn_size: int = None,
                 embedding_size: int = None,
                 output_projection: OutputProjectionSpec = None,
                 encoder_projection: Callable[
                     [tf.Tensor, Optional[int], Optional[List[Any]]],
                     tf.Tensor]=None,
                 attentions: List[Attention] = None,
                 embeddings_source: EmbeddedSequence = None,
                 attention_on_input: bool = True,
                 rnn_cell: str = 'GRU',
                 conditional_gru: bool = False,
                 save_checkpoint: str= None,
                 load_checkpoint: str = None) -> None:
        """Create a refactored version of monster decoder.

        Arguments:
            encoders: Input encoders of the decoder
            vocabulary: Target vocabulary
            data_id: Target data series
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects
            max_output_len: Maximum length of an output sequence
            dropout_keep_prob: Probability of keeping a value during dropout

        Keyword arguments:
            rnn_size: Size of the decoder hidden state, if None set
                according to encoders.
            embedding_size: Size of embedding vectors for target words
            output_projection: How to generate distribution over vocabulary
                from decoder rnn_outputs
            encoder_projection: How to construct initial state from encoders
            attention: The attention object to use. Optional.
            embeddings_source: Embedded sequence to take embeddings from
            rnn_cell: RNN Cell used by the decoder (GRU or LSTM)
            conditional_gru: Flag whether to use the Conditional GRU
                architecture
            attention_on_input: Flag whether attention from previous decoding
                step should be combined with the input in the next step.
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        log("Initializing decoder, name: '{}'".format(name))

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_output_len = max_output_len
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.output_projection_spec = output_projection
        self.encoder_projection = encoder_projection
        self.attentions = attentions
        self.embeddings_source = embeddings_source
        self._conditional_gru = conditional_gru
        self._attention_on_input = attention_on_input
        self._rnn_cell_str = rnn_cell

        if self.embedding_size is None and self.embeddings_source is None:
            raise ValueError("You must specify either embedding size or the "
                             "embedded sequence from which to reuse the "
                             "embeddings (e.g. set either 'embedding_size' or "
                             " 'embeddings_source' parameter)")

        if self.embeddings_source is not None:
            if self.embedding_size is not None:
                warn("Overriding the embedding_size parameter with the"
                     " size of the reused embeddings from the encoder.")

            self.embedding_size = (
                self.embeddings_source.embedding_matrix.get_shape()[1].value)

        if self.encoder_projection is None:
            if not self.encoders:
                log("No encoder - language model only.")
                self.encoder_projection = empty_initial_state
            elif rnn_size is None:
                log("No rnn_size or encoder_projection: Using concatenation of"
                    " encoded states")
                self.encoder_projection = concat_encoder_projection
                self.rnn_size = sum(e.output.get_shape()[1].value
                                    for e in encoders)
            else:
                log("Using linear projection of encoders as the initial state")
                self.encoder_projection = linear_encoder_projection(
                    self.dropout_keep_prob)

        assert self.rnn_size is not None

        if self._rnn_cell_str not in RNN_CELL_TYPES:
            raise ValueError("RNN cell must be a either 'GRU' or 'LSTM'")

        if self.output_projection_spec is None:
            log("No output projection specified - using tanh projection")
            self.output_projection = nonlinear_output(
                self.rnn_size, tf.tanh)[0]
            self.output_projection_size = self.rnn_size
        elif isinstance(self.output_projection_spec, tuple):
            (self.output_projection,
             self.output_projection_size) = tuple(self.output_projection_spec)
        else:
            self.output_projection = self.output_projection_spec
            self.output_projection_size = self.rnn_size

        if self._attention_on_input:
            self.input_projection = self.input_plus_attention
        else:
            self.input_projection = self.embed_input_symbol

        with self.use_scope():
            with tf.variable_scope("attention_decoder") as self.step_scope:
                pass

        # TODO when it is possible, remove the printing of the cost var
        log("Decoder initalized. Cost var: {}".format(str(self.cost)))
    # pylint: enable=too-many-arguments,too-many-branches,too-many-statements

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, shape=[], name="train_mode")

    @tensor
    def go_symbols(self) -> tf.Tensor:
        return tf.placeholder(tf.int32, shape=[None], name="go_symbols")
    # pylint: enable=no-self-use

    @tensor
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self.go_symbols)[0]

    # pylint: disable=no-self-use
    @tensor
    def train_inputs(self) -> tf.Tensor:
        # NOTE transposed shape (time, batch)
        return tf.placeholder(tf.int32, [None, None], name="train_inputs")

    @tensor
    def train_padding(self) -> tf.Tensor:
        # NOTE transposed shape (time, batch)
        # rename padding to mask
        return tf.placeholder(tf.float32, [None, None], name="train_mask")
    # pylint: enable=no-self-use

    @tensor
    def initial_state(self) -> tf.Tensor:
        """The part of the computation graph that computes
        the initial state of the decoder.
        """
        with tf.variable_scope("initial_state"):
            initial_state = dropout(
                self.encoder_projection(self.train_mode,
                                        self.rnn_size,
                                        self.encoders),
                self.dropout_keep_prob,
                self.train_mode)

            # pylint: disable=no-member
            # Pylint keeps complaining about initial shape being a tuple,
            # but it is a tensor!!!
            init_state_shape = initial_state.get_shape()
            # pylint: enable=no-member

            # Broadcast the initial state to the whole batch if needed
            if len(init_state_shape) == 1:
                assert init_state_shape[0].value == self.rnn_size
                tiles = tf.tile(initial_state,
                                tf.expand_dims(self.batch_size, 0))
                initial_state = tf.reshape(tiles, [-1, self.rnn_size])

        return initial_state

    @tensor
    def embedding_matrix(self) -> tf.Variable:
        """Variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes the embedding
        matrix from the first encoder
        """
        if self.embeddings_source is not None:
            return self.embeddings_source.embedding_matrix

        # TODO better initialization
        return tf.get_variable(
            "word_embeddings", [len(self.vocabulary), self.embedding_size],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

    @tensor
    def decoding_w(self) -> tf.Variable:
        with tf.name_scope("output_projection"):
            return tf.get_variable(
                "state_to_word_W",
                [self.output_projection_size, len(self.vocabulary)],
                initializer=tf.random_uniform_initializer(-0.5, 0.5))

    @tensor
    def decoding_b(self) -> tf.Variable:
        with tf.name_scope("output_projection"):
            return tf.get_variable(
                "state_to_word_b", [len(self.vocabulary)],
                initializer=tf.constant_initializer(
                    - math.log(len(self.vocabulary))))

    @tensor
    def train_logits(self) -> tf.Tensor:
        # THE LAST TRAIN INPUT IS NOT USED IN DECODING FUNCTION
        # (just as a target)
        logits, _, _ = self._decoding_loop(train_mode=True)

        return logits

    @tensor
    def runtime_loop_result(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self._decoding_loop(train_mode=False)

    @tensor
    def runtime_logits(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return self.runtime_loop_result[0]
        # pylint: enable=unsubscriptable-object

    @tensor
    def runtime_rnn_states(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return self.runtime_loop_result[1]
        # pylint: enable=unsubscriptable-object

    @tensor
    def runtime_mask(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return self.runtime_loop_result[2]
        # pylint: enable=unsubscriptable-object

    @tensor
    def train_xents(self) -> tf.Tensor:
        train_targets = tf.transpose(self.train_inputs)

        return tf.contrib.seq2seq.sequence_loss(
            tf.transpose(self.train_logits, perm=[1, 0, 2]),
            train_targets,
            tf.transpose(self.train_padding),
            average_across_batch=False)

    @tensor
    def train_loss(self) -> tf.Tensor:
        return tf.reduce_mean(self.train_xents)

    @property
    def cost(self) -> tf.Tensor:
        return self.train_loss

    @tensor
    def train_logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.train_logits)

    @tensor
    def decoded(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.argmax(self.runtime_logits[:, :, 1:], -1) + 1
        # pylint: enable=unsubscriptable-object

    @tensor
    def runtime_loss(self) -> tf.Tensor:
        train_targets = tf.transpose(self.train_inputs)
        batch_major_logits = tf.transpose(self.runtime_logits, [1, 0, 2])
        min_time = tf.minimum(tf.shape(train_targets)[1],
                              tf.shape(batch_major_logits)[1])

        # TODO if done properly, there should be padding of the shorter
        # sequence instead of cropping to the length of the shorter one

        return tf.contrib.seq2seq.sequence_loss(
            logits=batch_major_logits[:, :min_time],
            targets=train_targets[:, :min_time],
            weights=tf.transpose(self.train_padding)[:, :min_time])

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.runtime_logits)

    def _logit_function(self, state: tf.Tensor) -> tf.Tensor:
        state = dropout(state, self.dropout_keep_prob, self.train_mode)
        return tf.matmul(state, self.decoding_w) + self.decoding_b

    def _get_rnn_cell(self) -> tf.contrib.rnn.RNNCell:
        return RNN_CELL_TYPES[self._rnn_cell_str](self.rnn_size)

    def _get_conditional_gru_cell(self) -> tf.contrib.rnn.GRUCell:
        return tf.contrib.rnn.GRUCell(self.rnn_size)

    def embed_input_symbol(self, *args) -> tf.Tensor:
        loop_state = LoopState(*args)

        embedded_input = tf.nn.embedding_lookup(
            self.embedding_matrix, loop_state.input_symbol)

        return dropout(embedded_input, self.dropout_keep_prob, self.train_mode)

    def input_plus_attention(self, *args) -> tf.Tensor:
        """Merge input and previous attentions into one vector of the
         right size.
        """
        loop_state = LoopState(*args)

        embedded_input = self.embed_input_symbol(*loop_state)

        return linear([embedded_input] + loop_state.prev_contexts,
                      self.embedding_size)

    def get_body(self,
                 train_mode: bool,
                 sample: bool = False) -> Callable:
        # pylint: disable=too-many-branches
        def body(*args) -> LoopState:
            loop_state = LoopState(*args)
            step = loop_state.step

            with tf.variable_scope(self.step_scope):
                # Compute the input to the RNN
                rnn_input = self.input_projection(*loop_state)

                # Run the RNN.
                cell = self._get_rnn_cell()
                if self._rnn_cell_str == 'GRU':
                    cell_output, state = cell(rnn_input,
                                              loop_state.prev_rnn_output)
                    next_state = state
                    attns = [
                        a.attention(cell_output, loop_state.prev_rnn_output,
                                    rnn_input, att_loop_state, loop_state.step)
                        for a, att_loop_state in zip(
                            self.attentions,
                            loop_state.attention_loop_states)]
                    if self.attentions:
                        contexts, att_loop_states = zip(*attns)
                    else:
                        contexts, att_loop_states = [], []

                    if self._conditional_gru:
                        cell_cond = self._get_conditional_gru_cell()
                        cond_input = tf.concat(contexts, -1)
                        cell_output, state = cell_cond(cond_input, state,
                                                       scope="cond_gru_2_cell")
                elif self._rnn_cell_str == 'LSTM':
                    prev_state = tf.contrib.rnn.LSTMStateTuple(
                        loop_state.prev_rnn_state, loop_state.prev_rnn_output)
                    cell_output, state = cell(rnn_input, prev_state)
                    next_state = state.c
                    attns = [
                        a.attention(cell_output, loop_state.prev_rnn_output,
                                    rnn_input, att_loop_state, loop_state.step)
                        for a, att_loop_state in zip(
                            self.attentions,
                            loop_state.attention_loop_states)]
                    if self.attentions:
                        contexts, att_loop_states = zip(*attns)
                    else:
                        contexts, att_loop_states = [], []
                else:
                    raise ValueError("Unknown RNN cell.")

                with tf.name_scope("rnn_output_projection"):
                    embedded_input = tf.nn.embedding_lookup(
                        self.embedding_matrix, loop_state.input_symbol)

                    output = self.output_projection(
                        cell_output, embedded_input, list(contexts),
                        self.train_mode)

                logits = self._logit_function(output)

            self.step_scope.reuse_variables()

            if sample:
                next_symbols = tf.multinomial(logits, num_samples=1)
            elif train_mode:
                next_symbols = loop_state.train_inputs[step]
            else:
                next_symbols = tf.to_int32(tf.argmax(logits, axis=1))
                int_unfinished_mask = tf.to_int32(
                    tf.logical_not(loop_state.finished))

                # Note this works only when PAD_TOKEN_INDEX is 0. Otherwise
                # this have to be rewritten
                assert PAD_TOKEN_INDEX == 0
                next_symbols = next_symbols * int_unfinished_mask

            has_just_finished = tf.equal(next_symbols, END_TOKEN_INDEX)
            has_finished = tf.logical_or(loop_state.finished,
                                         has_just_finished)

            new_loop_state = LoopState(
                step=step + 1,
                input_symbol=next_symbols,
                train_inputs=loop_state.train_inputs,
                prev_rnn_state=next_state,
                prev_rnn_output=cell_output,
                rnn_outputs=loop_state.rnn_outputs.write(
                    step + 1, cell_output),
                prev_contexts=list(contexts),
                prev_logits=logits,
                logits=loop_state.logits.write(step, logits),
                finished=has_finished,
                mask=loop_state.mask.write(step,
                                           tf.logical_not(has_finished)),
                attention_loop_states=list(att_loop_states))
            return new_loop_state
        # pylint: enable=too-many-branches

        return body

    def get_initial_loop_state(self) -> LoopState:
        rnn_output_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True,
                                       size=0, name="rnn_outputs")
        rnn_output_ta = rnn_output_ta.write(0, self.initial_state)

        logit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True,
                                  size=0, name="logits")

        contexts = [tf.zeros([self.batch_size, a.input_state_size])
                    for a in self.attentions]

        mask_ta = tf.TensorArray(dtype=tf.bool, dynamic_size=True,
                                 size=0, name="mask")

        attn_loop_states = [a.initial_loop_state()
                            for a in self.attentions if a is not None]

        return LoopState(
            step=0,
            input_symbol=self.go_symbols,
            train_inputs=self.train_inputs,
            prev_rnn_state=self.initial_state,
            prev_rnn_output=self.initial_state,
            rnn_outputs=rnn_output_ta,
            prev_logits=tf.zeros([self.batch_size, len(self.vocabulary)]),
            logits=logit_ta,
            prev_contexts=contexts,
            mask=mask_ta,
            finished=tf.zeros([self.batch_size], dtype=tf.bool),
            attention_loop_states=attn_loop_states)

    def loop_continue_criterion(self, *args) -> tf.Tensor:
        loop_state = LoopState(*args)
        finished = loop_state.finished
        not_all_done = tf.logical_not(tf.reduce_all(finished))
        before_max_len = tf.less(loop_state.step,
                                 self.max_output_len)
        return tf.logical_and(not_all_done, before_max_len)

    def _decoding_loop(self, train_mode: bool, sample: bool = False)-> Tuple[
            tf.Tensor, tf.Tensor, tf.Tensor]:

        final_loop_state = tf.while_loop(
            self.loop_continue_criterion,
            self.get_body(train_mode, sample),
            self.get_initial_loop_state())

        for att_state, attn_obj in zip(
                final_loop_state.attention_loop_states, self.attentions):

            att_history_key = "{}_{}".format(
                self.name, "train" if train_mode else "run")

            attn_obj.finalize_loop(att_history_key, att_state)

        logits = final_loop_state.logits.stack()
        rnn_outputs = final_loop_state.rnn_outputs.stack()

        # TODO mask should include also the end symbol
        mask = final_loop_state.mask.stack()

        return logits, rnn_outputs, mask

    def _visualize_attention(self) -> None:
        """Create image summaries with attentions"""
        # TODO! this method will become part of attention that is a separate
        # ModelPart which will ensure that all lazily created tensors will be
        # already there.
        for i, a in enumerate(self.attentions):
            if not hasattr(a, "attentions_in_time"):
                continue

            alignments = tf.expand_dims(tf.transpose(
                tf.stack(a.attentions_in_time), perm=[1, 2, 0]), -1)

            tf.summary.image(
                "attention_{}".format(i), alignments,
                collections=["summary_val_plots"],
                max_outputs=256)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary for the decoder object

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run
        """
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        if sentences is None and train:
            raise ValueError("When training, you must feed "
                             "reference sentences")

        sentences_list = list(sentences) if sentences is not None else None

        fd = {}  # type: FeedDict
        fd[self.train_mode] = train

        go_symbol_idx = self.vocabulary.get_word_index(START_TOKEN)
        fd[self.go_symbols] = np.full([len(dataset)], go_symbol_idx,
                                      dtype=np.int32)

        if sentences is not None:
            # train_mode=False, since we don't want to <unk>ize target words!
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len, train_mode=False,
                add_start_symbol=False, add_end_symbol=True,
                pad_to_max_len=False)

            fd[self.train_inputs] = inputs
            fd[self.train_padding] = weights

        return fd
