"""Abstract class for autoregressive decoding.

Either for the recurrent decoder, or for the transformer decoder.

The autoregressive decoder uses the while loop to get the outputs.
Descendants should only specify the initial state and the while loop body.
"""
from typing import NamedTuple, Callable, Tuple, Optional, Any, List

import numpy as np
import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.logging import log, warn
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.nn.utils import dropout
from neuralmonkey.tf_utils import get_variable, get_state_shape_invariants
from neuralmonkey.vocabulary import Vocabulary, START_TOKEN, UNK_TOKEN_INDEX


class LoopState(NamedTuple(
        "LoopState",
        [("histories", Any),
         ("constants", Any),
         ("feedables", Any)])):
    """The loop state object.

    The LoopState is a structure that works with the tf.while_loop function the
    decoder loop state stores all the information that is not invariant for the
    decoder run.

    Attributes:
        histories: A set of tensors that grow in time as the decoder proceeds.
        constants: A set of independent tensors that do not change during the
            entire decoder run.
        feedables: A set of tensors used as the input of a single decoder step.
    """


class DecoderHistories(NamedTuple(
        "DecoderHistories",
        [("logits", tf.Tensor),
         ("decoder_outputs", tf.Tensor),
         ("outputs", tf.Tensor),
         ("mask", tf.Tensor)])):
    """The values collected during the run of an autoregressive decoder.

    Attributes:
        logits: A tensor of shape ``(time, batch, vocabulary)`` which contains
            the unnormalized output scores of words in a vocabulary.
        decoder_outputs: A tensor of shape ``(time, batch, state_size)``. The
            states of the decoder before the final output (logit) projection.
        outputs: An int tensor of shape ``(time, batch)``. Stores the generated
            symbols. (Either an argmax-ed value from the logits, or a target
            token, during training.)
        mask: A float tensor of zeros and ones of shape ``(time, batch)``.
            Keeps track of valid positions in the decoded data.
    """


class DecoderConstants(NamedTuple(
        "DecoderConstants",
        [("train_inputs", Optional[tf.Tensor])])):
    """The constants used by an autoregressive decoder.

    Attributes:
        train_inputs: During training, this is populated by the target token
            ids.
    """


class DecoderFeedables(NamedTuple(
        "DecoderFeedables",
        [("step", tf.Tensor),
         ("finished", tf.Tensor),
         ("input_symbol", tf.Tensor),
         ("prev_logits", tf.Tensor)])):
    """The input of a single step of an autoregressive decoder.

    Attributes:
        step: A scalar int tensor, stores the number of the current time step.
        finished: A boolean tensor of shape ``(batch)``,  which says whether
            the decoding of a sentence in the batch is finished or not. (E.g.
            whether the end token has already been generated.)
        input_symbol: A boolean ``batch``-sized tensor with the inputs to the
            decoder. During inference, this contains the previously generated
            tokens. During training, this contains the reference tokens.
        prev_logits: A tensor of shape ``(batch, vocabulary)``. Contains the
            logits from the previous decoding step.
    """


# pylint: disable=too-many-public-methods,too-many-instance-attributes
class AutoregressiveDecoder(ModelPart):

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 dropout_keep_prob: float = 1.0,
                 embedding_size: int = None,
                 embeddings_source: EmbeddedSequence = None,
                 tie_embeddings: bool = False,
                 label_smoothing: float = None,
                 supress_unk: bool = False,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize parameters common for all autoregressive decoders.

        Arguments:
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects.
            vocabulary: Target vocabulary.
            data_id: Target data series.
            max_output_len: Maximum length of an output sequence.
            reuse: Reuse the variables from the model part.
            dropout_keep_prob: Probability of keeping a value during dropout.
            embedding_size: Size of embedding vectors for target words.
            embeddings_source: Embedded sequence to take embeddings from.
            tie_embeddings: Use decoder.embedding_matrix also in place
                of the output decoding matrix.
            label_smoothing: Label smoothing parameter.
            supress_unk: If true, decoder will not produce symbols for unknown
                tokens.
        """
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        log("Initializing decoder, name: '{}'".format(name))

        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_output_len = max_output_len
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.embeddings_source = embeddings_source
        self.label_smoothing = label_smoothing
        self.tie_embeddings = tie_embeddings
        self.supress_unk = supress_unk

        self.encoder_states = []  # type: List[tf.Tensor]
        self.encoder_masks = []  # type: List[tf.Tensor]

        # Check the values of the parameters (max_output_len, ...)
        if max_output_len <= 0:
            raise ValueError("Maximum sequence length must be "
                             "a positive integer.")

        if dropout_keep_prob < 0.0 or dropout_keep_prob > 1.0:
            raise ValueError("Dropout keep probability must be"
                             "a real number in the interval [0,1].")

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

        with self.use_scope():
            self.go_symbols = tf.placeholder(tf.int32, [None], "go_symbols")

            self.train_inputs = tf.placeholder(
                tf.int32, [None, None], "train_inputs")
            self.train_mask = tf.placeholder(
                tf.float32, [None, None], "train_mask")
    # pylint: enable=too-many-arguments,too-many-locals

    @tensor
    def decoding_w(self) -> tf.Variable:
        if (self.tie_embeddings
                and self.embedding_size != self.output_dimension):
            raise ValueError(
                "`embedding_size must be equal to the output_projection "
                "size when using the `tie_embeddings` option")

        with tf.name_scope("output_projection"):
            if self.tie_embeddings:
                return tf.transpose(self.embedding_matrix)

            return get_variable(
                "state_to_word_W",
                [self.output_dimension, len(self.vocabulary)],
                initializer=tf.random_uniform_initializer(-0.5, 0.5))

    @tensor
    def decoding_b(self) -> Optional[tf.Variable]:
        if self.tie_embeddings:
            return tf.zeros(len(self.vocabulary))

        with tf.name_scope("output_projection"):
            return get_variable(
                "state_to_word_b",
                [len(self.vocabulary)],
                initializer=tf.zeros_initializer())

    @tensor
    def embedding_matrix(self) -> tf.Variable:
        """Variables and operations for embedding of input words.

        If we are reusing word embeddings, this function takes the embedding
        matrix from the first encoder
        """
        if self.embeddings_source is not None:
            return self.embeddings_source.embedding_matrix

        assert self.embedding_size is not None

        return get_variable(
            name="word_embeddings",
            shape=[len(self.vocabulary), self.embedding_size])

    def get_logits(self, state: tf.Tensor) -> tf.Tensor:
        """Project the decoder's output layer to logits over the vocabulary."""
        state = dropout(state, self.dropout_keep_prob, self.train_mode)
        logits = tf.matmul(state, self.decoding_w) + self.decoding_b

        if self.supress_unk:
            unk_mask = tf.one_hot(
                UNK_TOKEN_INDEX, depth=len(self.vocabulary), on_value=-1e9)
            logits += unk_mask

        return logits

    @tensor
    def train_loop_result(self) -> Tuple[tf.Tensor, tf.Tensor,
                                         tf.Tensor, tf.Tensor]:
        return self.decoding_loop(train_mode=True)

    @tensor
    def train_logits(self) -> tf.Tensor:
        # THE LAST TRAIN INPUT IS NOT USED IN DECODING FUNCTION
        # (just as a target)
        return tuple(self.train_loop_result)[0]

    @tensor
    def train_output_states(self) -> tf.Tensor:
        return tuple(self.train_loop_result)[1]

    @tensor
    def train_logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.train_logits)

    @tensor
    def train_xents(self) -> tf.Tensor:
        train_targets = tf.transpose(self.train_inputs)
        softmax_function = None
        if self.label_smoothing:
            softmax_function = (
                lambda labels, logits: tf.losses.softmax_cross_entropy(
                    tf.one_hot(labels, len(self.vocabulary)),
                    logits, label_smoothing=self.label_smoothing))

        return tf.contrib.seq2seq.sequence_loss(
            tf.transpose(self.train_logits, perm=[1, 0, 2]),
            train_targets,
            tf.transpose(self.train_mask),
            average_across_batch=False,
            softmax_loss_function=softmax_function)

    @tensor
    def train_loss(self) -> tf.Tensor:
        lenghts = tf.reduce_sum(self.train_mask, axis=1)

        return (tf.reduce_sum(self.train_xents * lengths)
                / tf.reduce_sum(lengths))

    @property
    def cost(self) -> tf.Tensor:
        return self.train_loss

    @tensor
    def runtime_loop_result(self) -> Tuple[tf.Tensor, tf.Tensor,
                                           tf.Tensor, tf.Tensor]:
        return self.decoding_loop(train_mode=False)

    @tensor
    def runtime_logits(self) -> tf.Tensor:
        return tuple(self.runtime_loop_result)[0]

    @tensor
    def runtime_output_states(self) -> tf.Tensor:
        return tuple(self.runtime_loop_result)[1]

    @tensor
    def runtime_mask(self) -> tf.Tensor:
        return tuple(self.runtime_loop_result)[2]

    @tensor
    def decoded(self) -> tf.Tensor:
        # We disable generating of <pad> tokens at index 0
        # (self.runtime_logits[:, :, 1:]). This shifts the indices
        # of the decoded tokens (therefore, we add +1 to the decoded
        # output indices).

        # self.runtime_logits is of size [batch, sentence_len, vocabulary_size]
        return tf.argmax(self.runtime_logits[:, :, 1:], -1) + 1

    @tensor
    def runtime_xents(self) -> tf.Tensor:
        train_targets = tf.transpose(self.train_inputs)
        batch_major_logits = tf.transpose(self.runtime_logits, [1, 0, 2])
        min_time = tf.minimum(tf.shape(train_targets)[1],
                              tf.shape(batch_major_logits)[1])

        # NOTE if done properly, there should be padding of the shorter
        # sequence instead of cropping to the length of the shorter one

        return tf.contrib.seq2seq.sequence_loss(
            logits=batch_major_logits[:, :min_time],
            targets=train_targets[:, :min_time],
            weights=tf.transpose(self.train_mask)[:, :min_time],
            average_across_batch=False)

    @tensor
    def runtime_loss(self) -> tf.Tensor:
        return tf.reduce_mean(self.runtime_xents)

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.runtime_logits)

    @property
    def output_dimension(self) -> int:
        raise NotImplementedError("Abstract property")

    def get_initial_loop_state(self) -> LoopState:

        dec_output = tf.zeros(
            shape=[0, self.batch_size, self.embedding_size],
            dtype=tf.float32,
            name="hist_decoder_outputs")

        logit = tf.zeros(
            shape=[0, self.batch_size, len(self.vocabulary)],
            dtype=tf.float32,
            name="hist_logits")

        mask = tf.zeros(
            shape=[0, self.batch_size],
            dtype=tf.bool,
            name="mask")

        outputs = tf.zeros(
            shape=[0, self.batch_size],
            dtype=tf.int32,
            name="outputs")

        feedables = DecoderFeedables(
            step=tf.constant(0, tf.int32),
            finished=tf.zeros([self.batch_size], dtype=tf.bool),
            input_symbol=self.go_symbols,
            prev_logits=tf.zeros([self.batch_size, len(self.vocabulary)]))

        histories = DecoderHistories(
            logits=logit,
            decoder_outputs=dec_output,
            mask=mask,
            outputs=outputs)

        constants = DecoderConstants(train_inputs=self.train_inputs)

        return LoopState(
            histories=histories,
            constants=constants,
            feedables=feedables)

    def loop_continue_criterion(self, *args) -> tf.Tensor:
        """Decide whether to break out of the while loop.

        Arguments:
            loop_state: ``LoopState`` instance (see the docs for this module).
                Represents current decoder loop state.
        """
        loop_state = LoopState(*args)
        finished = loop_state.feedables.finished
        not_all_done = tf.logical_not(tf.reduce_all(finished))
        before_max_len = tf.less(loop_state.feedables.step,
                                 self.max_output_len)
        return tf.logical_and(not_all_done, before_max_len)

    def get_body(self, train_mode: bool, sample: bool = False,
                 temperature: float = 1) -> Callable:
        """Return the while loop body function."""
        raise NotImplementedError("Abstract method")

    def finalize_loop(self, final_loop_state: LoopState,
                      train_mode: bool) -> None:
        """Execute post-while loop operations.

        Arguments:
            final_loop_state: Decoder loop state at the end
                of the decoding loop.
            train_mode: Boolean flag, telling whether this is
                a training run.
        """

    def decoding_loop(self, train_mode: bool, sample: bool = False,
                      temperature: float = 1) -> Tuple[tf.Tensor, tf.Tensor,
                                                       tf.Tensor, tf.Tensor]:
        """Run the decoding while loop.

        Calls get_initial_loop_state and constructs tf.while_loop
        with the continuation criterion returned from loop_continue_criterion,
        and body function returned from get_body.

        After finishing the tf.while_loop, it calls finalize_loop
        to further postprocess the final decoder loop state (usually
        by stacking Tensors containing decoding histories).

        Arguments:
            train_mode: Boolean flag, telling whether this is
                a training run.
            sample: Boolean flag, telling whether we should sample
                the output symbols from the output distribution instead
                of using argmax or gold data.
            temperature: float value specifying the softmax temperature
        """
        initial_loop_state = self.get_initial_loop_state()
        final_loop_state = tf.while_loop(
            self.loop_continue_criterion,
            self.get_body(train_mode, sample, temperature),
            initial_loop_state,
            shape_invariants=tf.contrib.framework.nest.map_structure(
                get_state_shape_invariants, initial_loop_state))

        self.finalize_loop(final_loop_state, train_mode)

        logits = final_loop_state.histories.logits
        decoder_outputs = final_loop_state.histories.decoder_outputs
        decoded = final_loop_state.histories.outputs

        # TODO mask should include also the end symbol
        mask = final_loop_state.histories.mask

        return logits, decoder_outputs, mask, decoded

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary for the decoder object.

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run.
        """
        fd = ModelPart.feed_dict(self, dataset, train)

        sentences = dataset.maybe_get_series(self.data_id)

        if sentences is None and train:
            raise ValueError("When training, you must feed "
                             "reference sentences")

        go_symbol_idx = self.vocabulary.get_word_index(START_TOKEN)
        fd[self.go_symbols] = np.full([len(dataset)], go_symbol_idx,
                                      dtype=np.int32)

        if sentences is not None:
            sentences_list = list(sentences)
            # train_mode=False, since we don't want to <unk>ize target words!
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len, train_mode=False,
                add_start_symbol=False, add_end_symbol=True,
                pad_to_max_len=False)

            fd[self.train_inputs] = inputs
            fd[self.train_mask] = weights

        return fd
