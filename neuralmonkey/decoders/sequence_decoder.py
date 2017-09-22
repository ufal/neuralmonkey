"""Abstract class for decoding sequences left-to-right. Either for the
recurrent decoder, or for the transformer decoder.

The sequence decoder uses the while loop to get the outputs. Descendants should
only specify the initial state and the while loop body.
"""
import math
from typing import (NamedTuple, Any, Union, Callable, Tuple, cast, Iterable,
                    List)

import numpy as np
import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary, START_TOKEN

# The LoopState is a structure that works with the tf.while_loop function
# the decoder loop state stores all the information that is not invariant
# for the decoder run.
LoopState = NamedTuple(
    "LoopState",
    [("step", tf.Tensor),  # 1D int, number of the step
     ("finished", tf.Tensor),  # batch-sized, bool
     ("logits", tf.TensorArray),
     ("decoder_outputs", tf.TensorArray),
     ("outputs", tf.TensorArray),
     ("mask", tf.TensorArray),  # float matrix, 0s and 1s
     ("dec_ls", Any)])  # Decoder-specific loop state


# pylint: disable=too-many-public-methods
# TODO More refactoring needed.. (?)
class SequenceDecoder(ModelPart):

    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        log("Initializing decoder, name: '{}'".format(name))

        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_output_len = max_output_len
        self.dropout_keep_prob = dropout_keep_prob

        # TODO check the values of the parameters (vocab, ...)

        with self.use_scope():
            self.train_mode = tf.placeholder(tf.bool, [], "train_mode")
            self.go_symbols = tf.placeholder(tf.int32, [None], "go_symbols")

            self.train_inputs = tf.placeholder(
                tf.int32, [None, None], "train_inputs")
            self.train_mask = tf.placeholder(
                tf.float32, [None, None], "train_mask")

    @tensor
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self.go_symbols)[0]

    @tensor
    def decoding_w(self) -> tf.Variable:
        with tf.name_scope("output_projection"):
            return tf.get_variable(
                "logit_matrix",
                [self.output_dimension, len(self.vocabulary)],
                initializer=tf.random_uniform_initializer(-0.5, 0.5))

    @tensor
    def decoding_b(self) -> tf.Variable:
        with tf.name_scope("output_projection"):
            return tf.get_variable(
                "logit_bias", [len(self.vocabulary)],
                initializer=tf.constant_initializer(
                    - math.log(len(self.vocabulary))))

    def get_logits(self, state: tf.Tensor) -> tf.Tensor:
        state = dropout(state, self.dropout_keep_prob, self.train_mode)
        return tf.matmul(state, self.decoding_w) + self.decoding_b

    @tensor
    def train_logits(self) -> tf.Tensor:
        # THE LAST TRAIN INPUT IS NOT USED IN DECODING FUNCTION
        # (just as a target)
        logits, _, _, _ = self.decoding_loop(train_mode=True)
        return logits

    @tensor
    def train_logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.train_logits)

    @tensor
    def train_xents(self) -> tf.Tensor:
        train_targets = tf.transpose(self.train_inputs)

        return tf.contrib.seq2seq.sequence_loss(
            tf.transpose(self.train_logits, perm=[1, 0, 2]),
            train_targets,
            tf.transpose(self.train_mask),
            average_across_batch=False)

    @tensor
    def train_loss(self) -> tf.Tensor:
        return tf.reduce_mean(self.train_xents)

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
    def runtime_rnn_states(self) -> tf.Tensor:
        return tuple(self.runtime_loop_result)[1]

    @tensor
    def runtime_mask(self) -> tf.Tensor:
        return tuple(self.runtime_loop_result)[2]

    @tensor
    def decoded(self) -> tf.Tensor:
        return tf.argmax(self.runtime_logits[:, :, 1:], -1) + 1

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
            weights=tf.transpose(self.train_mask)[:, :min_time])

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self.runtime_logits)

    @property
    def output_dimension(self) -> Union[int, tf.Tensor]:
        raise NotImplementedError("Abstract property")

    def get_initial_loop_state(self) -> LoopState:
        raise NotImplementedError("Abstract method")

    def loop_continue_criterion(self, *args) -> tf.Tensor:
        loop_state = LoopState(*args)
        finished = loop_state.finished
        not_all_done = tf.logical_not(tf.reduce_all(finished))
        before_max_len = tf.less(loop_state.step,
                                 self.max_output_len)
        return tf.logical_and(not_all_done, before_max_len)

    def get_body(self, train_mode: bool, sample: bool = False) -> Callable:
        raise NotImplementedError("Abstract method")

    def finalize_loop(self, final_loop_state: LoopState,
                      train_mode: bool) -> None:
        pass

    def decoding_loop(self, train_mode: bool, sample: bool = False) -> Tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        initial_loop_state = self.get_initial_loop_state()
        final_loop_state = tf.while_loop(
            self.loop_continue_criterion,
            self.get_body(train_mode, sample),
            initial_loop_state)

        self.finalize_loop(final_loop_state, train_mode)

        logits = final_loop_state.logits.stack()
        decoder_outputs = final_loop_state.decoder_outputs.stack()
        decoded = final_loop_state.outputs.stack()

        # TODO mask should include also the end symbol
        mask = final_loop_state.mask.stack()

        return logits, decoder_outputs, mask, decoded

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
            fd[self.train_mask] = weights

        return fd
