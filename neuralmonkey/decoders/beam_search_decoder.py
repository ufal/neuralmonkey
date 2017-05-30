from typing import NamedTuple, Tuple, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoding_function import BaseAttention
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.dataset import Dataset
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import (START_TOKEN_INDEX, END_TOKEN_INDEX,
                                     PAD_TOKEN_INDEX)

# pylint: disable=invalid-name
SearchState = NamedTuple("SearchState",
                         [("logprob_sum", tf.Tensor),  # beam x 1
                          ("lengths", tf.Tensor),  # beam x 1
                          ("finished", tf.Tensor),  # beam x 1
                          ("last_word_ids", tf.Tensor),  # beam x 1
                          ("last_state", tf.Tensor),  # beam x rnn_size
                          ("last_attns", tf.Tensor)])  # beam x ???

SearchStepOutput = NamedTuple("SearchStepOutput",
                              [("scores", tf.Tensor),
                               ("parent_ids", tf.Tensor),
                               ("token_ids", tf.Tensor)])
# pylint: enable=invalid-name


class BeamSearchDecoder(ModelPart):
    """In-graph beam search for batch size 1.

    The hypothesis scoring algorithm is taken from
    https://arxiv.org/pdf/1609.08144.pdf. Length normalization is parameter
    alpha from equation 14.
    """
    def __init__(self,
                 name: str,
                 parent_decoder: Decoder,
                 beam_size: int,
                 length_normalization: float,
                 max_steps: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        self.parent_decoder = parent_decoder
        self._beam_size = beam_size
        self._length_normalization = length_normalization

        self._max_steps = max_steps
        if self._max_steps is None:
            self._max_steps = parent_decoder.max_output_len

        self.outputs = self._decoding_loop()

    @property
    def beam_size(self):
        return self._beam_size

    @property
    def vocabulary(self):
        return self.parent_decoder.vocabulary

    def _decoding_loop(self):
        # collect attention objects
        att_objects = [self.parent_decoder.get_attention_object(e, False)
                       for e in self.parent_decoder.encoders]
        att_objects = [a for a in att_objects if a is not None]

        state = SearchState(
            logprob_sum=tf.zeros([1]),
            lengths=tf.ones([1], dtype=tf.int32),
            finished=tf.zeros([1], dtype=tf.bool),
            last_word_ids=tf.expand_dims(tf.constant(START_TOKEN_INDEX), 0),
            last_state=self.parent_decoder.initial_state,
            last_attns=[tf.zeros([1, a.attn_size]) for a in att_objects]
        )

        # TODO rewrite using tf.while_loop

        outputs = []
        for _ in range(self._max_steps):
            state, output = self.step(att_objects, state)
            outputs.append(output)

        return outputs

    # pylint: disable=too-many-locals
    def step(self,
             att_objects: List[BaseAttention],
             bs_state: SearchState) -> Tuple[SearchState, SearchStepOutput]:

        # embed the previously decoded word
        input_ = self.parent_decoder.embed_and_dropout(
            bs_state.last_word_ids)

        # don't want to use this decoder with uninitialized parent
        assert self.parent_decoder.step_scope.reuse

        # run the parent decoder decoding step
        # shapes:
        # logits: beam x vocabulary
        # state: beam x rnn_size
        # attns: encoder x beam x context vector size
        logits, state, attns = self.parent_decoder.step(
            att_objects, input_, bs_state.last_state, bs_state.last_attns)

        # mask the probabilities
        # shape(logprobs) = beam x vocabulary
        logprobs = tf.nn.log_softmax(logits)

        finished_mask = tf.expand_dims(tf.to_float(bs_state.finished), 1)
        unfinished_logprobs = (1. - finished_mask) * logprobs

        finished_row = tf.one_hot(
            PAD_TOKEN_INDEX,
            len(self.parent_decoder.vocabulary),
            dtype=tf.float32,
            on_value=0.,
            off_value=tf.float32.min)

        finished_logprobs = finished_mask * finished_row
        logprobs = unfinished_logprobs + finished_logprobs

        # update hypothesis scores
        # shape(hyp_probs) = beam x vocabulary
        hyp_probs = tf.expand_dims(bs_state.logprob_sum, 1) + logprobs

        # update hypothesis lengths
        hyp_lengths = bs_state.lengths + 1 - tf.to_int32(bs_state.finished)

        # shape(scores) = beam x vocabulary
        scores = hyp_probs / tf.expand_dims(
            self._length_penalty(hyp_lengths), 1)

        # flatten so we can use top_k
        scores_flat = tf.reshape(scores, [-1])

        # shape(both) = beam
        topk_scores, topk_indices = tf.nn.top_k(scores_flat, self._beam_size)

        topk_scores.set_shape([self._beam_size])
        topk_indices.set_shape([self._beam_size])

        # flatten the hypothesis probabilities
        hyp_probs_flat = tf.reshape(hyp_probs, [-1])

        # select logprobs of the best hyps (disregard lenghts)
        next_logprob_sum = tf.gather(hyp_probs_flat, topk_indices)
        # pylint: disable=no-member
        next_logprob_sum.set_shape([self._beam_size])
        # pylint: enable=no-member

        next_word_ids = tf.mod(topk_indices,
                               len(self.parent_decoder.vocabulary))

        next_beam_ids = tf.div(topk_indices,
                               len(self.parent_decoder.vocabulary))

        next_beam_prev_state = tf.gather(state, next_beam_ids)
        next_beam_prev_attns = [tf.gather(a, next_beam_ids) for a in attns]
        next_lengths = tf.gather(hyp_lengths, next_beam_ids)

        # update finished flags
        has_just_finished = tf.equal(next_word_ids, END_TOKEN_INDEX)
        next_finished = tf.logical_or(
            tf.gather(bs_state.finished, next_beam_ids),
            has_just_finished)

        output = SearchStepOutput(
            scores=topk_scores,
            parent_ids=next_beam_ids,
            token_ids=next_word_ids)

        search_state = SearchState(
            logprob_sum=next_logprob_sum,
            lengths=next_lengths,
            finished=next_finished,
            last_word_ids=next_word_ids,
            last_state=next_beam_prev_state,
            last_attns=next_beam_prev_attns)

        return search_state, output
    # pylint: enable=too-many-locals

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary for the decoder object

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run
        """
        assert not train
        assert len(dataset) == 1

        return {}

    def _length_penalty(self, lengths):
        """lp term from eq. 14"""

        return ((5. + tf.to_float(lengths)) ** self._length_normalization /
                (5. + 1.) ** self._length_normalization)
