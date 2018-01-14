"""Beam search decoder.

This module implements the beam search algorithm for the recurrent decoder.

As well as the recurrent decoder, this decoder works dynamically, which means
it uses the ``tf.while_loop`` function conditioned on both maximum output
length and list of finished hypotheses.

The beam search decoder works by appending data from ``SearchStepOutput``
objects to a ``SearchStepOutputTA`` object. The ``SearchStepOutput`` object
stores information about the hypotheses in the beam. Each hypothesis keeps its
score, its final token, and a pointer to a "parent" hypothesis, which is a
one-token-shorter hypothesis which shares the tokens with the child hypothesis.

For the beam search decoder to work, it must keep an inner state which stores
information about hypotheses in the beam. It is an object of type
``SearchState`` which stores, *for each hypothesis*, its sum of log
probabilities of the tokens, its length, finished flag, ID of the last token,
and the last decoder and attention states.

There is another inner state object here, the ``BeamSearchLoopState``. It is a
technical structure used with the ``tf.while_loop`` function. It stores all the
previously mentioned information, plus the decoder ``LoopState``, which is used
in the decoder when its own ``tf.while_loop`` function is used - this is not
the case when using beam search because we want to run the decoder's steps
manually.
"""
from typing import NamedTuple, List, Callable, Any

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.dataset import Dataset
from neuralmonkey.decoders.autoregressive import (
    LoopState, AutoregressiveDecoder)
from neuralmonkey.vocabulary import (END_TOKEN_INDEX, PAD_TOKEN_INDEX)
from neuralmonkey.decorators import tensor

# pylint: disable=invalid-name
SearchState = NamedTuple("SearchState",
                         [("input_beam_size", tf.Tensor),  # req. for reshaping
                          ("logprob_sum", tf.Tensor),  # (batch*beam)
                          ("prev_logprobs", tf.Tensor),  # (batch*beam) x Vocab
                          ("lengths", tf.Tensor),  # (batch*beam)
                          ("finished", tf.Tensor)])  # (batch*beam)

SearchStepOutput = NamedTuple("SearchStepOutput",
                              [("scores", tf.Tensor),  # batch x beam
                               ("parent_ids", tf.Tensor),  # batch x beam
                               ("token_ids", tf.Tensor)])  # batch x beam

SearchStepOutputTA = NamedTuple("SearchStepOutputTA",
                                [("scores", tf.TensorArray),
                                 ("parent_ids", tf.TensorArray),
                                 ("token_ids", tf.TensorArray)])

BeamSearchLoopState = NamedTuple("BeamSearchLoopState",
                                 [("bs_state", SearchState),
                                  ("bs_output", SearchStepOutputTA),
                                  ("decoder_loop_state", LoopState)])

BeamSearchOutput = NamedTuple("SearchStepOutput",
                              [("last_search_step_output", SearchStepOutput),
                               ("last_dec_loop_state", NamedTuple),
                               ("last_search_state", SearchState),
                               ("attention_loop_states", List[Any])])


# pylint: enable=invalid-name
class BeamSearchDecoder(ModelPart):
    """In-graph beam search for batch size 1.

    The hypothesis scoring algorithm is taken from
    https://arxiv.org/pdf/1609.08144.pdf. Length normalization is parameter
    alpha from equation 14.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 parent_decoder: AutoregressiveDecoder,
                 beam_size: int,
                 length_normalization: float,
                 max_steps: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.parent_decoder = parent_decoder
        self._beam_size = beam_size
        self._length_normalization = length_normalization

        # The parent_decoder is one step ahead. This is required for ensembling
        # support.
        # At the end of the Nth step we generate logits for ensembling
        # in the N+1th step by the parent_decoder. These need to be first
        # ensembled outside of the session.run before finishing the N+1th
        # step of the beam_search_decoder (collecting topk outputs, selecting
        # beams and running next parent_decoder step based on the chosen beam).
        if max_steps is None:
            max_steps = parent_decoder.max_output_len - 1
        self._max_steps = tf.constant(max_steps)
        self.max_output_len = max_steps

        # Feedables
        self._search_state = None  # type: SearchState
        self._decoder_state = None  # type: NamedTuple

        # Output
        self.outputs = self._decoding_loop()
    # pylint: enable=too-many-arguments

    @property
    def batch_size(self):
        return self.parent_decoder.batch_size

    @property
    def beam_size(self):
        return self._beam_size

    @property
    def vocabulary(self):
        return self.parent_decoder.vocabulary

    @tensor
    def search_state(self):
        return self._search_state

    @tensor
    def decoder_state(self):
        return self._decoder_state

    @tensor
    def max_steps(self):
        return self._max_steps

    def get_initial_loop_state(self) -> BeamSearchLoopState:
        # TODO make these feedable
        output_ta = SearchStepOutputTA(
            scores=tf.TensorArray(dtype=tf.float32, dynamic_size=True,
                                  size=0, name="beam_scores"),
            parent_ids=tf.TensorArray(dtype=tf.int32, dynamic_size=True,
                                      size=0, name="beam_parents"),
            token_ids=tf.TensorArray(dtype=tf.int32, dynamic_size=True,
                                     size=0, name="beam_tokens"))

        # We run the decoder once to get logits for ensembling
        dec_ls = self.parent_decoder.get_initial_loop_state()
        decoder_body = self.parent_decoder.get_body(False)
        dec_ls = decoder_body(*dec_ls)

        # We want to feed these values in ensembles
        self._search_state = SearchState(
            input_beam_size=tf.placeholder_with_default(
                input=1, shape=[], name="input_beam_size"),
            logprob_sum=tf.placeholder_with_default(
                input=[0.0], shape=[None], name="bs_logprob_sum"),
            prev_logprobs=tf.nn.log_softmax(dec_ls.feedables.prev_logits),
            lengths=tf.placeholder_with_default(
                input=[0], shape=[None], name="bs_lengths"),
            finished=tf.zeros([self.batch_size], dtype=tf.bool))

        self._decoder_state = dec_ls.feedables

        # TODO make TensorArrays also feedable
        return BeamSearchLoopState(
            bs_state=self._search_state,
            bs_output=output_ta,
            decoder_loop_state=dec_ls)

    def _decoding_loop(self) -> BeamSearchOutput:
        # collect attention objects
        beam_body = self.get_body()

        initial_loop_state = self.get_initial_loop_state()

        def cond(*args) -> tf.Tensor:
            bsls = BeamSearchLoopState(*args)
            return tf.less(
                bsls.decoder_loop_state.feedables.step - 1, self._max_steps)

        # First step has to be run manually because while_loop needs the same
        # shapes between steps and the first beam state is not beam-sized, but
        # just a single state.
        #
        # When running ensembles, we want to provide
        # ensembled logprobs to the beam_body before manually running
        # the first step
        next_bs_loop_state = tf.cond(
            cond(*initial_loop_state),
            lambda: beam_body(*initial_loop_state),
            lambda: initial_loop_state)

        final_state = tf.while_loop(cond, beam_body, next_bs_loop_state)
        dec_loop_state = final_state.decoder_loop_state
        bs_state = final_state.bs_state

        # Because the "batch" dimension is dynamic and the TensorArrays
        # are empty during initial call during ensembling, we have
        # to define default values (the runner itself has to handle them,
        # if needed)
        # NOTE this code might change in the future, if tf.cond gets replaced
        # by something more reasonable
        scores = tf.cond(
            tf.equal(final_state.bs_output.scores.size(), 0),
            lambda: tf.fill([1, self.batch_size, self.beam_size], 0.0),
            final_state.bs_output.scores.stack)
        parent_ids = tf.cond(
            tf.equal(final_state.bs_output.parent_ids.size(), 0),
            lambda: tf.fill([1, self.batch_size, self.beam_size], 0),
            final_state.bs_output.parent_ids.stack)
        token_ids = tf.cond(
            tf.equal(final_state.bs_output.token_ids.size(), 0),
            lambda: tf.fill([1, self.batch_size, self.beam_size], 0),
            final_state.bs_output.token_ids.stack)

        # TODO: return att_loop_states properly
        return BeamSearchOutput(
            last_search_step_output=SearchStepOutput(
                scores=scores,
                parent_ids=parent_ids,
                token_ids=token_ids),
            last_dec_loop_state=dec_loop_state.feedables,
            last_search_state=bs_state,
            attention_loop_states=[])

    def get_body(self) -> Callable:
        """Return a body function for ``tf.while_loop``."""
        decoder_body = self.parent_decoder.get_body(train_mode=False)

        # pylint: disable=too-many-locals
        def body(*args) -> BeamSearchLoopState:
            """Execute a single beam search step.

            The beam search body function.
            This is where the beam search algorithm is implemented.

            Arguments:
                loop_state: ``BeamSearchLoopState`` instance (see the docs
                    for this module)
            """

            # We expect that we already executed decoder_body once
            # before entering the while_loop.
            # This is because we need to give the BeamSearchDecoder
            # body function updated logits (in case of ensembles)
            # to correctly choose next beam
            #
            # For this reason, it is recommended to run the decoder
            # for n+1 steps, if you want to generate n steps by the decoder.
            loop_state = BeamSearchLoopState(*args)
            dec_loop_state = loop_state.decoder_loop_state
            bs_state = loop_state.bs_state
            bs_output = loop_state.bs_output

            # Don't want to use this decoder with uninitialized parent
            # assert self.parent_decoder.step_scope.reuse

            # The decoder should be "one step ahead" (see above)
            step = dec_loop_state.feedables.step - 1

            # mask the probabilities
            # shape(logprobs) = (batch*beam) x vocabulary
            logprobs = bs_state.prev_logprobs

            finished_mask = tf.expand_dims(tf.to_float(bs_state.finished), 1)
            unfinished_logprobs = (1. - finished_mask) * logprobs

            finished_row = tf.one_hot(
                PAD_TOKEN_INDEX,
                len(self.vocabulary),
                dtype=tf.float32,
                on_value=0.,
                off_value=tf.float32.min)

            finished_logprobs = finished_mask * finished_row
            logprobs = unfinished_logprobs + finished_logprobs

            # update hypothesis scores
            # shape(hyp_probs) = (batch*beam) x vocabulary
            hyp_probs = tf.expand_dims(bs_state.logprob_sum, 1) + logprobs

            # update hypothesis lengths
            hyp_lengths = bs_state.lengths + 1 - tf.to_int32(bs_state.finished)

            # shape(scores) = (batch*beam) x vocabulary
            scores = hyp_probs / tf.expand_dims(
                self._length_penalty(hyp_lengths), 1)

            # reshape to batch x (beam*vocabulary) for topk
            scores_flat = tf.reshape(
                scores, [-1, bs_state.input_beam_size * len(self.vocabulary)])

            # shape(both) = batch x beam
            topk_scores, topk_indices = tf.nn.top_k(
                scores_flat, self._beam_size)
            topk_indices.set_shape([None, self._beam_size])
            topk_scores.set_shape([None, self._beam_size])

            # flatten the hypothesis probabilities
            hyp_probs_flat = tf.reshape(hyp_probs, [-1])

            # First, we need the starting offsets of each batch
            # with respect to the shape(hyp_probs)[-1] to flatten topk
            # indices so we can gather correct hyp_probs values.
            #
            # ("|" marks beam border, "||" marks batch border)
            # e.g. scores (vocabulary=2, beam_size=3, batch_size=2):
            #
            # [ 2.0, 3.0 | 1.0, 4.0 | 2.0, 5.0 ]
            # [ 1.0, 2.0 | 3.0, 4.0 | 5.0, 1.0 ]
            #
            # topk(k=3) produces following indices:
            # [ 5, 3, 1 ]
            # [ 4, 3, 2 ]
            #
            # To correctly gather values from flattened hyp_probs
            # shape(hyp_probs_flat) = (batch*beam*vocabulary) we compute
            # the offsets of the values starting at each batch (||), so
            # the following holds:
            # [ 0 .. (beam*voc) - 1 || (beam*voc) + 0 .. 2*(beam*voc) - 1 ||
            #   2*(beam*voc) + 0 .. 3*(beam*voc) - 1 || .. ]
            #
            # In the case of topk from above:
            # [ 0*(3*2) + 5, 0*(3*2) + 3, 0*(3*2) + 1 ||
            #   1*(3*2) + 4, 1*(3*2) + 3, 1*(3*2) + 2 ]
            # or
            # [ 5,  3, 1 ||
            #   10, 9, 8 ]
            beam_voc_offset = tf.expand_dims(
                tf.range(
                    start=0,
                    limit=(self.batch_size * bs_state.input_beam_size
                           * len(self.vocabulary)),
                    delta=(bs_state.input_beam_size * len(self.vocabulary))),
                axis=1)
            topk_indices_flat = tf.reshape(
                topk_indices + beam_voc_offset, [-1])

            # select logprobs of the best hyps (disregard lenghts)
            next_beam_logprob_sum = tf.gather(hyp_probs_flat,
                                              topk_indices_flat)

            # Next, we compute the starting offsets of each batch with respect
            # to the beam_size to flatten beam_id indices
            # to gather correct beams from the respective batches.
            #
            # Similar to the previous offset computation however,
            # now we have to recompute beam_ids with regard to their
            # corresponding batch_id (again, the values are stargin at each
            # batch, ||).
            #
            # e.g beam_ids (beam_size=5, batch_size=2).
            # [ 3, 4, 1 ]
            # [ 2, 0, 2 ]
            #
            # ...after reshaping should be:
            # [ 0*(5) + 3, 0*(5) + 4, 0*(5) + 1 ||
            #   1*(5) + 2, 1*(5) + 0, 1*(5) + 2 ]
            # or
            # [ 3, 4, 1 ||
            #   7, 5, 7 ]
            beam_offset = tf.expand_dims(
                tf.range(
                    start=0,
                    limit=(self.batch_size * bs_state.input_beam_size),
                    delta=bs_state.input_beam_size),
                axis=1)

            next_word_ids = tf.mod(topk_indices, len(self.vocabulary))
            next_beam_ids = tf.div(topk_indices, len(self.vocabulary))

            next_word_ids_flat = tf.reshape(next_word_ids, [-1])
            next_beam_ids_flat = tf.reshape(
                next_beam_ids + beam_offset, [-1])

            next_finished = tf.gather(bs_state.finished, next_beam_ids_flat)
            next_just_finished = tf.equal(next_word_ids_flat, END_TOKEN_INDEX)
            next_finished = tf.logical_or(next_finished, next_just_finished)

            next_feedables_dict = {
                "input_symbol": next_word_ids_flat,
                "finished": next_finished}
            for key, val in dec_loop_state.feedables._asdict().items():
                # Note that the parent decoder is working with "batches"
                # of the size (batch*beam)

                if key in ["step", "input_symbol", "finished"]:
                    continue

                if isinstance(val, tf.Tensor):
                    next_feedables_dict[key] = tf.gather(val,
                                                         next_beam_ids_flat)
                elif isinstance(val, list):
                    if not all(isinstance(t, tf.Tensor) for t in val):
                        raise TypeError("Expected tf.Tensor among feedables")

                    next_feedables_dict[key] = [
                        tf.gather(t, next_beam_ids_flat) for t in val]
                else:
                    raise TypeError("Expected only tensors or list of tensors "
                                    "among feedables")
            next_beam_lengths = tf.gather(hyp_lengths, next_beam_ids_flat)

            # During beam search decoding, we are not interested in recording
            # of the computation as done by the decoder. The record is stored
            # in search states and step outputs of this decoder.
            next_feedables = dec_loop_state.feedables._replace(
                **next_feedables_dict)

            dec_loop_state = dec_loop_state._replace(feedables=next_feedables)

            # CALL THE DECODER BODY FUNCTION
            # TODO figure out why mypy throws too-many-arguments on this
            next_loop_state = decoder_body(*dec_loop_state)  # type: ignore

            next_search_state = SearchState(
                input_beam_size=self.beam_size,
                logprob_sum=next_beam_logprob_sum,
                prev_logprobs=tf.nn.log_softmax(
                    next_loop_state.feedables.prev_logits),
                lengths=next_beam_lengths,
                finished=next_finished)

            next_output = SearchStepOutputTA(
                scores=bs_output.scores.write(step, topk_scores),
                parent_ids=bs_output.parent_ids.write(step, next_beam_ids),
                token_ids=bs_output.token_ids.write(step, next_word_ids))

            return BeamSearchLoopState(
                bs_state=next_search_state,
                bs_output=next_output,
                decoder_loop_state=next_loop_state)
        # pylint: enable=too-many-locals

        return body

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary for the decoder object.

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run
        """
        return {}

    def _length_penalty(self, lengths):
        """Apply lp term from eq. 14."""

        return ((5. + tf.to_float(lengths)) ** self._length_normalization /
                (5. + 1.) ** self._length_normalization)
