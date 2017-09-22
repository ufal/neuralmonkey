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

from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.dataset import Dataset
from neuralmonkey.decoders.sequence_decoder import LoopState
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import (END_TOKEN_INDEX, PAD_TOKEN_INDEX)
from neuralmonkey.decorators import tensor

# pylint: disable=invalid-name
DecoderState = NamedTuple("DecoderState",
                          [("step", tf.Tensor),
                           ("input_symbol", tf.Tensor),
                           ("prev_rnn_state", tf.Tensor),
                           ("prev_rnn_output", tf.Tensor),
                           ("prev_logits", tf.Tensor),
                           ("prev_contexts", List[tf.Tensor]),
                           ("finished", tf.Tensor)])

SearchState = NamedTuple("SearchState",
                         [("logprob_sum", tf.Tensor),  # beam x 1
                          ("prev_logprobs", tf.Tensor),  # beam x Vocabulary
                          ("lengths", tf.Tensor),
                          ("finished", tf.Tensor)])

SearchStepOutput = NamedTuple("SearchStepOutput",
                              [("scores", tf.Tensor),
                               ("parent_ids", tf.Tensor),
                               ("token_ids", tf.Tensor)])

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
                               ("last_dec_loop_state", DecoderState),
                               ("last_search_state", SearchState),
                               ("attention_loop_states", List[Any])])


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

        # In the n+1th step, outputs  of lenght n will be collected
        # and the n+1th step of decoder (which is discarded) will be executed
        if max_steps is None:
            max_steps = parent_decoder.max_output_len
        self._max_steps = tf.constant(max_steps + 1)
        self.max_output_len = max_steps

        # Feedables
        self._search_state = None  # type: SearchState
        self._decoder_state = None  # type: DecoderState

        # Output
        self.outputs = self._decoding_loop()

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
        # TODO: make these feedable
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
        dec_rnn_ls = dec_ls.dec_ls

        # We want to feed these values in ensembles
        self._search_state = SearchState(
            logprob_sum=tf.placeholder_with_default([0.0], [None]),
            prev_logprobs=tf.nn.log_softmax(dec_rnn_ls.prev_logits),
            lengths=tf.placeholder_with_default([1], [None]),
            finished=tf.placeholder_with_default([False], [None]))

        # We create BeamSearchDecoder attributes
        # that can be directly fed from outside
        # the Session.run() due to the logprob recombination
        # in ensembles.
        self._decoder_state = DecoderState(
            step=dec_ls.step,
            input_symbol=dec_rnn_ls.input_symbol,
            prev_rnn_state=dec_rnn_ls.prev_rnn_state,
            prev_rnn_output=dec_rnn_ls.prev_rnn_output,
            prev_logits=dec_rnn_ls.prev_logits,
            prev_contexts=dec_rnn_ls.prev_contexts,
            finished=dec_ls.finished)
        # dec_ls = dec_ls._replace(**self._decoder_state._asdict())
        # this line no longer works

        # TODO:
        # Make TensorArrays also feedable
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
            return tf.less(bsls.decoder_loop_state.step - 1, self._max_steps)

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

        scores = final_state.bs_output.scores.stack()
        parent_ids = final_state.bs_output.parent_ids.stack()
        token_ids = final_state.bs_output.token_ids.stack()

        # TODO: return att_loop_states properly
        return BeamSearchOutput(
            last_search_step_output=SearchStepOutput(
                scores=scores,
                parent_ids=parent_ids,
                token_ids=token_ids),
            last_dec_loop_state=DecoderState(
                step=dec_loop_state.step,
                input_symbol=dec_loop_state.dec_ls.input_symbol,
                prev_rnn_state=dec_loop_state.dec_ls.prev_rnn_state,
                prev_rnn_output=dec_loop_state.dec_ls.prev_rnn_output,
                prev_logits=dec_loop_state.dec_ls.prev_logits,
                prev_contexts=dec_loop_state.dec_ls.prev_contexts,
                finished=dec_loop_state.finished),
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
            assert self.parent_decoder.step_scope.reuse

            # The decoder should be "one step ahead" (see above)
            step = dec_loop_state.step - 1
            # current_logits = dec_loop_state.prev_logits

            # mask the probabilities
            # shape(logprobs) = beam x vocabulary
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
            topk_scores, topk_indices = tf.nn.top_k(
                scores_flat, self._beam_size)

            topk_scores.set_shape([self._beam_size])
            topk_indices.set_shape([self._beam_size])

            # flatten the hypothesis probabilities
            hyp_probs_flat = tf.reshape(hyp_probs, [-1])

            # select logprobs of the best hyps (disregard lenghts)
            next_beam_logprob_sum = tf.gather(hyp_probs_flat, topk_indices)
            # pylint: disable=no-member
            next_beam_logprob_sum.set_shape([self._beam_size])
            # pylint: enable=no-member

            next_word_ids = tf.mod(topk_indices,
                                   len(self.vocabulary))

            next_beam_ids = tf.div(topk_indices,
                                   len(self.vocabulary))

            rnn_state = dec_loop_state.dec_ls.prev_rnn_state
            rnn_output = dec_loop_state.dec_ls.prev_rnn_output
            contexts = dec_loop_state.dec_ls.prev_contexts

            next_beam_prev_rnn_state = tf.gather(rnn_state, next_beam_ids)
            next_beam_prev_rnn_output = tf.gather(rnn_output, next_beam_ids)
            next_beam_prev_contexts = [tf.gather(ctx, next_beam_ids)
                                       for ctx in contexts]
            next_beam_prev_logits = tf.gather(
                dec_loop_state.dec_ls.prev_logits, next_beam_ids)
            next_beam_lengths = tf.gather(hyp_lengths, next_beam_ids)

            next_finished = tf.gather(dec_loop_state.finished, next_beam_ids)
            next_just_finished = tf.equal(next_word_ids, END_TOKEN_INDEX)
            next_finished = tf.logical_or(next_finished, next_just_finished)

            # Update decoder state before computing the next state
            # For run-time computation, the decoder needs:
            # - step
            # - input_symbol
            # - prev_rnn_state
            # - prev_rnn_output
            # - prev_contexts
            # - attention_loop_states
            # - finished

            # For train-mode computation, it also needs
            # - train_inputs

            # For recording the computation in time, it needs
            # - rnn_outputs (TA)
            # - logits (TA)
            # - mask (TA)

            # Because of the beam search algorithm, it outputs
            # (but does not not need)
            # - prev_logits

            # During beam search decoding, we are not interested in recording
            # of the computation as done by the decoder. The record is stored
            # in search states and step outputs of this decoder.
            dec_rnn_ls = dec_loop_state.dec_ls._replace(
                input_symbol=next_word_ids,
                prev_rnn_state=next_beam_prev_rnn_state,
                prev_rnn_output=next_beam_prev_rnn_output,
                prev_logits=next_beam_prev_logits,
                prev_contexts=next_beam_prev_contexts)

            dec_loop_state = dec_loop_state._replace(
                finished=next_finished,
                dec_ls=dec_rnn_ls)

            # CALL THE DECODER BODY FUNCTION
            # TODO figure out why mypy throws too-many-arguments on this
            next_loop_state = decoder_body(*dec_loop_state)  # type: ignore
            next_dec_rnn_ls = next_loop_state.dec_ls

            next_search_state = SearchState(
                logprob_sum=next_beam_logprob_sum,
                prev_logprobs=tf.nn.log_softmax(next_dec_rnn_ls.prev_logits),
                lengths=next_beam_lengths,
                finished=dec_loop_state.finished)

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
        assert not train
        assert len(dataset) == 1

        return {}

    def _length_penalty(self, lengths):
        """Apply lp term from eq. 14."""

        return ((5. + tf.to_float(lengths)) ** self._length_normalization /
                (5. + 1.) ** self._length_normalization)
