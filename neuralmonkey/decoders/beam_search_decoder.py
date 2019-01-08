"""Beam search decoder.

This module implements the beam search algorithm for autoregressive decoders.

As any autoregressive decoder, this decoder works dynamically, which means
it uses the ``tf.while_loop`` function conditioned on both maximum output
length and list of finished hypotheses.

The beam search decoder uses four data strcutures during the decoding process.
``SearchState``, ``SearchResults``, ``BeamSearchLoopState``, and
``BeamSearchOutput``. The purpose of these is described in their own docstring.

These structures help the decoder to keep track of the decoding, enabling it
to be called e.g. during ensembling, when the content of the structures can be
changed and then fed back to the model.

The implementation mimics the API of the ``AutoregressiveDecoder`` class. There
are functions that prepare and return values that are supplied to the
``tf.while_loop`` function.

"""
# pylint: disable=too-many-lines
# Maybe move the definitions of the named tuple structures to a separate file?
from typing import Any, Callable, List, NamedTuple
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.autoregressive import (
    AutoregressiveDecoder, LoopState)
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.tf_utils import (
    append_tensor, gather_flat, get_state_shape_invariants, partial_transpose,
    get_shape_list)
from neuralmonkey.vocabulary import (
    Vocabulary, END_TOKEN_INDEX, PAD_TOKEN_INDEX)

# Constant we use in place of the np.inf
INF = 1e9


class SearchState(NamedTuple(
        "SearchState",
        [("logprob_sum", tf.Tensor),
         ("prev_logprobs", tf.Tensor),
         ("lengths", tf.Tensor),
         ("finished", tf.Tensor)])):
    """Search state of a beam search decoder.

    This structure keeps track of a current state of the beam search
    algorithm. The search state contains tensors that represent hypotheses in
    the beam, namely their log probability, length, and distribution over the
    vocabulary when decoding the last word, as well as if the hypothesis is
    finished or not.

    Attributes:
        logprob_sum: A ``(batch, beam)``-shaped tensor with the sums of token
            log-probabilities of each hypothesis.
        prev_logprobs: A ``(batch, beam, vocabulary)``-sized tensor. Stores
            the log-distribution over the vocabulary from the previous decoding
            step for each hypothesis.
        lengths: A ``(batch, beam)``-shaped tensor with the lengths of the
            hypotheses.
        finished: A boolean tensor with shape ``(batch, beam)``. Marks finished
            and unfinished hypotheses.
    """


class SearchResults(NamedTuple(
        "SearchResults",
        [("scores", tf.Tensor),
         ("token_ids", tf.Tensor)])):
    """The intermediate results of the beam search decoding.

    A cummulative structure that holds the actual decoded tokens and hypotheses
    scores (after applying a length penalty term).

    Attributes:
        scores: A ``(time, batch, beam)``-shaped tensor with the scores for
            each hypothesis. The score is computed from the ``logprob_sum`` of
            a hypothesis and accounting for the hypothesis length.
        token_ids: A ``(time, batch, beam)``-shaped tensor with the vocabulary
            indices of the tokens in each hypothesis.
    """


class BeamSearchLoopState(NamedTuple(
        "BeamSearchLoopState",
        [("search_state", SearchState),
         ("search_results", SearchResults),
         ("decoder_loop_state", LoopState)])):
    """The loop state of the beam search decoder.

    A loop state object that is used for transferring data between cycles
    through the symbolic while loop. It groups together the ``SearchState`` and
    ``SearchResults`` structures and also keeps track of the underlying decoder
    loop state.

    Attributes:
        search_state: A ``SearchState`` object representing the current search
            state.
        search_results: The growing ``SearchResults`` object which accummulates
            the outputs of the decoding process.
        decoder_loop_state: The current loop state of the underlying
            autoregressive decoder.
    """


class BeamSearchOutput(NamedTuple(
        "BeamSearchOutput",
        [("last_search_step_output", SearchResults),
         ("last_dec_loop_state", NamedTuple),
         ("last_search_state", SearchState),
         ("attention_loop_states", List[Any])])):
    """The final structure that is returned from the while loop.

    Attributes:
        last_search_step_output: A populated ``SearchResults`` object.
        last_dec_loop_state: Final loop state of the underlying decoder.
        last_search_state: Final loop state of the beam search decoder.
        attention_loop_states: The final loop states of the attention objects.
    """


class BeamSearchDecoder(ModelPart):
    """In-graph beam search decoder.

    The hypothesis scoring algorithm is taken from
    https://arxiv.org/pdf/1609.08144.pdf. Length normalization is parameter
    alpha from equation 14.
    """

    def __init__(self,
                 name: str,
                 parent_decoder: AutoregressiveDecoder,
                 beam_size: int,
                 max_steps: int,
                 length_normalization: float) -> None:
        """Construct the beam search decoder graph.

        Arguments:
            name: The name for the model part.
            parent_decoder: An autoregressive decoder from which to sample.
            beam_size: The number of hypotheses in the beam.
            max_steps: The maximum number of time steps to perform.
            length_normalization: The alpha parameter from Eq. 14 in the paper.
        """
        check_argument_types()
        ModelPart.__init__(self, name)

        self.parent_decoder = parent_decoder
        self.beam_size = beam_size
        self.length_normalization = length_normalization
        self.max_steps_int = max_steps

        # Create a placeholder for maximum number of steps that is necessary
        # during ensembling, when the decoder is called repetitively with the
        # max_steps attribute set to one.
        self.max_steps = tf.placeholder_with_default(self.max_steps_int, [])

        self._initial_loop_state = None  # type: Optional[BeamSearchLoopState]

    @tensor
    def outputs(self) -> tf.Tensor:
        # This is an ugly hack for handling the whole graph when expanding to
        # the beam. We need to access all the inner states of the network in
        # the graph, replace them with beam-size-times copied originals, create
        # the beam search graph, and then replace the inner states back.

        enc_states = self.parent_decoder.encoder_states
        enc_masks = self.parent_decoder.encoder_masks

        setattr(self.parent_decoder, "encoder_states",
                lambda: [self.expand_to_beam(sts) for sts in enc_states()])
        setattr(self.parent_decoder, "encoder_masks",
                lambda: [self.expand_to_beam(mask) for mask in enc_masks()])

        # Create the beam search symbolic graph.
        with self.use_scope():
            self._initial_loop_state = self.get_initial_loop_state()
            outputs = self.decoding_loop()

        # Reassign the original encoder states and mask back
        setattr(self.parent_decoder, "encoder_states", enc_states)
        setattr(self.parent_decoder, "encoder_masks", enc_masks)

        return outputs

    @property
    def initial_loop_state(self) -> BeamSearchLoopState:
        if self._initial_loop_state is None:
            raise RuntimeError("Initial loop state was not initialized")
        return self._initial_loop_state

    @property
    def vocabulary(self) -> Vocabulary:
        return self.parent_decoder.vocabulary

    # Note that the attributes search_state, decoder_state, and search_results
    # are used only when ensembling, which is done with max_steps set to one
    # and calling the beam search decoder repetitively.
    @tensor
    def search_state(self) -> SearchState:
        return self.initial_loop_state.search_state

    @tensor
    def decoder_state(self) -> LoopState:
        return self.initial_loop_state.decoder_loop_state

    @tensor
    def search_results(self) -> SearchResults:
        return self.initial_loop_state.search_results

    def get_initial_loop_state(self) -> BeamSearchLoopState:
        """Construct the initial loop state for the beam search decoder.

        During the construction, the body function of the underlying decoder
        is called once to retrieve the initial log probabilities of the first
        token.

        The values are initialized as follows:

        - ``search_state``
            - ``logprob_sum`` - For each sentence in batch, logprob sum of the
              first hypothesis in the beam is set to zero while the others are
              set to negative infinity.
            - ``prev_logprobs`` - This is the softmax over the logits from the
              initial decoder step.
            - ``lengths`` - All zeros.
            - ``finshed`` - All false.

        - ``search_results``
            - ``scores`` - A (batch, beam)-sized tensor of zeros.
            - ``token_ids`` - A (1, batch, beam)-sized tensor filled with
              indices of decoder-specific initial input symbols (usually start
              symbol IDs).

        - ``decoder_loop_state`` - The loop state of the underlying
            autoregressive decoder, as returned from the initial call to the
            body function.

        Returns:
            A populated ``BeamSearchLoopState`` structure.
        """
        # Get the initial loop state of the underlying decoder. Then, expand
        # the tensors from the loop state to (batch * beam) and inject them
        # back into the decoder loop state.

        dec_init_ls = self.parent_decoder.get_initial_loop_state()

        feedables = tf.contrib.framework.nest.map_structure(
            self.expand_to_beam, dec_init_ls.feedables)
        histories = tf.contrib.framework.nest.map_structure(
            lambda x: self.expand_to_beam(x, dim=1), dec_init_ls.histories)

        constants = tf.constant(0)
        if dec_init_ls.constants:
            constants = tf.contrib.framework.nest.map_structure(
                self.expand_to_beam, dec_init_ls.constants)

        dec_init_ls = dec_init_ls._replace(
            feedables=feedables,
            histories=histories,
            constants=constants)

        # Call the decoder body function with the expanded loop state to get
        # the log probabilities of the possible first tokens.

        decoder_body = self.parent_decoder.get_body(False)
        dec_next_ls = decoder_body(*dec_init_ls)

        # Construct the initial loop state of the beam search decoder. To allow
        # ensembling, the values are replaced with placeholders with a default
        # value. Despite this is necessary only for variables that grow in
        # time, the placeholder replacement is done on the whole structures, as
        # you can see below.

        search_state = SearchState(
            logprob_sum=tf.tile(
                tf.expand_dims([0.0] + [-INF] * (self.beam_size - 1), 0),
                [self.batch_size, 1],
                name="bs_logprob_sum"),
            prev_logprobs=tf.reshape(
                tf.nn.log_softmax(dec_next_ls.feedables.prev_logits),
                [self.batch_size, self.beam_size, len(self.vocabulary)]),
            lengths=tf.zeros(
                [self.batch_size, self.beam_size], dtype=tf.int32,
                name="bs_lengths"),
            finished=tf.zeros(
                [self.batch_size, self.beam_size], dtype=tf.bool))

        # We add the input_symbol to token_ids during search_results
        # initialization for simpler beam_body implementation

        search_results = SearchResults(
            scores=tf.zeros(
                shape=[self.batch_size, self.beam_size],
                dtype=tf.float32,
                name="beam_scores"),
            token_ids=tf.reshape(
                feedables.input_symbol,
                [1, self.batch_size, self.beam_size],
                name="beam_tokens"))

        # In structures that contain tensors that grow in time, we replace
        # tensors with placeholders with loosened shape constraints in the time
        # dimension.

        dec_next_ls = tf.contrib.framework.nest.map_structure(
            lambda x: tf.placeholder_with_default(
                x, get_state_shape_invariants(x)),
            dec_next_ls)

        search_results = tf.contrib.framework.nest.map_structure(
            lambda x: tf.placeholder_with_default(
                x, get_state_shape_invariants(x)),
            search_results)

        return BeamSearchLoopState(
            search_state=search_state,
            search_results=search_results,
            decoder_loop_state=dec_next_ls)

    def loop_continue_criterion(self, *args) -> tf.Tensor:
        """Decide whether to break out of the while loop.

        The criterion for stopping the loop is that either all hypotheses are
        finished or a maximum number of steps has been reached. Here the number
        of steps is the number of steps of the underlying decoder minus one,
        because this function is evaluated after the decoder step has been
        called and its step has been incremented. This is caused by the fact
        that we call the decoder body function at the end of the beam body
        function. (And that, in turn, is to support ensembling.)

        Arguments:
            args: A ``BeamSearchLoopState`` instance.

        Returns:
            A scalar boolean ``Tensor``.
        """
        loop_state = BeamSearchLoopState(*args)

        beam_step = loop_state.decoder_loop_state.feedables.step - 1
        finished = loop_state.search_state.finished

        max_step_cond = tf.less(beam_step, self.max_steps)
        unfinished_cond = tf.logical_not(tf.reduce_all(finished))

        return tf.logical_and(max_step_cond, unfinished_cond)

    def decoding_loop(self) -> BeamSearchOutput:
        """Create the decoding loop.

        This function mimics the behavior of the ``decoding_loop`` method of
        the ``AutoregressiveDecoder``, except the initial loop state is created
        outside this method because it is accessed and fed during ensembling.

        TODO: The ``finalize_loop`` method and the handling of attention loop
        states might be implemented in the future.

        Returns:
            This method returns a populated ``BeamSearchOutput`` object.
        """

        final_loop_state = tf.while_loop(
            self.loop_continue_criterion,
            self.get_body(),
            self.initial_loop_state,
            shape_invariants=tf.contrib.framework.nest.map_structure(
                get_state_shape_invariants, self.initial_loop_state))

        # TODO: return att_loop_states properly
        return BeamSearchOutput(
            last_search_step_output=final_loop_state.search_results,
            last_dec_loop_state=final_loop_state.decoder_loop_state,
            last_search_state=final_loop_state.search_state,
            attention_loop_states=[])

    def get_body(self) -> Callable[[Any], BeamSearchLoopState]:
        """Return a body function for ``tf.while_loop``.

        Returns:
            A function that performs a single decoding step.
        """
        decoder_body = self.parent_decoder.get_body(train_mode=False)

        # pylint: disable=too-many-locals
        def body(*args: Any) -> BeamSearchLoopState:
            """Execute a single beam search step.

            An implementation of the beam search algorithm, which works as
            follows:

            1. Create a valid ``logprobs`` tensor which contains distributions
               over the output tokens for each hypothesis in the beam. For
               finished hypotheses, the log probabilities of all tokens except
               the padding token are set to negative infinity.

            2. Expand the beam by appending every possible token to every
               existing hypothesis. Update the log probabilitiy sum of each
               hypothesis and its length (add one for unfinished hypotheses).
               For each hypothesis, compute the score using the length penalty
               term.

            3. Select the ``beam_size`` best hypotheses from the score pool.
               This is implemented by flattening the scores tensor and using
               the ``tf.nn.top_k`` function.

            4. Reconstruct the beam by gathering elements from the original
               data structures using the data indices computed in the previous
               step.

            5. Call the ``body`` function of the underlying decoder.

            6. Populate a new ``BeamSearchLoopState`` object with the selected
               values and with the newly obtained decoder loop state.

            Note that this function expects the decoder to be called at least
            once prior the first execution.

            Arguments:
                args: An instance of the ``BeamSearchLoopState`` structure.
                    (see the docs for this module)

            Returns:
                A ``BeamSearchLoopState`` after one step of the decoding.

            """
            loop_state = BeamSearchLoopState(*args)
            dec_loop_state = loop_state.decoder_loop_state
            search_state = loop_state.search_state
            search_results = loop_state.search_results

            # mask the probabilities
            # shape(logprobs) = [batch, beam, vocabulary]
            logprobs = search_state.prev_logprobs

            finished_mask = tf.expand_dims(
                tf.to_float(search_state.finished), 2)
            unfinished_logprobs = (1. - finished_mask) * logprobs

            finished_row = tf.one_hot(
                PAD_TOKEN_INDEX,
                len(self.vocabulary),
                dtype=tf.float32,
                on_value=0.,
                off_value=-INF)

            finished_logprobs = finished_mask * finished_row
            logprobs = unfinished_logprobs + finished_logprobs

            # update hypothesis scores
            # shape(hyp_probs) = [batch, beam, vocabulary]
            hyp_probs = tf.expand_dims(search_state.logprob_sum, 2) + logprobs

            # update hypothesis lengths
            hyp_lengths = search_state.lengths + 1 - tf.to_int32(
                search_state.finished)

            # shape(scores) = [batch, beam, vocabulary]
            scores = hyp_probs / tf.expand_dims(
                self._length_penalty(hyp_lengths), 2)

            # reshape to [batch, beam * vocabulary] for topk
            scores_flat = tf.reshape(
                scores, [-1, self.beam_size * len(self.vocabulary)])

            # shape(both) = [batch, beam]
            topk_scores, topk_indices = tf.nn.top_k(
                scores_flat, k=self.beam_size)

            topk_indices.set_shape([None, self.beam_size])
            topk_scores.set_shape([None, self.beam_size])

            next_word_ids = tf.to_int64(
                tf.mod(topk_indices, len(self.vocabulary)))
            next_beam_ids = tf.div(topk_indices, len(self.vocabulary))

            # batch offset for tf.gather_nd
            batch_offset = tf.tile(
                tf.expand_dims(tf.range(self.batch_size), 1),
                [1, self.beam_size])
            batch_beam_ids = tf.stack([batch_offset, next_beam_ids], axis=2)

            # gather the topk logprob_sums
            next_beam_lengths = tf.gather_nd(hyp_lengths, batch_beam_ids)
            next_beam_logprob_sum = tf.gather_nd(
                tf.reshape(
                    hyp_probs, [-1, self.beam_size * len(self.vocabulary)]),
                tf.stack([batch_offset, topk_indices], axis=2))

            # mark finished beams
            next_finished = tf.gather_nd(search_state.finished, batch_beam_ids)
            next_just_finished = tf.equal(next_word_ids, END_TOKEN_INDEX)
            next_finished = tf.logical_or(next_finished, next_just_finished)

            # we need to flatten the feedables for the parent_decoder
            next_feedables = tf.contrib.framework.nest.map_structure(
                lambda x: gather_flat(x, batch_beam_ids,
                                      self.batch_size, self.beam_size),
                dec_loop_state.feedables)

            next_feedables = next_feedables._replace(
                input_symbol=tf.reshape(next_word_ids, [-1]),
                finished=tf.reshape(next_finished, [-1]))

            # histories have shape [len, batch, ...]
            def gather_fn(x):
                return partial_transpose(
                    gather_flat(
                        partial_transpose(x, [1, 0]),
                        batch_beam_ids,
                        self.batch_size,
                        self.beam_size),
                    [1, 0])

            next_histories = tf.contrib.framework.nest.map_structure(
                gather_fn, dec_loop_state.histories)

            dec_loop_state = dec_loop_state._replace(
                feedables=next_feedables,
                histories=next_histories)

            # CALL THE DECODER BODY FUNCTION
            next_loop_state = decoder_body(*dec_loop_state)

            next_search_state = SearchState(
                logprob_sum=next_beam_logprob_sum,
                prev_logprobs=tf.reshape(
                    tf.nn.log_softmax(next_loop_state.feedables.prev_logits),
                    [self.batch_size, self.beam_size, len(self.vocabulary)]),
                lengths=next_beam_lengths,
                finished=next_finished)

            next_token_ids = tf.transpose(search_results.token_ids, [1, 2, 0])
            next_token_ids = tf.gather_nd(next_token_ids, batch_beam_ids)
            next_token_ids = tf.transpose(next_token_ids, [2, 0, 1])
            next_output = SearchResults(
                scores=topk_scores,
                token_ids=append_tensor(next_token_ids, next_word_ids))

            return BeamSearchLoopState(
                search_state=next_search_state,
                search_results=next_output,
                decoder_loop_state=next_loop_state)
        # pylint: enable=too-many-locals

        return body

    def _length_penalty(self, lengths: tf.Tensor) -> tf.Tensor:
        """Apply length penalty ("lp") term from Eq. 14.

        https://arxiv.org/pdf/1609.08144.pdf

        Arguments:
            lengths: A ``Tensor`` of lengths of the hypotheses in the beam.

        Returns:
            A float ``Tensor`` with the length penalties for each hypothesis
            in the beam.
        """
        return ((5. + tf.to_float(lengths)) / 6.) ** self.length_normalization

    def expand_to_beam(self, val: tf.Tensor, dim: int = 0) -> tf.Tensor:
        """Copy a tensor along a new beam dimension.

        Arguments:
            val: The ``Tensor`` to expand.
            dim: The dimension along which to expand. Usually, the batch axis.

        Returns:
            The expanded tensor.
        """
        orig_shape = get_shape_list(val)
        if val.shape.ndims == 0:
            return val

        orig_shape[dim] *= self.beam_size
        tile_shape = [1] * (len(orig_shape) + 1)
        tile_shape[dim + 1] = self.beam_size

        val = tf.tile(tf.expand_dims(val, 1), tile_shape)
        val = tf.reshape(val, orig_shape)

        return val
