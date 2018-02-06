"""Beam search decoder with gaussian length estimation."""
from typing import NamedTuple, Callable, Optional, Tuple, Set

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.dataset import Dataset
from neuralmonkey.decoders.decoder import Decoder, LoopState
from neuralmonkey.decoders.gaussian_estimator import GaussianEstimator
from neuralmonkey.vocabulary import START_TOKEN_INDEX, END_TOKEN_INDEX

# pylint: disable=invalid-name
SearchState = NamedTuple(
    "SearchState",
    [("logprob_sum", tf.Tensor),
     ("entropy_sum", tf.Tensor),
     ("last_symbol", tf.Tensor)])

FinishedBeam = NamedTuple(
    "FinishedBeam",
    [("score", tf.Tensor),
     ("length", tf.Tensor),
     ("prefix_beam_id", tf.Tensor)])

SearchHistory = NamedTuple(
    "SearchHistory",
    [("prefix_beam_ids", tf.TensorArray),
     ("symbols", tf.TensorArray)])

BeamSearchLoopState = NamedTuple(
    "BeamSearchLoopState",
    [("unfinished_beam", SearchState),
     ("finished_beam", Optional[FinishedBeam]),
     ("history", SearchHistory),
     ("decoder_loop_state", LoopState)])
# pylint: enable=invalid-name


class BeamSearchDecoder(ModelPart):
    """In-graph beam search with Gaussian estimator for batch size 1."""

    def __init__(self,
                 name: str,
                 parent_decoder: Decoder,
                 beam_size: int,
                 length_estimator: GaussianEstimator = None,
                 perplexity_scoring: bool = True,
                 length_normalize: bool = False,
                 max_steps: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        self.parent_decoder = parent_decoder
        self.beam_size = beam_size
        self.length_estimator = length_estimator
        self.perplexity_scoring = perplexity_scoring
        self.length_normalize = length_normalize

        self.max_steps = max_steps
        if self.max_steps is None:
            self.max_steps = parent_decoder.max_output_len

        self.outputs = self.decoding_loop()

    @property
    def vocabulary(self):
        return self.parent_decoder.vocabulary

    def get_initial_loop_state(self) -> BeamSearchLoopState:
        search_state = SearchState(
            logprob_sum=tf.constant([0.0]),
            entropy_sum=tf.constant([0.0]),
            last_symbol=tf.constant([START_TOKEN_INDEX]))

        finished_beam = None
        history = SearchHistory(
            prefix_beam_ids=tf.TensorArray(dtype=tf.int32, dynamic_size=True,
                                           size=0, name="beam_parents"),
            symbols=tf.TensorArray(dtype=tf.int32, dynamic_size=True,
                                   size=0, name="beam_symbols"))

        dec_loop_state = self.parent_decoder.get_initial_loop_state()

        return BeamSearchLoopState(
            unfinished_beam=search_state,
            finished_beam=finished_beam,
            history=history,
            decoder_loop_state=dec_loop_state)

    def decoding_loop(self) -> Tuple[tf.Tensor, tf.Tensor, FinishedBeam]:
        # collect attention objects
        beam_body = self.get_body()

        # first step has to be run manually because while_loop needs the same
        # shapes between steps and the first beam state is not beam-sized, but
        # just a single state.
        initial_loop_state = self.get_initial_loop_state()
        next_bs_loop_state = beam_body(*initial_loop_state)

        def cond(*args) -> tf.Tensor:
            # TODO: udelat chytre
            bsls = BeamSearchLoopState(*args)
            return tf.less(
                bsls.decoder_loop_state.feedables.step, self.max_steps)

        final_state = tf.while_loop(cond, beam_body, next_bs_loop_state)

        symbols = final_state.history.symbols.stack()
        prefix_beam_ids = final_state.history.prefix_beam_ids.stack()

        return symbols, prefix_beam_ids, final_state.finished_beam

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
            loop_state = BeamSearchLoopState(*args)
            unfinished_beam = loop_state.unfinished_beam
            prev_decoder_ls = loop_state.decoder_loop_state

            # don't want to use this decoder with uninitialized parent
            assert self.parent_decoder.step_scope.reuse

            # CALL THE DECODER BODY FUNCTION
            # TODO figure out why mypy throws too-many-arguments on this
            decoder_ls = decoder_body(*prev_decoder_ls)  # type: ignore

            # For unfinished hypotheses we don't want to generate end symbols.
            not_end_token_penalty = tf.one_hot(
                END_TOKEN_INDEX,
                len(self.parent_decoder.vocabulary),
                dtype=tf.float32,
                off_value=0.,
                on_value=tf.float32.min)

            # shape(logprobs) = beam x vocabulary
            logprobs = tf.nn.log_softmax(decoder_ls.feedables.prev_logits)
            entropy = -tf.reduce_sum(logprobs * tf.exp(logprobs), axis=1)

            # update hypothesis scores
            # shape(hyp_probs) = beam x vocabulary
            hyp_probs = (tf.expand_dims(unfinished_beam.logprob_sum, 1) +
                         logprobs + not_end_token_penalty)

            # flatten so we can use top_k
            hyp_probs_flat = tf.reshape(hyp_probs, [-1])

            # shape(both) = beam
            next_logprob_sum, topk_indices = tf.nn.top_k(
                hyp_probs_flat, self.beam_size)

            next_logprob_sum.set_shape([self.beam_size])
            topk_indices.set_shape([self.beam_size])

            next_symbol = tf.mod(topk_indices,
                                 len(self.parent_decoder.vocabulary))

            next_beam_id = tf.div(topk_indices,
                                  len(self.parent_decoder.vocabulary))
            logprob_flat = tf.exp(tf.reshape(logprobs, [-1]))

            entropy_sum = unfinished_beam.entropy_sum + entropy
            next_entropy_sum = tf.gather(entropy_sum, next_beam_id)

            # TODO: vyclenit gatherovani jako funkci
            next_feedables_dict = {
                "input_symbol": next_symbol,
                "finished": tf.zeros(dtype=tf.bool, shape=[self.beam_size])}
            for key, val in decoder_ls.feedables._asdict().items():
                if key in ["step", "input_symbol", "finished"]:
                    continue

                if isinstance(val, tf.Tensor):
                    next_feedables_dict[key] = tf.gather(val, next_beam_id)
                elif isinstance(val, list):
                    if not all(isinstance(t, tf.Tensor) for t in val):
                        raise TypeError("Expected tf.Tensor among feedables")

                    next_feedables_dict[key] = [tf.gather(t, next_beam_id)
                                                for t in val]
                else:
                    raise TypeError("Expected only tensors or list of tensors "
                                    "among feedables")

            next_feedables = decoder_ls.feedables._replace(
                **next_feedables_dict)

            next_decoder_ls = decoder_ls._replace(feedables=next_feedables)

            history = loop_state.history

            step = prev_decoder_ls.feedables.step
            updated_history = SearchHistory(
                prefix_beam_ids=history.prefix_beam_ids.write(
                    step, next_beam_id),
                symbols=history.symbols.write(step, next_symbol))

            search_state = SearchState(
                logprob_sum=next_logprob_sum,
                entropy_sum=next_entropy_sum,
                last_symbol=next_symbol)

            if loop_state.finished_beam is None:
                # If there is no previous finished beam, we construct it out
                # of empty hypotheses.
                next_finished_beam = FinishedBeam(
                    score=tf.fill([self.beam_size], tf.float32.min),
                    length=tf.ones(self.beam_size, dtype=tf.int32),
                    prefix_beam_id=tf.zeros(self.beam_size, dtype=tf.int32))
            else:
                # Compute score of the newly finished hypotheses.
                end_logprobs = logprobs[:, END_TOKEN_INDEX]
                new_finished_score = unfinished_beam.logprob_sum + end_logprobs#) /

                if self.perplexity_scoring:
                    new_finished_score += (
                        unfinished_beam.entropy_sum + entropy)

                if self.length_normalize:
                    new_finished_score /= tf.to_float(step)

                if self.length_estimator is not None:
                    length_prob = self.length_estimator.probability_around(
                            tf.to_float(step - 2))
                    new_finished_score += tf.log(length_prob)


                # We concatenate the best hypotheses so far with newly finished
                # hypotheses and select the `beam_size` from them.
                all_finished_score = tf.concat(
                    [loop_state.finished_beam.score, new_finished_score],
                    axis=0)
                all_length = tf.concat(
                    [loop_state.finished_beam.length,
                     tf.fill([self.beam_size], step)],
                    axis=0)
                all_prefix_beam_id = tf.concat(
                    [loop_state.finished_beam.prefix_beam_id,
                     tf.range(self.beam_size)],
                    axis=0)

                next_score, fin_indices = tf.nn.top_k(
                    all_finished_score, self.beam_size)
                next_length = tf.gather(all_length, fin_indices)
                next_prefix_beam_id = tf.gather(
                    all_prefix_beam_id, fin_indices)

                next_finished_beam = FinishedBeam(
                    score=next_score,
                    length=next_length,
                    prefix_beam_id=next_prefix_beam_id)

            return BeamSearchLoopState(
                unfinished_beam=search_state,
                finished_beam=next_finished_beam,
                history=updated_history,
                decoder_loop_state=next_decoder_ls)
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

    def get_dependencies(self) -> Set[ModelPart]:
        """Collect recusively all encoders and decoders."""
        to_return = ModelPart.get_dependencies(self)

        return to_return.union(self.length_estimator.get_dependencies())
