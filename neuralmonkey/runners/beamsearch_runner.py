from typing import Callable, List, Dict, Optional, Set, cast

import scipy
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.beam_search_decoder import BeamSearchDecoder
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)
# pylint: disable=unused-import
from neuralmonkey.runners.base_runner import FeedDict
# pylint: enable=unused-import
from neuralmonkey.vocabulary import PAD_TOKEN_INDEX, END_TOKEN, PAD_TOKEN


class BeamSearchExecutable(Executable):
    def __init__(self,
                 rank: int,
                 all_coders: Set[ModelPart],
                 num_sessions: int,
                 decoder: BeamSearchDecoder,
                 postprocess: Optional[Callable]) -> None:
        """TODO: docstring describing the whole knowhow."""

        self._rank = rank
        self._num_sessions = num_sessions
        self._all_coders = all_coders
        self._decoder = decoder
        self._postprocess = postprocess

        # Length of the currently sequence decoded so far
        self._step = 0

        self._next_feed = [{} for _ in range(self._num_sessions)] \
            # type: List[FeedDict]

        # During ensembling, we execute only on decoder step per session.run
        # In the first step we do not generate any symbols only logprobs to be
        # ensembled together with initialization of the decoder itself,
        # therefore decoder.max_steps is set to 0
        if self._num_sessions > 1:
            for fd in self._next_feed:
                fd.update({self._decoder.max_steps: 0})

        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        return (self._all_coders,
                {"bs_outputs": self._decoder.outputs},
                self._next_feed)

    # pylint: disable=too-many-locals
    def collect_results(self, results: List[Dict]) -> None:
        # Recompute logits
        # Only necessary when ensembling models
        prev_logprobs = [res["bs_outputs"].last_search_state.prev_logprobs
                         for res in results]

        # Arithmetic mean
        ens_logprobs = (scipy.misc.logsumexp(prev_logprobs, 0)
                        - np.log(self._num_sessions))

        # Now we update the scores, parent_ids, token_ids based on the last
        # session.run of the beamsearch_decoder
        bs_outputs = results[0]["bs_outputs"]

        # step_size varies between single model run and ensembling
        # single model: step_size == max_sequence_len
        # ensembles: step_size == 1
        step_size = bs_outputs.last_dec_loop_state.step - 1

        batch_size = bs_outputs.last_search_step_output.scores.shape[1]
        # pylint: disable=attribute-defined-outside-init
        if self._step == 0:
            self._scores = np.empty(
                [0, batch_size, self._decoder.beam_size],
                dtype=float)
            self._parent_ids = np.empty(
                [0, batch_size, self._decoder.beam_size],
                dtype=int)
            self._token_ids = np.empty(
                [0, batch_size, self._decoder.beam_size],
                dtype=int)
        self._step += step_size
        self._scores = np.append(
            self._scores,
            bs_outputs.last_search_step_output.scores[0:step_size],
            axis=0)

        self._parent_ids = np.append(
            self._parent_ids,
            bs_outputs.last_search_step_output.parent_ids[0:step_size],
            axis=0)

        self._token_ids = np.append(
            self._token_ids,
            bs_outputs.last_search_step_output.token_ids[0:step_size],
            axis=0)
        # pylint: enable=attribute-defined-outside-init

        if (self._decoder.max_output_len is not None and
                self._step >= self._decoder.max_output_len):
            self.prepare_results()
            return

        # Prepare the next feed_dict (required for ensembles)
        self._next_feed = []
        for result in results:
            bs_outputs = result["bs_outputs"]

            input_beam_size = len(bs_outputs.last_search_state.prev_logprobs)
            input_beam_size //= batch_size
            search_state = bs_outputs.last_search_state._replace(
                input_beam_size=input_beam_size,
                prev_logprobs=ens_logprobs)

            # in the next iteration, we want to generate one new symbol
            # based on the ensembled logprobs (and then use this symbol
            # to get new set of logprobs for ensembling)
            fd = {self._decoder.max_steps: 1,
                  self._decoder.search_state: search_state}

            dec_feedables = bs_outputs.last_dec_loop_state

            # NOTE Due to the arrays in DecoderState (prev_contexts),
            # we have to create feed for each value separately.
            for field in self._decoder.decoder_state._fields:
                tensor = getattr(self._decoder.decoder_state, field)
                if field == "step":
                    value = 1
                else:
                    value = getattr(dec_feedables, field)
                if isinstance(tensor, list) and isinstance(value, list):
                    for t, val in zip(tensor, value):
                        fd.update({t: val})
                else:
                    fd.update({tensor: value})

            self._next_feed.append(fd)

        if self._step == 0:
            return

        # We assume that we can stop decoding when all tokens
        # in the last step were <pad>
        # TODO: investigate this and fix this if necessary
        if np.all(np.equal(self._token_ids[-1], PAD_TOKEN_INDEX)):
            self.prepare_results()
    # pylint: enable=too-many-locals

    def prepare_results(self):
        max_time = self._step

        decoded_tokens = []
        bs_scores = []
        # We extract last hyp_idx for each sentence in the batch
        hyp_indices = np.argpartition(
            -self._scores[-1], self._rank - 1)[:, self._rank - 1]

        for batch_idx, hyp_idx in enumerate(hyp_indices):
            output_tokens = []
            bs_scores.append(self._scores[-1][batch_idx][hyp_idx])
            for time in reversed(range(max_time)):
                token_id = self._token_ids[time][batch_idx][hyp_idx]
                token = self._decoder.vocabulary.index_to_word[token_id]
                output_tokens.append(token)
                hyp_idx = self._parent_ids[time][batch_idx][hyp_idx]

            output_tokens.reverse()

            before_eos_tokens = []
            for tok in output_tokens:
                if tok == END_TOKEN:
                    break
                # TODO: investigate why the decoder can start generating
                # padding before generating the END_TOKEN
                if tok != PAD_TOKEN:
                    before_eos_tokens.append(tok)

            decoded_tokens.append(before_eos_tokens)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess(decoded_tokens)

        # TODO: provide better summaries in case (issue #599)
        # we want to use the runner during training.
        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[np.mean(bs_scores) * len(bs_scores)],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class BeamSearchRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: BeamSearchDecoder,
                 rank: int = 1,
                 postprocess: Callable[[List[str]], List[str]] = None) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

        if rank < 1 or rank > decoder.beam_size:
            raise ValueError(
                ("Rank of output hypothesis must be between 1 and the beam "
                 "size ({}), was {}.").format(decoder.beam_size, rank))

        self._rank = rank
        self._postprocess = postprocess

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1) -> BeamSearchExecutable:
        decoder = cast(BeamSearchDecoder, self._decoder)

        return BeamSearchExecutable(
            self._rank, self.all_coders, num_sessions, decoder,
            self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["beam_search_score"]

    @property
    def decoder_data_id(self) -> Optional[str]:
        return None


def beam_search_runner_range(output_series: str,
                             decoder: BeamSearchDecoder,
                             max_rank: int = None,
                             postprocess: Callable[
                                 [List[str]], List[str]]=None
                            ) -> List[BeamSearchRunner]:
    """Return beam search runners for a range of ranks from 1 to max_rank.

    This means there is max_rank output series where the n-th series contains
    the n-th best hypothesis from the beam search.

    Args:
        output_series: Prefix of output series.
        decoder: Beam search decoder shared by all runners.
        max_rank: Maximum rank of the hypotheses.
        postprocess: Series-level postprocess applied on output.

    Returns:
        List of beam search runners getting hypotheses with rank from 1 to
        max_rank.
    """
    check_argument_types()

    if max_rank is None:
        max_rank = decoder.beam_size

    if max_rank > decoder.beam_size:
        raise ValueError(
            ("The maximum rank ({}) cannot be "
             "bigger than beam size {}.").format(
                 max_rank, decoder.beam_size))

    return [BeamSearchRunner("{}.rank{:03d}".format(output_series, r),
                             decoder, r, postprocess)
            for r in range(1, max_rank + 1)]
