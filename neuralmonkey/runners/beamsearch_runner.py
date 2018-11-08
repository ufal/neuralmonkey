from typing import Callable, List, Dict, Optional, Set

import scipy
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.decoders.beam_search_decoder import BeamSearchDecoder
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)
# pylint: disable=unused-import
from neuralmonkey.runners.base_runner import FeedDict
# pylint: enable=unused-import
from neuralmonkey.vocabulary import END_TOKEN_INDEX


class BeamSearchExecutable(Executable):
    def __init__(self,
                 rank: int,
                 all_coders: Set[GenericModelPart],
                 num_sessions: int,
                 decoder: BeamSearchDecoder,
                 postprocess: Optional[Callable]) -> None:
        self._rank = rank
        self._num_sessions = num_sessions
        self._all_coders = all_coders
        self._decoder = decoder
        self._postprocess = postprocess

        self._next_feed = [{} for _ in range(self._num_sessions)] \
            # type: List[FeedDict]

        # During ensembling, we set the decoder max_steps to zero because the
        # loop is run manually in the runner.
        if self._num_sessions > 1:
            for fd in self._next_feed:
                fd.update({self._decoder.max_steps: 0})

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        return (self._all_coders, {"bs_outputs": self._decoder.outputs},
                self._next_feed)

    def collect_results(self, results: List[Dict]) -> None:
        # Recompute logits
        # Only necessary when ensembling models
        prev_logprobs = [res["bs_outputs"].last_search_state.prev_logprobs
                         for res in results]

        # Arithmetic mean
        ens_logprobs = (scipy.misc.logsumexp(prev_logprobs, 0)
                        - np.log(self._num_sessions))

        if self._is_finished(results):
            self.prepare_results(
                results[0]["bs_outputs"].last_search_step_output)
            return

        # Prepare the next feed_dict (required for ensembles)
        self._next_feed = []
        for result in results:
            bs_out = result["bs_outputs"]

            search_state = bs_out.last_search_state._replace(
                prev_logprobs=ens_logprobs)

            dec_ls = bs_out.last_dec_loop_state
            feedables = dec_ls.feedables._replace(step=1)
            dec_ls = dec_ls._replace(feedables=feedables)

            fd = {
                self._decoder.max_steps: 1,
                self._decoder.search_state: search_state,
                self._decoder.search_results: bs_out.last_search_step_output,
                self._decoder.decoder_state: dec_ls}

            self._next_feed.append(fd)

        return

    def prepare_results(self, output):
        bs_scores = [s[self._rank - 1] for s in output.scores]

        tok_ids = np.transpose(output.token_ids, [1, 2, 0])
        decoded_tokens = [toks[self._rank - 1][1:] for toks in tok_ids]

        for i, sent in enumerate(decoded_tokens):
            decoded = []
            for tok_id in sent:
                if tok_id == END_TOKEN_INDEX:
                    break
                decoded.append(self._decoder.vocabulary.index_to_word[tok_id])
            decoded_tokens[i] = decoded

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

    def _is_finished(self, results):
        finished = [
            all(res["bs_outputs"].last_dec_loop_state.feedables.finished)
            for res in results]
        if all(finished):
            return True
        bs_out = results[0]["bs_outputs"]
        step = len(bs_out.last_search_step_output.token_ids) - 1
        if step >= self._decoder.max_steps_int:
            return True
        return False


class BeamSearchRunner(BaseRunner[BeamSearchDecoder]):
    """A runner which takes the output from a beam search decoder.

    The runner and the beam search decoder support ensembling.
    """

    def __init__(self,
                 output_series: str,
                 decoder: BeamSearchDecoder,
                 rank: int = 1,
                 postprocess: Callable[[List[str]], List[str]] = None) -> None:
        """Initialize the beam search runner.

        Arguments:
            output_series: Name of the series produced by the runner.
            decoder: The beam search decoder to use.
            rank: The hypothesis from the beam to select. Setting rank to 1
                selects the best hypothesis.
            postprocess: The postprocessor to apply to the output data.
        """
        check_argument_types()
        BaseRunner[BeamSearchDecoder].__init__(self, output_series, decoder)

        if rank < 1 or rank > decoder.beam_size:
            raise ValueError(
                ("Rank of output hypothesis must be between 1 and the beam "
                 "size ({}), was {}.").format(decoder.beam_size, rank))

        self._rank = rank
        self._postprocess = postprocess

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1) -> BeamSearchExecutable:
        return BeamSearchExecutable(self._rank, self.all_coders, num_sessions,
                                    self._decoder, self._postprocess)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["beam_search_score"]


# TODO: allow the beam search runner to accept multiple ranks because using
# beam_search_runner_range is too slow.
def beam_search_runner_range(
        output_series: str,
        decoder: BeamSearchDecoder,
        max_rank: int = None,
        postprocess: Callable[[List[str]], List[str]] = None) -> List[
            BeamSearchRunner]:
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
