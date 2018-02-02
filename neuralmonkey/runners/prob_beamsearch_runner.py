from typing import Callable, List, Dict, Optional, Set

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.prob_beam_search_decoder import (
    BeamSearchDecoder, FinishedBeam)
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
from neuralmonkey.vocabulary import Vocabulary

# pylint: disable=invalid-name
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class BeamSearchExecutable(Executable):
    def __init__(self,
                 rank: int,
                 all_encoders: Set[ModelPart],
                 symbols: tf.Tensor,
                 prefix_beam_ids: tf.Tensor,
                 finished_beam: FinishedBeam,
                 vocabulary: Vocabulary,
                 postprocess: Optional[Callable]) -> None:

        self._rank = rank
        self._all_encoders = all_encoders
        self._symbols = symbols
        self._prefix_beam_ids = prefix_beam_ids
        self._finished_beam = finished_beam
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        fetches = {
            "symbols": self._symbols,
            "prefix_beam_ids": self._prefix_beam_ids,
            "finished_beam": self._finished_beam}
        return self._all_encoders, fetches, None

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) > 1:
            raise ValueError("Beam search runner does not support ensembling.")

        evaluated_beam = results[0]["finished_beam"]
        symbols = results[0]["symbols"]
        prefix_beam_ids = results[0]["prefix_beam_ids"]
        max_time = evaluated_beam.length[self._rank - 1]

        # pick the end of the hypothesis based on its rank
        bs_score = evaluated_beam.score[self._rank - 1]

        # now backtrack
        hyp_index = evaluated_beam.prefix_beam_id[self._rank - 1]
        output_tokens = []  # type: List[str]
        for time in reversed(range(max_time)):
            symbol = symbols[time][hyp_index]
            token = self._vocabulary.index_to_word[symbol]
            output_tokens.insert(0, token)
            hyp_index = prefix_beam_ids[time][hyp_index]

        if self._postprocess is not None:
            output_tokens = self._postprocess([output_tokens])

        self.result = ExecutionResult(
            outputs=output_tokens,
            losses=[bs_score],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class BeamSearchRunner(BaseRunner[BeamSearchDecoder]):

    def __init__(self,
                 output_series: str,
                 decoder: BeamSearchDecoder,
                 rank: int = 1,
                 postprocess: Postprocessor = None) -> None:
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
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> BeamSearchExecutable:
        return BeamSearchExecutable(
            rank=self._rank,
            all_encoders=self.all_coders,
            symbols=self._decoder.outputs[0],
            prefix_beam_ids=self._decoder.outputs[1],
            finished_beam=self._decoder.outputs[2],
            vocabulary=self._decoder.vocabulary,
            postprocess=self._postprocess)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["beam_search_score"]

    @property
    def decoder_data_id(self) -> Optional[str]:
        return None


def beam_search_runner_range(output_series: str,
                             decoder: BeamSearchDecoder,
                             max_rank: int = None,
                             postprocess: Postprocessor = None
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
