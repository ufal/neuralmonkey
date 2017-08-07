from typing import Callable, List, Dict, Optional

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.decoder import LoopState
from neuralmonkey.decoders.beam_search_decoder import (BeamSearchDecoder,
                                                       SearchStepOutput)
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN

#BeamSearchLoopState = NamedTuple("BeamSearchLoopState",
#                                 [("bs_state", SearchState),
#                                  ("bs_output", SearchStepOutputTA),
#                                  ("decoder_loop_state", LoopState)])

class EnsembleExecutable(Executable):
    def __init__(self,
                 rank: int,
                 all_encoders: List[ModelPart],
                 bs_outputs: SearchStepOutput,
                 dec_loop_state: LoopState,
                 vocabulary: Vocabulary,
                 max_steps: int,
                 beam_size: int,
                 postprocess: Optional[Callable]) -> None:

        self._rank = rank
        self._all_encoders = all_encoders
        self._bs_outputs = bs_outputs
        self._dec_loop_state = dec_loop_state
        self._vocabulary = vocabulary
        self._max_steps = max_steps
        self._postprocess = postprocess
        self._beam_size = beam_size
        self._next_feed = {}

        self._outputs = []
        self._step = 0
        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        return self._all_encoders, {'bs_outputs': self._bs_outputs}, self._next_feed #'dec_loop_state': self._dec_loop_state}, self._next_feed

    def collect_results(self, results: List[Dict]) -> None:
        if self._rank > 1:
            raise ValueError("Ensemble beam search runner does not support n-best lists.")

        # beam_size x vocabulary_size
        # Contains info about averaged score and pointer
        # to corresponding session-beams
        ens_scores = {}
        # Check whether all decoders are finished
        #finished = 1
        for sess_idx in range(len(results)):
            bs_outputs = results[sess_idx]['bs_outputs']

            for hyp_idx in range(len(bs_outputs.scores[self._step])):
                token_id = bs_outputs.token_ids[self._step][hyp_idx]
                if not str(token_id) in ens_scores.keys():
                    ens_scores[str(token_id)] = 0
                ens_scores[str(token_id)] += bs_outputs.scores[self._step][hyp_idx] / len(results)
                #finished *= np.prod(results[sess_idx]['bs_state'].finished)

        # Collect ensembled beam scores and indices
        top_token_id, top_token_score = sorted(ens_scores.items(), key=lambda x: -x[1])[0]
        top_token_id = int(top_token_id)
        top_token = self._vocabulary.index_to_word[top_token_id]
        self._outputs.append(top_token)

        #if finished or top_token == END_TOKEN:
        if top_token == END_TOKEN:
            self.prepare_result(results)
            return

        # Otherwise prepare next feed
        dec_loop_state = results[0]['dec_loop_state']
        self._next_feed = {}
        self._next_feed.update({
            dec_loop_state.input_symbol: np.full([self._beam_size], best_token_id)})
        self._step += 1


    def prepare_result(self, best_score):
        if self._outputs[-1] == END_TOKEN:
            self._outputs = self._outputs[:-1]

        if self._postprocess is not None:
            self._outputs = self._postprocess([self._outputs])
        else:
            self._outputs = [self._outputs]

        self.result = ExecutionResult(
            outputs=self._outputs,
            losses=[best_score],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class EnsembleRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: BeamSearchDecoder,
                 rank: int = 1,
                 postprocess: Callable[[List[str]], List[str]] = None) -> None:
        super(EnsembleRunner, self).__init__(output_series, decoder)
        check_argument_types()

        if rank < 1 or rank > decoder.beam_size:
            raise ValueError(
                ("Rank of output hypothesis must be between 1 and the beam "
                 "size ({}), was {}.").format(decoder.beam_size, rank))

        self._rank = rank
        self._postprocess = postprocess

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> EnsembleExecutable:
        return EnsembleExecutable(
            self._rank, self.all_coders, self._decoder.outputs,
            self._decoder.dec_loop_state,
            self._decoder.vocabulary, self._decoder._max_steps,
            self._decoder._beam_size, self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["beam_search_score"]

    @property
    def decoder_data_id(self) -> Optional[str]:
        return None
