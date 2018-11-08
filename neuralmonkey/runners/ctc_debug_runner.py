from typing import Dict, List, Set, Optional

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.decoders.ctc_decoder import CTCDecoder


class CTCDebugExecutable(Executable):

    def __init__(self,
                 all_coders: Set[GenericModelPart],
                 fetches: FeedDict,
                 vocabulary: Vocabulary) -> None:
        self._all_coders = all_coders
        self._fetches = fetches
        self._vocabulary = vocabulary

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._all_coders, self._fetches, []

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) != 1:
            raise RuntimeError("CTCDebug runner does not support ensembling.")

        logits = results[0]["logits"]
        argmaxes = np.argmax(logits, axis=2).T

        decoded_batch = []
        for indices in argmaxes:
            decoded_instance = []
            for index in indices:
                if index == len(self._vocabulary):
                    symbol = "<BLANK>"
                else:
                    symbol = self._vocabulary.index_to_word[index]
                decoded_instance.append(symbol)
            decoded_batch.append(decoded_instance)

        self.result = ExecutionResult(
            outputs=decoded_batch,
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class CTCDebugRunner(BaseRunner[CTCDecoder]):
    """A runner that print out raw CTC output including the blank symbols."""

    def __init__(self,
                 output_series: str,
                 decoder: CTCDecoder) -> None:
        check_argument_types()
        BaseRunner[CTCDecoder].__init__(self, output_series, decoder)

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> CTCDebugExecutable:
        fetches = {"logits": self._decoder.logits}

        return CTCDebugExecutable(
            self.all_coders,
            fetches,
            self._decoder.vocabulary)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return []
