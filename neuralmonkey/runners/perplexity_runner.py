from typing import Dict, List
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

from typeguard import check_argument_types
import tensorflow as tf
import numpy as np

from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)


class PerplexityExecutable(Executable):
    def __init__(self, xent_op: tf.Tensor) -> None:
        self._xent_op = xent_op

        self._result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        return {"xents": self._xent_op}, []

    def collect_results(self, results: List[Dict]) -> None:
        perplexities = np.mean([2 ** res["xents"] for res in results], axis=0)
        xent = float(np.mean([res["xents"] for res in results]))
        self._result = ExecutionResult(
            outputs=perplexities.tolist(),
            losses=[xent],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class PerplexityRunner(BaseRunner[AutoregressiveDecoder]):
    def __init__(self,
                 output_series: str,
                 decoder: AutoregressiveDecoder) -> None:
        check_argument_types()
        BaseRunner[AutoregressiveDecoder].__init__(
            self, output_series, decoder)

        self._decoder_xent = self._decoder.train_xents

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> PerplexityExecutable:
        return PerplexityExecutable(self._decoder_xent)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["xent"]
