from typing import Dict, List

from typeguard import check_argument_types
import tensorflow as tf
import numpy as np

from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import BaseRunner


class PerplexityRunner(BaseRunner[AutoregressiveDecoder]):

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["PerplexityRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            perplexities = np.mean(
                [2 ** res["xents"] for res in results], axis=0)
            xent = float(np.mean([res["xents"] for res in results]))
            self.set_runner_result(outputs=perplexities.tolist(),
                                   losses=[xent])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: AutoregressiveDecoder) -> None:
        check_argument_types()
        BaseRunner[AutoregressiveDecoder].__init__(
            self, output_series, decoder)

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"xents": self.decoder.train_xents}

    @property
    def loss_names(self) -> List[str]:
        return ["xents"]
