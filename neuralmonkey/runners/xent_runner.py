from typing import Dict, List, Union

from typeguard import check_argument_types
import tensorflow as tf
import numpy as np

from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import BaseRunner


SupportedDecoders = Union[AutoregressiveDecoder]


class XentRunner(BaseRunner[SupportedDecoders]):

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["XentRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            xents = np.concatenate([res["xents"] for res in results], axis=0)
            self.set_runner_result(outputs=xents.tolist(),
                                   losses=[float(np.mean(xents))])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: SupportedDecoders) -> None:
        check_argument_types()
        BaseRunner[SupportedDecoders].__init__(
            self, output_series, decoder)

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"xents": self.decoder.train_xents}

    @property
    def loss_names(self) -> List[str]:
        return ["xent"]
