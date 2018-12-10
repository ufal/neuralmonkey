from typing import Dict, List

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.runners.base_runner import BaseRunner
from neuralmonkey.decoders.ctc_decoder import CTCDecoder
from neuralmonkey.decorators import tensor


class CTCDebugRunner(BaseRunner[CTCDecoder]):
    """A runner that print out raw CTC output including the blank symbols."""

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["CTCDebugRunner"]):

        def collect_results(self, results: List[Dict]) -> None:

            vocabulary = self.executor.decoder.vocabulary
            if len(results) != 1:
                raise RuntimeError("CTCDebugRunners do not support ensembles.")

            logits = results[0]["logits"]
            argmaxes = np.argmax(logits, axis=2).T

            decoded_batch = []
            for indices in argmaxes:
                decoded_instance = []
                for index in indices:
                    if index == len(vocabulary):
                        symbol = "<BLANK>"
                    else:
                        symbol = vocabulary.index_to_word[index]
                    decoded_instance.append(symbol)
                decoded_batch.append(decoded_instance)

            self.set_runner_result(outputs=decoded_batch, losses=[])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: CTCDecoder) -> None:
        check_argument_types()
        BaseRunner[CTCDecoder].__init__(self, output_series, decoder)

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"logits": self.decoder.logits}

    @property
    def loss_names(self) -> List[str]:
        return []
