from typing import List, Dict, Callable

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.vocabulary import END_TOKEN_INDEX
from neuralmonkey.runners.base_runner import BaseRunner
from neuralmonkey.decoders.sequence_labeler import SequenceLabeler

# pylint: disable=invalid-name
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class LabelRunner(BaseRunner[SequenceLabeler]):

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["LabelRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            loss = results[0].get("loss", 0.)
            summed_logprobs = results[0]["label_logprobs"]
            input_mask = results[0]["input_mask"]

            for sess_result in results[1:]:
                loss += sess_result.get("loss", 0.)
                summed_logprobs = np.logaddexp(summed_logprobs,
                                               sess_result["label_logprobs"])
                assert input_mask == sess_result["input_mask"]

            argmaxes = np.argmax(summed_logprobs, axis=2)

            # CAUTION! FABULOUS HACK BELIEVE ME
            argmaxes -= END_TOKEN_INDEX
            argmaxes *= input_mask.astype(int)
            argmaxes += END_TOKEN_INDEX

            # transpose argmaxes because vectors_to_sentences is time-major
            vocabulary = self.executor.decoder.vocabulary
            decoded_labels = vocabulary.vectors_to_sentences(argmaxes.T)

            if self.executor.postprocess is not None:
                decoded_labels = self.executor.postprocess(decoded_labels)

            self.set_runner_result(outputs=decoded_labels, losses=[loss])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: SequenceLabeler,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[SequenceLabeler].__init__(self, output_series, decoder)
        self.postprocess = postprocess

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {
            "label_logprobs": self.decoder.logprobs,
            "input_mask": self.decoder.encoder.input_sequence.temporal_mask,
            "loss": self.decoder.cost}

    @property
    def loss_names(self) -> List[str]:
        return ["loss"]
