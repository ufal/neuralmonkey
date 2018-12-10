from typing import Dict, List, Union, Callable

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.ctc_decoder import CTCDecoder
from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.decoders.sequence_labeler import SequenceLabeler
from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import BaseRunner

# pylint: disable=invalid-name
SupportedDecoder = Union[
    AutoregressiveDecoder, CTCDecoder, Classifier, SequenceLabeler]
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class PlainRunner(BaseRunner[SupportedDecoder]):
    """A runner which takes the output from decoder.decoded."""

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["PlainRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            if len(results) != 1:
                raise ValueError("PlainRunner needs exactly 1 execution "
                                 "result, got {}".format(len(results)))

            vocabulary = self.executor.decoder.vocabulary

            train_loss = results[0]["train_loss"]
            runtime_loss = results[0]["runtime_loss"]
            decoded = results[0]["decoded"]

            decoded_tokens = vocabulary.vectors_to_sentences(decoded)

            if self.executor.postprocess is not None:
                decoded_tokens = self.executor.postprocess(decoded_tokens)

            self.set_runner_result(outputs=decoded_tokens,
                                   losses=[train_loss, runtime_loss])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: SupportedDecoder,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[SupportedDecoder].__init__(self, output_series, decoder)
        self.postprocess = postprocess

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"decoded": self.decoder.decoded,
                "train_loss": self.decoder.train_loss,
                "runtime_loss": self.decoder.runtime_loss}

    @property
    def loss_names(self) -> List[str]:
        return ["train_loss", "runtime_loss"]
