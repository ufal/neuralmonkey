from typing import Any, List, Callable, Dict
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.vocabulary import END_TOKEN_INDEX
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


class LabelRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: Any,
                 postprocess: Callable[[List[str]], List[str]] = None
                ) -> None:
        super(LabelRunner, self).__init__(output_series, decoder)
        self._postprocess = postprocess

        # make sure the lazy decoder creates its output tensor
        log("Decoder output tensor: {}".format(decoder.decoded))

    def get_executable(self,
                       compute_losses=False,
                       summaries=True,
                       num_sessions=1):
        if compute_losses:
            fetches = {"loss": self._decoder.cost}
        else:
            fetches = {}

        fetches["label_logprobs"] = self._decoder.logprobs
        fetches["input_mask"] = self._decoder.encoder.input_sequence.mask

        return LabelRunExecutable(self.all_coders, fetches,
                                  num_sessions,
                                  self._decoder.vocabulary,
                                  self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["loss"]


class LabelRunExecutable(Executable):

    def __init__(self,
                 all_coders,
                 fetches,
                 num_sessions,
                 vocabulary,
                 postprocess):
        self.all_coders = all_coders
        self._fetches = fetches
        self._num_sessions = num_sessions
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.decoded_labels = []
        self.result = None  # type: Option[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return (self.all_coders,
                self._fetches,
                [{} for _ in range(self._num_sessions)])

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

        # must transpose argmaxes because vectors_to_sentences is time-major
        decoded_labels = self._vocabulary.vectors_to_sentences(argmaxes.T)

        if self._postprocess is not None:
            decoded_labels = self._postprocess(decoded_labels)

        self.result = ExecutionResult(
            outputs=decoded_labels,
            losses=[loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
