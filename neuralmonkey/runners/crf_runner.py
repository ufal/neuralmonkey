from typing import List, Dict, Set, Optional, Callable
from typeguard import check_argument_types

from neuralmonkey.logging import log
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN_INDEX
from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, FeedDict, ExecutionResult, NextExecute)
from neuralmonkey.decoders.sequence_labeler import CRFLabeler

# pylint: disable=invalid-name
Postprocessor = Callable[[List[List[str]]], List[List[str]]]
# pylint: enable=invalid-name


class CRFRunExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: FeedDict,
                 vocabulary: Vocabulary,
                 postprocess: Optional[Postprocessor]) -> None:
        self._all_coders = all_coders
        self._fetches = fetches
        self._vocabulary = vocabulary
        self._postprocess = postprocess

        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._all_coders, self._fetches, None

    def collect_results(self, results: List[Dict]) -> None:
        if len(results) > 1:
            raise ValueError("CRFRunner does not support ensembling.")

        loss = results[0].get("loss", 0.)
        input_mask = results[0]["input_mask"]

        decoded_indices = results[0]["decoded_sequence"]

        assert (input_mask == results[0]["input_mask"]).all()

        # CAUTION! FABULOUS HACK BELIEVE ME
        decoded_indices -= END_TOKEN_INDEX
        decoded_indices *= input_mask.astype(int)
        decoded_indices += END_TOKEN_INDEX

        # must transpose argmaxes because vectors_to_sentences is time-major
        decoded_labels = \
            self._vocabulary.vectors_to_sentences(decoded_indices.T)

        if self._postprocess is not None:
            decoded_labels = self._postprocess(decoded_labels)

        self.result = ExecutionResult(
            outputs=decoded_labels,
            losses=[loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class CRFRunner(BaseRunner[CRFLabeler]):

    def __init__(self,
                 output_series: str,
                 decoder: CRFLabeler,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[CRFLabeler].__init__(self, output_series, decoder)

        self._postprocess = postprocess

        # Make sure the lazy decoder creates its output tensor
        log("Decoder output tensor: {}".format(decoder.decoded))

    # pylint: disable=unused-argument
    # Don't know why it works in Attention.attention and not here.
    # Parameters are unused beacause they are inherited.
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> CRFRunExecutable:
        fetches = {
            "decoded_sequence": self._decoder.decoded,
            "sequence_score": self._decoder.sequence_score,
            "input_mask": self._decoder.encoder.input_sequence.temporal_mask,
            "seq_len": self._decoder.encoder.lengths,
            "logits": self._decoder.logits}

        if compute_losses:
            fetches["loss"] = self._decoder.cost

        return CRFRunExecutable(
            self.all_coders, fetches, self._decoder.vocabulary,
            self._postprocess)
    # pylint: enable: unused-argument

    @property
    def loss_names(self) -> List[str]:
        return ["loss"]
