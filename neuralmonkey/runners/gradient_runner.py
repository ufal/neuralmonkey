from typing import Any, Dict, List, Set, Union, Optional

from typeguard import check_argument_types

from neuralmonkey.runners.base_runner import (
    BaseRunner, Executable, ExecutionResult, NextExecute)
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.trainers.generic_trainer import GenericTrainer

# pylint: disable=invalid-name
SupportedDecoder = Union[AutoregressiveDecoder, Classifier]
# pylint: enable=invalid-name


class GradientRunnerExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: Dict[str, List[Any]]) -> None:
        self._all_coders = all_coders
        self._fetches = fetches

        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self._all_coders, self._fetches, []

    def collect_results(self, results: List[Dict]) -> None:
        assert len(results) == 1

        for sess_result in results:
            gradient_dict = {}
            tensor_names = [t.name for t in self._fetches["gradients"]]
            for name, val in zip(tensor_names, sess_result["gradients"]):
                gradient_dict[name] = val

        self.result = ExecutionResult(
            outputs=[gradient_dict],
            losses=[],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class GradientRunner(BaseRunner[SupportedDecoder]):
    """Runner for fetching gradients computed over the dataset.

    Gradient runner applies provided trainer on a desired dataset
    and uses it to compute gradients over the gold data. It is currently
    used to gather gradients for Elastic Weight Consolidation.

    (https://arxiv.org/pdf/1612.00796.pdf)
    """

    def __init__(self,
                 output_series: str,
                 trainer: GenericTrainer,
                 decoder: SupportedDecoder) -> None:
        check_argument_types()
        BaseRunner[AutoregressiveDecoder].__init__(
            self, output_series, decoder)

        self._gradients = trainer.gradients

    # pylint: disable=unused-argument
    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> GradientRunnerExecutable:
        fetches = {"gradients": [g[1] for g in self._gradients]}

        return GradientRunnerExecutable(self.all_coders, fetches)
    # pylint: enable=unused-argument

    @property
    def loss_names(self) -> List[str]:
        return []
