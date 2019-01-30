from typing import Dict, List, Union

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.runners.base_runner import BaseRunner
from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.decoders.autoregressive import AutoregressiveDecoder
from neuralmonkey.decoders.classifier import Classifier
from neuralmonkey.trainers.generic_trainer import GenericTrainer

# pylint: disable=invalid-name
SupportedDecoder = Union[AutoregressiveDecoder, Classifier]
# pylint: enable=invalid-name


class GradientRunner(BaseRunner[GenericModelPart]):
    """Runner for fetching gradients computed over the dataset.

    Gradient runner applies provided trainer on a desired dataset
    and uses it to compute gradients over the gold data. It is currently
    used to gather gradients for Elastic Weight Consolidation.

    (https://arxiv.org/pdf/1612.00796.pdf)
    """

    # pylint: disable=too-few-public-methods
    class Executable(BaseRunner.Executable["GradientRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            assert len(results) == 1

            for sess_result in results:
                gradient_dict = {}
                tensor_names = [
                    t.name for t in self.executor.fetches()["gradients"]]
                for name, val in zip(tensor_names, sess_result["gradients"]):
                    gradient_dict[name] = val

            self.set_runner_result(outputs=gradient_dict, losses=[])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: SupportedDecoder,
                 trainer: GenericTrainer) -> None:
        check_argument_types()
        BaseRunner[GenericModelPart].__init__(
            self, output_series, decoder)

        self._gradients = trainer.gradients

    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"gradients": [g[1] for g in self._gradients]}

    @property
    def loss_names(self) -> List[str]:
        return []
