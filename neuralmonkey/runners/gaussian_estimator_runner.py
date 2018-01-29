from typing import Dict, List, Callable, Set, cast

import numpy as np
import tensorflow as tf

from typeguard import check_argument_types
from neuralmonkey.decoders.gaussian_estimator import GaussianEstimator
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
# pylint: disable=too-few-public-methods


class GaussianEstimatorRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: GaussianEstimator) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True) -> Executable:
        decoder = cast(GaussianEstimator, self._decoder)

        if compute_losses:
            fetches = {"neg_log_density": decoder.cost}
        else:
            fetches = {}

        fetches["mean"] = decoder.distribution.mean()
        fetches["stddev"] = decoder.distribution.stddev()

        return GaussianEstimatorRunExecutable(self.all_coders, fetches)

    @property
    def loss_names(self) -> List[str]:
        return ["neg_log_density"]


class GaussianEstimatorRunExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: Dict[str, tf.Tensor]) -> None:

        self.all_coders = all_coders
        self._fetches = fetches
        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        """Get the feedables and tensors to run."""
        return self.all_coders, self._fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        means_sum = np.zeros_like(results[0]["mean"])
        stddevs_sum = np.zeros_like(results[0]["stddev"])
        neg_density_loss = 0.

        for sess_result in results:
            if "neg_log_density" in sess_result:
                neg_density_loss += sess_result["neg_log_density"]

            means_sum += sess_result["mean"]
            stddevs_sum += sess_result["stddev"]

        means = means_sum / len(results)
        stddevs = stddevs_sum / len(results)

        formatted_output = ["{:.4g} +/- {:.4g}".format(m, s)
                            for m, s in zip(means, stddevs)]

        self.result = ExecutionResult(
            outputs=formatted_output,
            losses=[neg_density_loss],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
