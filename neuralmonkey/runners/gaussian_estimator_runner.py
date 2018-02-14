from typing import Dict, List, Set, cast

import numpy as np
import tensorflow as tf

from typeguard import check_argument_types
from neuralmonkey.decoders.gaussian_estimator import GaussianEstimator
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)


_LOSS_NAMES = ["neg_log_density", "mean_squared_error", "stddev_value_loss",
               "cost"]


# pylint: disable=too-few-public-methods
class GaussianEstimatorRunner(BaseRunner):

    def __init__(self,
                 output_series: str,
                 decoder: GaussianEstimator) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)
        log("Decoder cost: {}".format(decoder.cost))

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1) -> Executable:
        decoder = cast(GaussianEstimator, self._decoder)

        if compute_losses:
            fetches = {
                "neg_log_density": decoder.gauss_density_loss,
                "mean_squared_error": decoder.mse_loss,
                "stddev_value_loss": decoder.stddev_value_loss,
                "cost": decoder.cost}
        else:
            fetches = {}

        fetches["mean"] = decoder.distribution.mean()
        fetches["stddev"] = decoder.distribution.stddev()

        return GaussianEstimatorRunExecutable(
            self.all_coders, fetches, num_sessions)

    @property
    def loss_names(self) -> List[str]:
        return _LOSS_NAMES


class GaussianEstimatorRunExecutable(Executable):

    def __init__(self,
                 all_coders: Set[ModelPart],
                 fetches: Dict[str, tf.Tensor],
                 num_sessions: int) -> None:

        self.all_coders = all_coders
        self._fetches = fetches
        self._num_sessions = num_sessions
        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        return (self.all_coders, self._fetches,
                [{} for _ in range(self._num_sessions)])

    def collect_results(self, results: List[Dict]) -> None:
        means_sum = np.zeros_like(results[0]["mean"])
        stddevs_sum = np.zeros_like(results[0]["stddev"])
        neg_density_loss = 0.

        losses = [0. for _ in _LOSS_NAMES]
        for sess_result in results:
            losses = [loss + sess_result[name]
                      for loss, name in zip(losses, _LOSS_NAMES)]

            means_sum += sess_result["mean"]
            stddevs_sum += sess_result["stddev"]

        means = means_sum / len(results)
        stddevs = stddevs_sum / len(results)

        # formatted_output = ["{:.4g} +/- {:.4g}".format(m, s)
        #                     for m, s in zip(means, stddevs)]

        self.result = ExecutionResult(
            outputs=np.expand_dims(means, 1).tolist(),
            losses=losses,
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)
