from typing import Dict, List, Callable

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.sequence_regressor import SequenceRegressor
from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import BaseRunner

# pylint: disable=invalid-name
Postprocessor = Callable[[List[float]], List[float]]
# pylint: enable=invalid-name


class RegressionRunner(BaseRunner[SequenceRegressor]):
    """A runnner that takes the predictions of a sequence regressor."""

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(BaseRunner.Executable["RegressionRunner"]):

        def collect_results(self, results: List[Dict]) -> None:
            predictions_sum = np.zeros_like(results[0]["prediction"])
            mse_loss = 0.

            for sess_result in results:
                if "mse" in sess_result:
                    mse_loss += sess_result["mse"]

                predictions_sum += sess_result["prediction"]

            predictions = (predictions_sum / len(results)).tolist()

            if self.executor.postprocess is not None:
                predictions = self.executor.postprocess(predictions)

            self.set_runner_result(outputs=predictions, losses=[mse_loss])
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: SequenceRegressor,
                 postprocess: Postprocessor = None) -> None:
        check_argument_types()
        BaseRunner[SequenceRegressor].__init__(self, output_series, decoder)
        self.postprocess = postprocess

    @tensor
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"prediction": self.decoder.predictions,
                "mse": self.decoder.cost}

    @property
    def loss_names(self) -> List[str]:
        return ["mse"]
