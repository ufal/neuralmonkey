import numbers
from typing import Any, List, Tuple
import numpy as np
from scipy.stats import norm


# pylint: disable=too-few-public-methods
class GaussDensityEvaluator(object):

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, decoded: List[Tuple[float, float]],
                 reference: List[Any]) -> float:
        if isinstance(reference[0], numbers.Real):
            targets = np.array(reference)
        elif isinstance(reference[0], list):
            targets = np.array([len(l) for l in reference])
        else:
            raise ValueError(
                "Provided data series must consist either "
                "of numbers or lists, but was '{}''."
                .format(type(reference[0])))

        density_sum = 0.
        for (mean, stddev), target in zip(decoded, targets):
            density_sum += norm.pdf(target, mean, stddev)
        return density_sum / len(decoded)
# pylint: enable=too-few-public-methods


# pylint: disable=invalid-name
GaussDensity = GaussDensityEvaluator("GaussDensity")
# pylint: enable=invalid-name
