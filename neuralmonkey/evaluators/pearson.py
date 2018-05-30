from typing import List
import math
import numpy as np


def _average(x):
    assert x
    return float(sum(x)) / len(x)


def _pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = _average(x)
    avg_y = _average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


class PearsonCorrelationEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 name: str = "PearsonCorrelation") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[List[float]],
                 references: List[List[float]]) -> float:

        dec = np.array(decoded).flatten()
        ref = np.array(references).flatten()
        return _pearson_def(dec, ref)


# pylint: disable=invalid-name
Pearson = PearsonCorrelationEvaluator()
# pylint: enable=invalid-name
