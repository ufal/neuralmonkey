from typing import List

import numpy as np
from sklearn.metrics import f1_score


# pylint: disable=too-few-public-methods
class F1Evaluator(object):
    """F1 evaluator for Quality Estimation tagging.

    The entities are usually 'OK' and 'BAD' (or something along these lines)
    The mapping for custom entity labels can be provided by the user.
    """

    def __init__(self, name: str = "F1 measure",
                 labels: List[str] = None) -> None:
        if labels is None:
            labels = ["BAD", "OK"]
        self.name = name
        self.tag_map = {tag: i for i, tag in enumerate(labels)}

    def __call__(self, decoded: List[List[str]],
                 references: List[List[str]]) -> float:
        assert len(decoded) == len(references)
        d = _flatten(decoded)
        r = _flatten(references)

        f1_bad, f1_good = f1_score(r, d, average=None, pos_label=None)

        return f1_bad * f1_good


# convert list of lists into a flat list
def _flatten(lofl):
    if _list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif isinstance(lofl, dict):
        return lofl.values()

    # TODO: raise Exception here instead
    return False


def _list_of_lists(a_list):
    if not isinstance(a_list, (list, tuple, np.ndarray)):
        return False
    if not a_list:
        return False
    if not all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return False
    return True


# pylint: disable=invalid-name
QEF1Score = F1Evaluator()
# pylint: enable=invalid-name
