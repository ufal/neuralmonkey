import re
from typing import List

import numpy as np

# tests: mypy
# pylint: disable=too-few-public-methods

ID_SEP = re.compile(r"[-:]")

class WordAlignmentPreprocessor(object):

    def __init__(self, source_len, target_len, dtype=np.float32,
                 normalize=True, zero_based=True):
        self._source_len = source_len
        self._target_len = target_len
        self._dtype = dtype
        self._normalize = normalize
        self._zero_based = zero_based

    def __call__(self, sentence: List[str]):
        result = np.zeros((self._target_len, self._source_len), self._dtype)

        for ali in sentence:
            ids, _, str_weight = ali.partition('/')
            i, j = map(int, ID_SEP.split(ids))
            weight = float(str_weight) if str_weight != '' else 1.

            if not self._zero_based:
                i -= 1
                j -= 1

            if i < self._source_len and j < self._target_len:
                result[j][i] = weight

        if self._normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                result /= result.sum(axis=1, keepdims=True)
                result[np.isnan(result)] = 0

        return result
