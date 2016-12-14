from typing import List

import numpy as np

# tests: mypy
# pylint: disable=too-few-public-methods

class WordAlignmentPreprocessor(object):

    def __init__(self, source_len, target_len, dtype=np.float32,
                 normalize=True):
        self._source_len = source_len
        self._target_len = target_len
        self._dtype = dtype
        self._normalize = normalize

    def __call__(self, sentence: List[str]):
        result = np.zeros((self._target_len, self._source_len), self._dtype)

        for ali in sentence:
            i, j = map(int, ali.split("-"))
            if i < self._source_len and j < self._target_len:
                result[j][i] = 1

        if self._normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                result /= result.sum(axis=1, keepdims=True)
                result[np.isnan(result)] = 0

        return result
