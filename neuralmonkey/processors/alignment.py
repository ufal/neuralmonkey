import re
from typing import List

import numpy as np

# pylint: disable=too-few-public-methods

ID_SEP = re.compile(r"[-:]")


class WordAlignmentPreprocessor(object):
    """A preprocessor for word alignments in a text format.

    One of the following formats is expected:

        s1-t1 s2-t2 ...

        s1:1/w1 s2:t2/w2 ...

    where each `s` and `t` is the index of a word in the source and target
    sentence, respectively, and `w` is the corresponding weight. If the weight
    is not given, it is assumend to be 1. The separators `-` and `:` are
    interchangeable.

    The output of the preprocessor is an alignment matrix of the fixed shape
    (target_len, source_len) for each sentence.
    """

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
