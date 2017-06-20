from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from neuralmonkey.dataset import Dataset


# pylint: disable=too-few-public-methods
class Preprocess(object):
    """Preprocessor transorming two series into series of edit operations."""
    def __init__(self,
                 source_id: str, target_id: str,
                 steps_only: bool = False) -> None:
        self._source_id = source_id
        self._target_id = target_id
        self._steps_only = steps_only

    def __call__(self, dataset: Dataset) -> Iterable[List[str]]:
        source_series = dataset.get_series(self._source_id)
        target_series = dataset.get_series(self._target_id)

        for src_seq, tgt_seq in zip(source_series, target_series):
            yield convert_to_edits(src_seq, tgt_seq, self._steps_only)


class Postprocess(object):
    """Proprocessor applying edit operations on a series."""

    def __init__(
            self, source_id: str, edits_id: str,
            result_postprocess: Callable[
                [Iterable[List[str]]], Iterable[List[str]]]=None,
            steps_only: bool = False) -> None:

        self._source_id = source_id
        self._edits_id = edits_id
        self._result_postprocess = result_postprocess
        self._steps_only = steps_only

    def _do_postprocess(
            self, dataset: Dataset,
            generated_series: Dict[str, Iterable[Any]]) -> Iterable[List[str]]:

        source_series = generated_series.get(self._source_id)
        if source_series is None:
            source_series = dataset.get_series(self._source_id)
        edits_series = generated_series.get(self._edits_id)
        if edits_series is None:
            edits_series = dataset.get_series(self._edits_id)

        for src_seq, edit_seq in zip(source_series, edits_series):
            if self._steps_only:
                reconstructed = remove_steps(edit_seq)
            else:
                reconstructed = reconstruct(src_seq, edit_seq)
            yield reconstructed

    def __call__(
            self, dataset: Dataset,
            generated_series: Dict[str, Iterable[Any]]) -> Iterable[List[str]]:

        reconstructed_seq = self._do_postprocess(dataset, generated_series)

        if self._result_postprocess is not None:
            return self._result_postprocess(reconstructed_seq)

        return reconstructed_seq
# pylint: enable=too-few-public-methods
KEEP = '<keep>'
DELETE = '<delete>'
STEP = '<step>'

def convert_to_edits(
        source: List[str],
        target: List[str],
        steps_only: bool = False) -> List[str]:
    """ Create a sequence of edit operations

    steps_only: insert only <step> labels instead
        of <keep>, <delete>
        (see https://arxiv.org/pdf/1706.04138.pdf)
    """


    lev = np.zeros([len(source) + 1, len(target) + 1])
    edits = [[[] for _ in range(len(target) + 1)]
             for _ in range(len(source) + 1)]  # type: List[List[List[str]]]

    for i in range(len(source) + 1):
        lev[i, 0] = i
        edits[i][0] = [STEP if steps_only else DELETE for _ in range(i)]

    for j in range(len(target) + 1):
        lev[0, j] = j
        edits[0][j] = target[:j]

    for j in range(1, len(target) + 1):
        for i in range(1, len(source) + 1):

            if source[i - 1] == target[j - 1]:
                keep_cost = lev[i - 1, j - 1]
            else:
                keep_cost = np.inf

            delete_cost = lev[i - 1, j] + 1
            insert_cost = lev[i, j - 1] + 1

            lev[i, j] = min(keep_cost, delete_cost, insert_cost)

            if lev[i, j] == keep_cost:
                edits[i][j] = edits[i - 1][j - 1] + [KEEP]
                if steps_only:
                    edits[i][j] = edits[i - 1][j - 1] +\
                        [STEP] + [target[j - 1]]

            elif lev[i, j] == delete_cost:
                edits[i][j] = edits[i - 1][j] +\
                     [STEP if steps_only else DELETE]

            else:
                edits[i][j] = edits[i][j - 1] + [target[j - 1]]

    res = edits[-1][-1]
    if res[0] == STEP:
        res = res[1:]

    return res


def reconstruct(source: List[str], edits: List[str]) -> List[str]:
    index = 0
    target = []

    for edit in edits:
        if edit == KEEP:
            if index < len(source):
                target.append(source[index])
            index += 1

        elif edit == DELETE:
            index += 1

        else:
            target.append(edit)

    # we may have created a shorter sequence of edit ops due to the
    # decoder limitations -> now copy the rest of source
    if index < len(source):
        target.extend(source[index:])

    return target

def remove_steps(input_sequence: List[str]) -> List[str]:
    target = []

    for token in input_sequence:
        if token != STEP:
            target.append(token)

    return target
