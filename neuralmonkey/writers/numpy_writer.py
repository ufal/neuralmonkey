from typing import Iterator, Dict, List
import numpy as np
from neuralmonkey.logging import log
from neuralmonkey.writers.plain_text_writer import Writer


def numpy_array_writer(path: str, data: np.ndarray) -> None:
    np.save(path, data)
    log("Result saved as numpy array to '{}'".format(path))


def _check_savable_dict(data: Any) -> bool:
    """Check if the data is of savable type.

    Arguments:
        data: Variable that holds some results.

    Returns:
        Boolean that says whether the saving of this type is implemented.
    """
    if not (data and data[0]):
        return False

    supported_type = Union[
        List[Dict[str, np.ndarray]],
        List[List[Dict[str, np.ndarray]]]]

    return match_type(data, supported_type)  # type: ignore


def numpy_dict_writer(
        path: str, data: Iterator[Dict[str, np.ndarray]]) -> None:

    # assert check_savable_dict(data)
    unbatched = dict(
        zip(data[0], zip(*[d.values() for d in data])))

    np.savez(path, **unbatched)
    log("Result saved as numpy data to '{}.npz'".format(path))
