from typing import Iterator, Dict, List
import numpy as np
from neuralmonkey.logging import log
from neuralmonkey.writers.plain_text_writer import Writer


def numpy_array_writer(path: str, data: np.ndarray) -> None:
    np.save(path, data)
    log("Result saved as numpy array to '{}'".format(path))


def numpy_dict_writer(
        path: str, data: Iterator[Dict[str, np.ndarray]]) -> None:
    unbatched = dict(
        zip(data[0], zip(*[d.values() for d in data])))

    np.savez(path, **unbatched)
    log("Result saved as numpy data to '{}.npz'".format(path))
