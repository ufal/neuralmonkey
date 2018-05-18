import io
from typing import Callable, List

import numpy as np


def _load_file(filename: str):
    with open(filename, "rb") as f_in:
        if f_in.seekable():
            return np.load(f_in)
        else:
            buf = io.BytesIO(f_in.read())
            return np.load(buf)


def numpy_reader(lazy: bool = False) -> Callable:
    def reader(files: List[str]):
        if len(files) == 1:
            return _load_file(files[0])

        if lazy:
            return (record for f in files for record in _load_file(f))
        else:
            return np.concatenate([_load_file(f) for f in files], axis=0)

    return reader
