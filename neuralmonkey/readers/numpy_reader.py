from typing import List
import os

from typeguard import check_argument_types
import numpy as np


def single_tensor(files: List[str]):
    """Load a single tensor from a numpy file."""
    check_argument_types()
    if len(files) == 1:
        return np.load(files[0])

    return np.concatenate([np.load(f) for f in files], axis=0)


def from_file_list(prefix: str, default_tensor_name: str = "arr_0"):
    """Load list of numpy arrays according to a list of files.

    Args:
        prefix: A common prefix of the files in lists of relative paths.
        default_tensor_name: Key of the tensors to load in the npz files.

    Return:
        A reader function that loads numpy arrays from files on path writen
        path relatively to the given prefix.
    """
    check_argument_types()

    def load(files: List[str]):
        for list_file in files:
            with open(list_file, encoding="utf-8") as f_list:
                for line in f_list:
                    path = os.path.join(prefix, line.rstrip())
                    with np.load(path) as npz:
                        yield npz[default_tensor_name]

    return load


# pylint: disable=invalid-name
numpy_file_list_reader = from_file_list("")
