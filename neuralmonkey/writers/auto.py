from typing import Any, List, Dict, Union
import collections
import numpy as np

from neuralmonkey.util.match_type import match_type
from neuralmonkey.writers.plain_text_writer import (
    Writer, tokenized_text_writer, text_writer)
from neuralmonkey.writers.numpy_writer import (
    numpy_array_writer, numpy_dict_writer)


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


def auto_writer(encoding: str = "utf-8") -> Writer:

    text_tok_writer = tokenized_text_writer(encoding)
    text_plain_writer = text_writer(encoding)

    def writer(path: str, data: Any) -> None:
        if isinstance(data, np.ndarray):
            numpy_array_writer(path, data)
        elif _check_savable_dict(data):
            numpy_dict_writer(path, data)
        elif isinstance(next(iter(data)), collections.Iterable):
            text_tok_writer(path, data)
        else:
            text_plain_writer(path, data)

    return writer


# pylint: disable=invalid-name
AutoWriter = auto_writer()
