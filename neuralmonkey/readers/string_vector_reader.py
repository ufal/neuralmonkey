from typing import List, Type, Callable

import numpy as np
import tensorflow as tf

from neuralmonkey.readers.plain_text_reader import tokenized_text_reader


def get_string_vector_reader(
        dtype: Type = np.float32,
        columns: int = None) -> Callable[[List[str]], tf.data.Dataset]:
    """Get a dataset from vectors encoded as whitespace-separated numbers."""

    def reader(files: List[str]) -> tf.data.Dataset:
        def process_line(line: tf.Tensor) -> tf.Tensor:
            if columns is not None:
                cond = tf.assert_equal(
                    tf.shape(line), [columns], message="Bad number of columns")
                with tf.control_dependencies([cond]):
                    line = tf.identity(line)

            return tf.strings.to_number(line, out_type=tf.as_dtype(dtype))
        return tokenized_text_reader(files).map(process_line)
    return reader


def float_vector_reader(files: List[str]) -> tf.data.Dataset:
    return get_string_vector_reader(dtype=np.float32)(files)


def int_vector_reader(files: List[str]) -> tf.data.Dataset:
    return get_string_vector_reader(dtype=np.int32)(files)


# pylint: disable=invalid-name
def FloatVectorReader(*args, **kwargs):
    raise NotImplementedError(
        "FloatVectorReader is deprecated. Use float_vector_reader instead")


def IntVectorReader(*args, **kwargs):
    raise NotImplementedError(
        "IntVectorReader is deprecated. Use int_vector_reader instead")
# pylint: enable=invalid-name
