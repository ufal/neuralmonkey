from typing import List, Callable
import sys
import unicodedata

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types


ALNUM_CHARSET = set(
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith("L")
        or unicodedata.category(chr(i)).startswith("N")))


def string_reader(files: List[str]) -> tf.data.Dataset:
    compressed = ["GZIP" if path.endswith(".gz") else None for path in files]
    dataset = tf.data.TextLineDataset(files[0], compressed[0])

    for path, compression in zip(files[1:], compressed[1:]):
        new_dataset = tf.data.TextLineDataset(path, compression)
        # pylint: disable=redefined-variable-type
        dataset = dataset.concatenate(new_dataset)
        # pylint: enable=redefined-variable-type

    return dataset


def tokenized_text_reader(files: List[str]) -> tf.data.Dataset:
    """Get dataset of space-separated tokenized text."""

    def tokenize(line: tf.Tensor) -> tf.Tensor:
        return tf.sparse.to_dense(
            tf.strings.split([tf.strings.strip(line)]), default_value="")[0]

    return string_reader(files).map(tokenize)


def t2t_tokenized_text_reader(files: List[str]) -> tf.data.Dataset:
    """Get dataset with tokenizer for plain text.

    Tokenization is inspired by the tensor2tensor tokenizer:
    https://github.com/tensorflow/tensor2tensor/blob/v1.5.5/tensor2tensor/data_generators/text_encoder.py

    The text is split to groups of consecutive alphanumeric or non-alphanumeric
    tokens, dropping single spaces inside the text. Basically the goal here is
    to preserve the whitespace around weird characters and whitespace on weird
    positions (beginning and end of the text).
    """

    def tokenize(b_line: bytes) -> np.ndarray:
        line = tf.compat.as_text(b_line)

        if not line:
            return []
        line = line.strip()

        tokens = []
        is_alnum = [ch in ALNUM_CHARSET for ch in line]
        current_token_start = 0

        for pos in range(1, len(line)):
            # Boundary of alnum and non-alnum character groups
            if is_alnum[pos] != is_alnum[pos - 1]:
                token = line[current_token_start:pos]

                # Drop single space if it's not on the beginning
                if token != " " or current_token_start == 0:
                    tokens.append(tf.compat.as_bytes(token))

                current_token_start = pos

        # Add a final token (even if it's a single space)
        final_token = line[current_token_start:]
        tokens.append(tf.compat.as_bytes(final_token))

        return np.array(tokens, dtype=np.object)

    def mapper(line: tf.Tensor) -> tf.Tensor:
        tokenized = tf.py_func(tokenize, [line], tf.string)
        tokenized.set_shape([None])
        return tokenized

    return string_reader(files).map(mapper)


def column_separated_reader(
        column: int, default: str = "", delimiter: str = "\t",
        header: bool = False) -> Callable[[List[str]], tf.data.Dataset]:
    """Get reader for delimiter-separated tokenized text.

    Args:
        files: List of input files.
        column: number of column to be returned. Indices start at 0.
        default: When the value is missing, it is replaced by this.
        delimiter: Delimiter. Defaults to tab.
        header: If True, ignore first line.

    Returns:
        The newly created dataset.
    """
    check_argument_types()

    def reader(files: List[str]) -> tf.data.Dataset:
        compression = None
        if all(path.endswith(".gz") for path in files):
            compression = "GZIP"
        elif any(path.endswith(".gz") for path in files):
            raise ValueError("All CSV files must be either compressed or not.")

        return tf.data.experimental.CsvDataset(
            files, [default], compression_type=compression, header=header,
            field_delim=delimiter, select_cols=[column])

    return reader


def csv_reader(column: int) -> Callable[[List[str]], tf.data.Dataset]:
    return column_separated_reader(column, delimiter=",")


def tsv_reader(column: int) -> Callable[[List[str]], tf.data.Dataset]:
    return column_separated_reader(column, delimiter="\t")
