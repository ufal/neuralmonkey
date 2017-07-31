from typing import List, Iterable, Callable
import gzip
import csv
import io
import sys

from neuralmonkey.logging import warn


# pylint: disable=invalid-name
PlainTextFileReader = Callable[[List[str]], Iterable[List[str]]]
# pylint: enable=invalid-name

csv.field_size_limit(sys.maxsize)


def string_reader(
        encoding: str = "utf-8") -> Callable[[List[str]], Iterable[str]]:
    def reader(files: List[str]) -> Iterable[str]:
        for path in files:
            if path.endswith(".gz"):
                with gzip.open(path, 'r') as f_data:
                    for line in f_data:
                        yield str(line, 'utf-8')
            else:
                with open(path, encoding=encoding) as f_data:
                    for line in f_data:
                        yield line

    return reader


def tokenized_text_reader(encoding: str = "utf-8") -> PlainTextFileReader:
    """Get reader for space-separated tokenized text."""
    def reader(files: List[str]) -> Iterable[List[str]]:
        lines = string_reader(encoding)
        for line in lines(files):
            yield line.strip().split(' ')

    return reader


def column_separated_reader(
        column: int, delimiter: str = "\t", quotechar: str = csv.QUOTE_NONE,
        encoding: str = "utf-8") -> PlainTextFileReader:
    """Get reader for delimiter-separated tokenized text.

    Args:
        column: number of column to be returned. It starts with 1 for the first
    """
    def reader(files: List[str]) -> Iterable[List[str]]:
        column_count = None
        text_reader = string_reader(encoding)
        for line in text_reader(files):
            io_line = io.StringIO(line.rstrip('\r\n'))
            if quotechar is None:
                parsed_csv = list(csv.reader(io_line, delimiter=delimiter,
                                             quotechar=quotechar,
                                             skipinitialspace=True))
            else:
                parsed_csv = list(csv.reader(io_line, delimiter=delimiter,
                                             quoting=csv.QUOTE_NONE,
                                             skipinitialspace=True))
            columns = len(parsed_csv[0])
            if column_count is None:
                column_count = columns
            elif column_count != columns:
                warn("A mismatch in number of columns. Expected {} got {}"
                     .format(column_count, columns))
            if columns < column:
                warn("There is a missing column number {} in the dataset."
                     .format(column))
                yield []
            else:
                yield parsed_csv[0][column - 1].split(' ')

    return reader


def csv_reader(column: int):
    return column_separated_reader(column=column, delimiter=',', quotechar='"')


def tsv_reader(column: int):
    return column_separated_reader(column=column, delimiter='\t',
                                   quotechar=csv.QUOTE_NONE)


# pylint: disable=invalid-name
UtfPlainTextReader = tokenized_text_reader()
# pylint: enable=invalid-name
