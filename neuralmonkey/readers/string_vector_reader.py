from typing import List, Iterable, Type
import gzip
import numpy as np

from neuralmonkey.readers.utils import FILETYPER


def get_string_vector_reader(dtype: Type = np.float32, columns: int = None):
    """Get a reader for vectors encoded as whitespace-separated numbers"""
    def process_line(line: str, lineno: int, path: str) -> np.ndarray:
        numbers = line.strip().split()
        if columns is not None and len(numbers) != columns:
            raise ValueError("Wrong number of columns ({}) on line {}, file {}"
                             .format(len(numbers), lineno, path))

        return np.array(numbers, dtype=dtype)

    def reader(files: List[str])-> Iterable[List[np.ndarray]]:
        for path in files:
            current_line = 0

            mime_type = FILETYPER.from_file(path)

            if mime_type == "application/gzip":
                with gzip.open(path, 'r') as f_data:
                    for line in f_data:
                        current_line += 1
                        if line.strip():
                            yield process_line(str(line), current_line, path)
            else:
                with open(path) as f_data:
                    for line in f_data:
                        current_line += 1
                        if line.strip():
                            yield process_line(line, current_line, path)

    return reader


# pylint: disable=invalid-name
FloatVectorReader = get_string_vector_reader(np.float32)
IntVectorReader = get_string_vector_reader(np.int32)
# pylint: enable=invalid-name
