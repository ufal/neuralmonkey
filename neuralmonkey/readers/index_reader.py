from typing import Callable, IO, Iterable, List

import gzip
import io
import os
import tarfile

from neuralmonkey.readers.utils import FILETYPER


def index_reader(prefix: str="", encoding: str='utf-8') -> Callable:
    """Get a reader of a file "index". This can be either a plain text file
    containing file paths, or a tar file.

    Args:
        prefix: Prefix of the paths to the files.
        encoding: Encoding in which to open the files. Use 'binary' for binary
            mode.

    Returns:
        The reader function that takes the index file and returns a generator
        of open file objects.
    """

    def reader(files: List[str]) -> Iterable:
        for path in files:
            if tarfile.is_tarfile(path):
                return _read_tar(path, encoding)
            else:
                return _read_list(path, encoding, prefix)

    return reader


def _read_tar(path: str, encoding: str) -> Iterable:
    with tarfile.open(path) as tar:
        for item in tar.getmembers():
            item_f = tar.extractfile(item)
            if hasattr(item_f, 'raw'):
                item_f.raw.name = item.name  # type: ignore
            if encoding != 'binary':
                yield io.TextIOWrapper(item_f, encoding=encoding)
            else:
                yield item_f


def _read_list(path: str, encoding: str, prefix: str) -> Iterable[IO]:
    mime_type = FILETYPER.from_file(path)

    if mime_type == 'application/gzip':
        list_f = gzip.open(path)
    elif mime_type == 'text/plain':
        list_f = open(path)
    else:
        raise ValueError("Unsupported file type '{}' for '{}'"
                         .format(mime_type, path))

    with list_f:
        for item_path in list_f:
            item_path = os.path.expanduser(os.path.join(prefix,
                                                        item_path.rstrip()))

            if encoding == 'binary':
                yield open(item_path, 'rb')
            else:
                yield open(item_path, encoding=encoding)
