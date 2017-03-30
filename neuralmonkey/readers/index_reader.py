from typing import Callable, IO, Iterable, List, Optional, Union

import gzip
import io
import os
import tarfile

from neuralmonkey.readers.utils import FILETYPER


# pylint: disable=invalid-name
FileOrPath = Union[IO[str], IO[bytes], str]
# pylint: enable=invalid-name


def index_reader(prefix: str="", encoding: Optional[str]='utf-8') -> Callable:
    """Get a reader of a file "index". This can be either a plain text file
    containing file paths, or a tar file.

    Args:
        prefix: Prefix of the paths to the files.
        encoding: Encoding in which to open the files. Use 'binary' for binary
            mode, and None to get file paths (the latter does not work for tar
            files). Default is 'utf-8'.

    Returns:
        The reader function that takes the index file and returns a generator
        of open file objects or file paths.
    """

    def reader(files: List[str]) -> Iterable[FileOrPath]:
        for path in files:
            if tarfile.is_tarfile(path):
                return _read_tar(path, encoding)
            else:
                return _read_list(path, encoding, prefix)

    return reader


def _read_tar(path: str, encoding: Optional[str]) -> Iterable[FileOrPath]:
    if encoding is None:
        # returning file paths is not supported for tar files
        encoding = 'utf-8'

    with tarfile.open(path) as tar:
        for item in tar.getmembers():
            item_f = tar.extractfile(item)
            if hasattr(item_f, 'raw'):
                item_f.raw.name = item.name  # type: ignore

            if encoding == 'binary':
                yield item_f
            else:
                yield io.TextIOWrapper(item_f, encoding=encoding)


def _read_list(path: str,
               encoding: Optional[str], prefix: str) -> Iterable[FileOrPath]:
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

            if encoding is None:
                yield item_path
            elif encoding == 'binary':
                yield open(item_path, 'rb')
            else:
                yield open(item_path, encoding=encoding)
