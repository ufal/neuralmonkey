from typing import List, Iterable
import gzip

from neuralmonkey.readers.utils import FILETYPER


def get_plain_text_reader(encoding: str="utf-8"):
    """Get reader for space-separated tokenized text."""
    def reader(files: List[str]) -> Iterable[List[str]]:
        for path in files:
            mime_type = FILETYPER.from_file(path)

            if mime_type == "application/gzip":
                with gzip.open(path, 'r') as f_data:
                    for line in f_data:
                        yield str(line, 'utf-8').strip().split(" ")
            else:
                with open(path, encoding=encoding) as f_data:
                    for line in f_data:
                        yield line.strip().split(" ")

    return reader


# pylint: disable=invalid-name
UtfPlainTextReader = get_plain_text_reader()
