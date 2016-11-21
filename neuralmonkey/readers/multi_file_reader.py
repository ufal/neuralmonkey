import magic

from neuralmonkey.readers.plain_text_reader import PlainTextFileReader
from neuralmonkey.readers.gzip_reader import GZipReader

class MultiFileReader(object):

    def __init__(self, paths, encoding="utf-8"):
        self.paths = paths
        self.encoding = encoding

    def read(self):
        for path in self.paths:
            if magic.from_file(path, mime=True) == "application/gzip":
                reader = GZipReader(path, self.encoding)
                yield reader.read()
            else:
                reader = PlainTextFileReader(path, self.encoding)
                yield reader.read()
