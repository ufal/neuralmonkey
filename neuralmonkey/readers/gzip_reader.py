import gzip

class GZipReader(object):

    def __init__(self, path, encoding="utf-8"):
        self.path = path
        self.encoding = encoding

    def read(self):
        with gzip.open(self.path, "r", encoding=self.encoding) as f_data:
            for line in f_data:
                yield line.strip().split(" ")
