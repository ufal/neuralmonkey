
class PlainTextFileReader(object):

    def __init__(self, path, encoding="utf-8"):
        self.path = path
        self.encoding = encoding

    def read(self):
        # type: () -> Iterator[List[str]]
        with open(self.path, encoding=self.encoding) as f_data:
            for line in f_data:
                yield line.strip().split(" ")
