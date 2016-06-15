#!/usr/bin/env python

import codecs

class PlainTextFileReader(object):

    def __init__(self, path, encoding="utf-8"):
        self.path = path
        self.encoding = encoding

    def read(self):
        # type: Generator[List[str]]
        with codecs.open(self.path, "r", self.encoding) as f_data:
            for line in f_data:
                yield line.strip().split(" ")
