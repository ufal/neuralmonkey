#!/usr/bin/env python3.5

import unittest

from neuralmonkey.dataset import LazyDataset
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader


class TestDataset(unittest.TestCase):

    def test_nonexistent_file(self):
        paths_and_readers = {
            "source": (["some_nonexistent_file"], UtfPlainTextReader)}

        with self.assertRaises(FileNotFoundError):
            LazyDataset("name", paths_and_readers, {}, None)


if __name__ == "__main__":
    unittest.main()
