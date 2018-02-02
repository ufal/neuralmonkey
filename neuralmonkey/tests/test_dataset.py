#!/usr/bin/env python3.5

from typing import Iterable, List
import unittest

from neuralmonkey.dataset import LazyDataset
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader


class TestDataset(unittest.TestCase):

    def test_nonexistent_file(self):
        paths_and_readers = {
            "source": (["some_nonexistent_file"], UtfPlainTextReader)}

        with self.assertRaises(FileNotFoundError):
            LazyDataset("name", paths_and_readers, {}, None)

    def test_lazy_dataset(self):
        i = 0  # iteration counter

        def reader(files: List[str]) -> Iterable[List[str]]:
            del files
            nonlocal i
            for i in range(10):  # pylint: disable=unused-variable
                yield ["foo"]

        dataset = LazyDataset(
            name="data",
            series_paths_and_readers={"source": ([], reader)},
            series_outputs={},
            preprocessors=[("source", "source_prep", lambda x: x)])
        series = dataset.get_series("source_prep")

        # Check that the reader is being iterated lazily
        for j, _ in enumerate(series):
            self.assertEqual(i, j)
        self.assertEqual(i, 9)


if __name__ == "__main__":
    unittest.main()
