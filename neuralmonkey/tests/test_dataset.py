#!/usr/bin/env python3.5

from typing import Iterable, List
import os
import tempfile
import unittest

from neuralmonkey.dataset import from_files
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader


class TestDataset(unittest.TestCase):

    def test_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            from_files(
                name="name",
                s_source=(["some_nonexistent_file"], UtfPlainTextReader),
                lazy=True)

    def test_lazy_dataset(self):
        i = 0  # iteration counter

        def reader(files: List[str]) -> Iterable[List[str]]:
            del files
            nonlocal i
            for i in range(10):  # pylint: disable=unused-variable
                yield ["foo"]

        dataset = from_files(
            name="data",
            s_source=([], reader),
            preprocessors=[("source", "source_prep", lambda x: x)],
            lazy=True)

        series = dataset.get_series("source_prep")

        # Check that the reader is being iterated lazily
        for j, _ in enumerate(series):
            self.assertEqual(i, j)
        self.assertEqual(i, 9)

    def test_glob(self):
        filenames = sorted(["abc1", "abc2", "abcxx", "xyz"])
        contents = ["a", "b", "c", "d"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname, text in zip(filenames, contents):
                with open(os.path.join(tmp_dir, fname), "w") as file:
                    print(text, file=file)

            dataset = from_files(
                name="dataset",
                s_data=[os.path.join(tmp_dir, "abc?"),
                        os.path.join(tmp_dir, "xyz*")])

            series_iterator = dataset.get_series("data")
            self.assertEqual(list(series_iterator), [["a"], ["b"], ["d"]])


if __name__ == "__main__":
    unittest.main()
