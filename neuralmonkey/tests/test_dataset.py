#!/usr/bin/env python3.5

import os
import tempfile
from typing import Iterable, List
import unittest

from neuralmonkey.dataset import LazyDataset, from_files
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader


class TestDataset(unittest.TestCase):

    def test_nonexistent_file(self):
        paths_and_readers = {
            "source": (["some_nonexistent_file"], UtfPlainTextReader)}

        with self.assertRaises(FileNotFoundError):
            LazyDataset("name", paths_and_readers, {}, None)

    def test_glob(self):
        filenames = sorted(["abc1", "abc2", "abcxx", "xyz"])
        contents = ["a", "b", "c", "d"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname, text in zip(filenames, contents):
                with open(os.path.join(tmp_dir, fname), "w") as file:
                    print(text, file=file)

            dataset = from_files(
                s_data=[os.path.join(tmp_dir, "abc?"),
                        os.path.join(tmp_dir, "xyz*")])

            self.assertEqual(dataset.get_series("data"), [["a"], ["b"], ["d"]])

    def test_lazy_dataset(self):
        i = 0  # iteration counter

        def reader(files: List[str]) -> Iterable[List[str]]:
            del files
            nonlocal i
            for i in range(10):  # pylint: disable=unused-variable
                yield ["foo"]

        dataset = from_files(
            s_data=([], reader),
            preprocessors=[("data", "data_prep", lambda x: x)],
            lazy=True)
        series = dataset.get_series("data_prep")

        # Check that the reader is being iterated lazily
        for j in range(5):
            next(series)
            self.assertEqual(i, j)


if __name__ == "__main__":
    unittest.main()
