#!/usr/bin/env python3.5

import os
import tempfile
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


if __name__ == "__main__":
    unittest.main()
