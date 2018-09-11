#!/usr/bin/env python3.5

from typing import Iterable, List
import os
import tempfile
import unittest

from neuralmonkey.dataset import Dataset, from_files, load
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader


class TestDataset(unittest.TestCase):

    def test_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load(name="name",
                 series=["source"],
                 sources=[(["some_nonexistent_file"], UtfPlainTextReader)],
                 lazy=True,
                 buffer_size=5)

    def test_nonexistent_file_deprec(self):
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

        dataset = load(
            name="data",
            series=["source", "source_prep"],
            sources=[([], reader), (lambda x: x, "source")],
            lazy=True,
            buffer_size=5)

        series = dataset.get_series("source_prep")

        # Check that the reader is being iterated lazily
        for j, _ in enumerate(series):
            self.assertEqual(i, j)
        self.assertEqual(i, 9)

    def test_lazy_dataset_deprec(self):
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

            dataset = load(
                name="dataset",
                series=["data"],
                sources=[[os.path.join(tmp_dir, "abc?"),
                          os.path.join(tmp_dir, "xyz*")]])

            series_iterator = dataset.get_series("data")
            self.assertEqual(list(series_iterator), [["a"], ["b"], ["d"]])

    def test_glob_deprec(self):
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

    def test_batching_eager_noshuffle(self):
        iterators = {
            "a": lambda: range(5),
            "b": lambda: range(10, 15)
        }

        dataset = Dataset(
            "dataset", iterators=iterators, lazy=False, shuffled=False)

        batches = []
        for epoch in range(2):
            epoch = []
            for batch in dataset.batches(3):
                epoch.append({s: list(batch.get_series(s)) for s in iterators})

            batches.append(epoch)

        self.assertEqual(
            batches, [[{"a": [0, 1, 2], "b": [10, 11, 12]},
                       {"a": [3, 4], "b": [13, 14]}],
                      [{"a": [0, 1, 2], "b": [10, 11, 12]},
                       {"a": [3, 4], "b": [13, 14]}]])

    def test_batching_lazy_noshuffle(self):
        iterators = {
            "a": lambda: range(5),
            "b": lambda: range(10, 15)
        }

        dataset = Dataset(
            "dataset", iterators=iterators, lazy=True, shuffled=False,
            buffer_size=4)

        batches = []
        for epoch in range(2):
            epoch = []
            for batch in dataset.batches(3):
                epoch.append({s: list(batch.get_series(s)) for s in iterators})

            batches.append(epoch)

        self.assertEqual(
            batches, [[{"a": [0, 1, 2], "b": [10, 11, 12]},
                       {"a": [3, 4], "b": [13, 14]}],
                      [{"a": [0, 1, 2], "b": [10, 11, 12]},
                       {"a": [3, 4], "b": [13, 14]}]])

    def test_batching_eager_shuffle(self):
        iterators = {
            "a": lambda: range(5),
            "b": lambda: range(5, 10)
        }

        dataset = Dataset(
            "dataset", iterators=iterators, lazy=False, shuffled=True)

        batches = []
        for epoch in range(2):
            epoch = []
            for batch in dataset.batches(3):
                epoch.append({s: list(batch.get_series(s)) for s in iterators})

            batches.append(epoch)

        epoch_data = []
        epoch_data.append(
            [c for batch in batches[0] for b in batch.values() for c in b])
        epoch_data.append(
            [c for batch in batches[1] for b in batch.values() for c in b])

        self.assertEqual(set(epoch_data[0]), set(range(10)))
        self.assertEqual(set(epoch_data[0]), set(epoch_data[1]))
        self.assertNotEqual(epoch_data[0], epoch_data[1])

    def test_batching_lazy_shuffle(self):
        iterators = {
            "a": lambda: range(5),
            "b": lambda: range(5, 10)
        }

        dataset = Dataset(
            "dataset", iterators=iterators, lazy=True, shuffled=True,
            buffer_size=4)

        batches = []
        for epoch in range(2):
            epoch = []
            for batch in dataset.batches(3):
                epoch.append({s: list(batch.get_series(s)) for s in iterators})

            batches.append(epoch)

        epoch_data = []
        epoch_data.append(
            [c for batch in batches[0] for b in batch.values() for c in b])
        epoch_data.append(
            [c for batch in batches[1] for b in batch.values() for c in b])

        self.assertEqual(set(epoch_data[0]), set(range(10)))
        self.assertEqual(set(epoch_data[0]), set(epoch_data[1]))
        self.assertNotEqual(epoch_data[0], epoch_data[1])


if __name__ == "__main__":
    unittest.main()
