#!/usr/bin/env python3.5

from typing import Iterable, List
import os
import tempfile
import unittest

import tensorflow as tf

from neuralmonkey.dataset import Dataset, load, BatchingScheme
from neuralmonkey.readers.plain_text_reader import tokenized_text_reader

DEFAULT_BATCHING_SCHEME = BatchingScheme(batch_size=3)


class TestDataset(tf.test.TestCase):

    def test_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load(name="name",
                 series=["source"],
                 data=[(["some_nonexistent_file"], tokenized_text_reader)],
                 batching=DEFAULT_BATCHING_SCHEME,
                 buffer_size=5)

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
                data=[[os.path.join(tmp_dir, "abc?"),
                       os.path.join(tmp_dir, "xyz*")]],
                batching=DEFAULT_BATCHING_SCHEME)

            tfdata = dataset.get_dataset(
                types={"data": tf.string},
                shapes={"data": tf.TensorShape([None])})

            iterator = tfdata.make_one_shot_iterator()

            with self.test_session():
                data = iterator.get_next()["data"].eval()

            self.assertTrue(all(data == [[b"a"], [b"b"], [b"d"]]))

    def test_batching_noshuffle(self):
        iterators = {
            "a": tf.data.Dataset.range(5),
            "b": tf.data.Dataset.range(10, 15)
        }

        dataset = Dataset(
            "dataset", data_series=iterators, batching=DEFAULT_BATCHING_SCHEME)

        tfdata = dataset.get_dataset(
            types={"a": tf.int32, "b": tf.int32},
            shapes={"a": tf.TensorShape([None]), "b": tf.TensorShape([None])})

        iterator = tfdata.make_initializable_iterator()

        batches = []
        with self.test_session() as sess:
            for _ in range(2):
                sess.run(iterator.initializer)
                epoch = []
                while True:
                    try:
                        batch_f = sess.run(iterator.get_next())
                        epoch.append(
                            tf.contrib.framework.nest.map_structure(
                                lambda x: x.tolist(), batch_f))
                    except tf.errors.OutOfRangeError:
                        break
                batches.append(epoch)

        self.assertAllEqual(
            batches, [[{"a": [0, 1, 2], "b": [10, 11, 12]},
                       {"a": [3, 4], "b": [13, 14]}],
                      [{"a": [0, 1, 2], "b": [10, 11, 12]},
                       {"a": [3, 4], "b": [13, 14]}]])


class TestBucketing(tf.test.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.types = {"sentences": tf.string}
        cls.shapes = {"sentences": tf.TensorShape([None])}

        def gen() -> Iterable:
            # testing dataset is 50 sequences of lengths 1 - 50
            return (["word" for _ in range(l)] for l in range(1, 50))

        cls.dataset = tf.data.Dataset.from_generator(
            gen, cls.types["sentences"], cls.shapes["sentences"])

    def _fetch_dataset(self, scheme: BatchingScheme) -> List:
        dataset = Dataset("dataset",
                          data_series={"sentences": self.dataset},
                          batching=scheme)

        tfdata = dataset.get_dataset(self.types, self.shapes)
        iterator = tfdata.make_one_shot_iterator()

        next_element = iterator.get_next()

        batches = []
        with self.test_session() as sess:
            while True:
                try:
                    batch_f = sess.run(next_element)
                    batch_sent = tf.contrib.framework.nest.map_structure(
                        tf.compat.as_text,
                        tf.contrib.framework.nest.map_structure(
                            lambda x: x.tolist(), batch_f))["sentences"]

                    for sent in batch_sent:
                        while True:
                            try:
                                sent.remove("<pad>")
                            except ValueError:
                                break

                    batches.append(batch_sent)

                except tf.errors.OutOfRangeError:
                    break

        return batches

    def test_bucketing(self) -> None:

        # we use batch size 7 and bucket span 10
        scheme = BatchingScheme(bucket_boundaries=[10, 20, 30, 40, 50],
                                bucket_batch_sizes=[7, 7, 7, 7, 7, 7])

        batches = self._fetch_dataset(scheme)
        ref_batches = [
            [["word" for _ in range(l)] for l in range(1, 8)],
            [["word" for _ in range(l)] for l in range(10, 17)],
            [["word" for _ in range(l)] for l in range(20, 27)],
            [["word" for _ in range(l)] for l in range(30, 37)],
            [["word" for _ in range(l)] for l in range(40, 47)],
            [["word" for _ in range(l)] for l in range(8, 10)],
            [["word" for _ in range(l)] for l in range(17, 20)],
            [["word" for _ in range(l)] for l in range(27, 30)],
            [["word" for _ in range(l)] for l in range(37, 40)],
            [["word" for _ in range(l)] for l in range(47, 50)]]

        self.assertSequenceEqual(ref_batches, batches)

    @unittest.skip("drop_remainder is not supported in bucketing")
    def test_bucketing_no_leftovers(self):

        # we use batch size 7 and bucket span 10
        scheme = BatchingScheme(bucket_boundaries=[10, 20, 30, 40, 50],
                                bucket_batch_sizes=[7, 7, 7, 7, 7, 7],
                                drop_remainder=True)

        batches = self._fetch_dataset(scheme)

        ref_batches = [
            [["word" for _ in range(l)] for l in range(1, 8)],
            [["word" for _ in range(l)] for l in range(10, 17)],
            [["word" for _ in range(l)] for l in range(20, 27)],
            [["word" for _ in range(l)] for l in range(30, 37)],
            [["word" for _ in range(l)] for l in range(40, 47)]]

        self.assertSequenceEqual(ref_batches, batches)


if __name__ == "__main__":
    tf.test.main()
