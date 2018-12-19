#!/usr/bin/env python3.5
"""Unit tests for readers"""

import tempfile
import numpy as np
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.framework.errors_impl import InvalidArgumentError
# pylint: enable=no-name-in-module

from neuralmonkey.readers.string_vector_reader import (
    get_string_vector_reader, float_vector_reader, int_vector_reader)
from neuralmonkey.readers.plain_text_reader import t2t_tokenized_text_reader


def _make_file(from_var):
    tmpfile = tempfile.NamedTemporaryFile(mode="w+")
    tmpfile.write(from_var)
    tmpfile.seek(0)
    return tmpfile


class TestStringVectorReader(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

        cls.str_ints = """ 1   2 3
                           4 5   6
                           7 8 9 10 """

        cls.str_floats = """ 1 2       3.5
                             4 -5.0e10     6
                             7 8 9.2e-12 10.1134123112312 """

        cls.str_ints_fine = """1 2 3
                               4 5 6
                               7 8 9 """

        cls.list_ints = [np.array(row.strip().split(), dtype=np.int32)
                         for row in cls.str_ints.strip().split("\n")]

        cls.list_floats = [np.array(row.strip().split(), dtype=np.float32)
                           for row in cls.str_floats.strip().split("\n")]

        cls.list_ints_fine = [np.array(row.strip().split(), dtype=np.int32)
                              for row in cls.str_ints_fine.strip().split("\n")]

    def setUp(self):
        self.tmpfile_floats = _make_file(self.str_floats)
        self.tmpfile_ints = _make_file(self.str_ints)
        self.tmpfile_ints_fine = _make_file(self.str_ints_fine)

    def test_float_reader(self):
        dataset = float_vector_reader([self.tmpfile_floats.name])
        iterator = dataset.make_one_shot_iterator().get_next()

        with self.test_session():
            floats = [iterator.eval() for _ in range(3)]

        equals = [np.array_equal(f, g)
                  for f, g in zip(floats, self.list_floats)]

        for comp in equals:
            self.assertTrue(comp)

    def test_int_reader(self):
        dataset = int_vector_reader(
            [self.tmpfile_ints.name, self.tmpfile_ints_fine.name])
        iterator = dataset.make_one_shot_iterator().get_next()

        with self.test_session():
            ints = [iterator.eval() for _ in range(3)]

        equals = [np.array_equal(f, g)
                  for f, g in zip(ints, self.list_ints + self.list_ints_fine)]

        for comp in equals:
            self.assertTrue(comp)

    def test_columns(self):
        for cols in range(2, 4):

            with self.assertRaisesRegex(
                    InvalidArgumentError, "Bad number of columns"):
                r = get_string_vector_reader(np.int32, columns=cols)
                dataset = r([self.tmpfile_ints.name])
                iterator = dataset.make_one_shot_iterator().get_next()

                with self.test_session():
                    list(iterator.eval() for _ in range(3))

            with self.assertRaisesRegex(
                    InvalidArgumentError, "Bad number of columns"):
                r = get_string_vector_reader(np.float32, columns=cols)
                dataset = r([self.tmpfile_floats.name])
                iterator = dataset.make_one_shot_iterator().get_next()

                with self.test_session():
                    list(iterator.eval() for _ in range(3))

            if cols != 3:
                with self.assertRaisesRegex(
                        InvalidArgumentError, "Bad number of columns"):
                    r = get_string_vector_reader(np.int32, columns=cols)
                    dataset = r([self.tmpfile_ints_fine.name])
                    iterator = dataset.make_one_shot_iterator().get_next()

                    with self.test_session():
                        list(iterator.eval() for _ in range(3))

        r = get_string_vector_reader(np.int32, columns=3)
        dataset = r([self.tmpfile_ints_fine.name])
        iterator = dataset.make_one_shot_iterator().get_next()

        with self.test_session():
            ints = list(iterator.eval() for _ in range(3))

        equals = [np.array_equal(f, g)
                  for f, g in zip(ints, self.list_ints_fine)]

        for comp in equals:
            self.assertTrue(comp)

    def tearDown(self):
        self.tmpfile_ints.close()
        self.tmpfile_floats.close()
        self.tmpfile_ints_fine.close()


class TestT2TReader(tf.test.TestCase):

    def setUp(self):
        self.reader = t2t_tokenized_text_reader

    def test_reader(self):
        text = "Ich bin  der čermák -=- - !!! alfonso "
        gold_tokens = ["Ich", "bin", "  ", "der", "čermák", " -=- - !!! ",
                       "alfonso"]

        tmpfile = _make_file(text)
        dataset = self.reader([tmpfile.name])
        iterator = dataset.make_one_shot_iterator().get_next()

        read = []
        with self.test_session():
            while True:
                try:
                    line = iterator.eval().tolist()
                    read.append([tf.compat.as_text(l) for l in line])
                except tf.errors.OutOfRangeError:
                    break

        tmpfile.close()

        self.assertEqual(len(read), 1)
        self.assertSequenceEqual(read[0], gold_tokens)


if __name__ == "__main__":
    tf.test.main()
