#!/usr/bin/env python3.5
"""Unit tests for readers"""

import unittest
import tempfile
import numpy as np

from neuralmonkey.readers.string_vector_reader import get_string_vector_reader

STRING_INTS = """
1   2 3
4 5   6
7 8 9 10

"""

LIST_INTS = [np.array(row.strip().split(), dtype=np.int32)
             for row in STRING_INTS.strip().split("\n")]

STRING_FLOATS = """
1 2       3.5
      4 -5.0e10     6
7 8 9.2e-12 10.1123213213214123141234123112312312
"""

LIST_FLOATS = [np.array(row.strip().split(), dtype=np.float32)
               for row in STRING_FLOATS.strip().split("\n")]

STRING_INTS_FINE = """
1 2 3
4 5 6
7 8 9
"""

LIST_INTS_FINE = [np.array(row.strip().split(), dtype=np.int32)
                  for row in STRING_INTS_FINE.strip().split("\n")]


def _make_file(from_var):
    tmpfile = tempfile.NamedTemporaryFile(mode="w+")
    tmpfile.write(from_var)
    tmpfile.seek(0)
    return tmpfile


class TestStringVectorReader(unittest.TestCase):

    def setUp(self):
        self.tmpfile_floats = _make_file(STRING_FLOATS)
        self.tmpfile_ints = _make_file(STRING_INTS)
        self.tmpfile_ints_fine = _make_file(STRING_INTS_FINE)

    def test_reader(self):
        r = get_string_vector_reader(np.float32)
        floats = list(r([self.tmpfile_floats.name]))
        equals = [np.array_equal(f, g) for f, g in zip(floats, LIST_FLOATS)]

        for comp in equals:
            self.assertTrue(comp)

        r = get_string_vector_reader(np.int32)
        ints = list(r([self.tmpfile_ints.name, self.tmpfile_ints_fine.name]))
        equals = [np.array_equal(f, g)
                  for f, g in zip(ints, LIST_INTS + LIST_INTS_FINE)]

        for comp in equals:
            self.assertTrue(comp)

    def test_columns(self):
        for cols in range(2, 4):
            with self.assertRaisesRegex(ValueError, "Wrong number of columns"):
                r = get_string_vector_reader(np.int32, columns=cols)
                list(r([self.tmpfile_ints.name]))

            with self.assertRaisesRegex(ValueError, "Wrong number of columns"):
                r = get_string_vector_reader(np.float32, columns=cols)
                list(r([self.tmpfile_floats.name]))

            if cols != 3:
                with self.assertRaisesRegex(ValueError,
                                            "Wrong number of columns"):
                    r = get_string_vector_reader(np.int32, columns=cols)
                    list(r([self.tmpfile_ints_fine.name]))

        r = get_string_vector_reader(np.int32, columns=3)
        ints = list(r([self.tmpfile_ints_fine.name]))
        equals = [np.array_equal(f, g)
                  for f, g in zip(ints, LIST_INTS_FINE)]

        for comp in equals:
            self.assertTrue(comp)

    def tearDown(self):
        self.tmpfile_ints.close()
        self.tmpfile_floats.close()
        self.tmpfile_ints_fine.close()


if __name__ == "__main__":
    unittest.main()
