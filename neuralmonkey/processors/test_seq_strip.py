import unittest

from neuralmonkey.processors.seq_strip import strip, left_strip, right_strip


class TestSeqStrip(unittest.TestCase):

    def test_left(self):
        arr = [1, 2, 3]

        prep0 = left_strip(0)
        prep2 = left_strip(2)

        self.assertSequenceEqual(prep0(arr), arr)
        self.assertSequenceEqual(prep2(arr), [3])

    def test_right(self):
        arr = [1, 2, 3]

        prep0 = right_strip(0)
        prep2 = right_strip(2)

        self.assertSequenceEqual(prep0(arr), arr)
        self.assertSequenceEqual(prep2(arr), [1])

    def test_both(self):
        arr = [1, 2, 3]

        prep0 = strip(0, 0)
        prep1 = strip(left=1, right=1)
        prep2 = strip(0, 1)
        prep3 = strip(3, 3)

        self.assertSequenceEqual(prep0(arr), arr)
        self.assertSequenceEqual(prep1(arr), [2])
        self.assertSequenceEqual(prep2(arr), [1, 2])
        self.assertSequenceEqual(prep3(arr), [])

    def test_neg(self):
        with self.assertRaises(ValueError):
            strip(left=-1, right=3)

        with self.assertRaises(ValueError):
            strip(left=1, right=-3)
