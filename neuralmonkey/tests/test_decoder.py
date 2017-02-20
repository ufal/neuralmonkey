#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
""" Unit tests for the decoder. (Tests only initialization so far) """

import unittest

from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import Vocabulary


class TestDecoder(unittest.TestCase):

    def test_init(self):
        decoder = Decoder(
            encoders=[],
            vocabulary=Vocabulary(),
            data_id="foo",
            name="test-decoder",
            max_output_len=5,
            dropout_keep_prob=1.0,
            embedding_size=10,
            rnn_size=10)
        self.assertIsNotNone(decoder)


if __name__ == "__main__":
    unittest.main()
