#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
""" Unit tests for the decoder. (Tests only initialization so far) """

import unittest
import copy

from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import Vocabulary

DECODER_PARAMS = dict(
    encoders=[],
    vocabulary=Vocabulary(),
    data_id="foo",
    name="test-decoder",
    max_output_len=5,
    dropout_keep_prob=1.0,
    embedding_size=10,
    rnn_size=10)


class TestDecoder(unittest.TestCase):

    def test_init(self):
        decoder = Decoder(**DECODER_PARAMS)
        self.assertIsNotNone(decoder)

    def test_max_output_len(self):
        dparams = copy.deepcopy(DECODER_PARAMS)

        dparams["max_output_len"] = -10
        with self.assertRaises(ValueError):
            Decoder(**dparams)

    def test_dropout(self):
        dparams = copy.deepcopy(DECODER_PARAMS)

        dparams["dropout_keep_prob"] = -0.5
        with self.assertRaises(ValueError):
            Decoder(**dparams)

        dparams["dropout_keep_prob"] = 1.5
        with self.assertRaises(ValueError):
            Decoder(**dparams)

    def test_embedding_size(self):
        dparams = copy.deepcopy(DECODER_PARAMS)

        dparams["embedding_size"] = None
        with self.assertRaises(ValueError):
            dec = Decoder(**dparams)
            print(dec.embedding_size)

        dparams["embedding_size"] = -10
        with self.assertRaises(ValueError):
            Decoder(**dparams)

    def test_cell_type(self):
        dparams = copy.deepcopy(DECODER_PARAMS)

        dparams.update({"rnn_cell": "bogus_cell"})
        with self.assertRaises(ValueError):
            Decoder(**dparams)

        for cell_type in ("GRU", "LSTM", "NematusGRU"):
            print(dparams)
            dparams["rnn_cell"] = cell_type
            dparams["name"] = "test-decoder-{}".format(cell_type)
            Decoder(**dparams)


if __name__ == "__main__":
    unittest.main()
