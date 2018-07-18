#!/usr/bin/env python3.5
""" Tests the config parsing module. """
# pylint: disable=protected-access

import unittest
from neuralmonkey.config import parsing

SPLITTER_TESTS = [
    ["empty", "", []],
    ["only_commas", ",,,,,,", []],
    ["commas_and_whitespace", ",    ,   ,,   , , ", []],
    ["no_commas", "without", ["without"]],
    ["common", "a,b,c", ["a", "b", "c"]],
    ["brackets", "(brackets),(brac,kets)", ["(brackets)", "(brac,kets)"]],
]


class TestParsing(unittest.TestCase):

    def test_splitter_bad_brackets(self):
        self.assertRaises(Exception, parsing._split_on_commas,
                          "(omg,brac],kets")

    def test_parse_value_int(self):
        self.assertEqual(parsing._parse_value("42", {}), 42)
        self.assertEqual(parsing._parse_value("-42", {}), -42)

    def test_parse_value_float(self):
        self.assertAlmostEqual(parsing._parse_value("0.5e-1", {}), 0.5e-1)
        self.assertAlmostEqual(parsing._parse_value(".5e-1", {}), .5e-1)
        self.assertAlmostEqual(parsing._parse_value("-.5e-1", {}), -.5e-1)
        self.assertAlmostEqual(parsing._parse_value("5.e-1", {}), 5.e-1)

    def test_parse_value_string(self):
        varz = {"pi": 3.14159, "greeting": "hello"}
        self.assertEqual(parsing._parse_value('"{greeting}"world"', varz),
                         'hello"world')
        self.assertEqual(parsing._parse_value('"pi = {pi:.0f}"', varz),
                         "pi = 3")


def test_splitter_gen(a, b):
    def test_case_fun(self):
        out = parsing._split_on_commas(a)
        self.assertEqual(out, b)
    return test_case_fun


if __name__ == "__main__":
    for case in SPLITTER_TESTS:
        test_name = "test_{}".format(case[0])
        test = test_splitter_gen(case[1], case[2])
        setattr(TestParsing, test_name, test)
    unittest.main()
