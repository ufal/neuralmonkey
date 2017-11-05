#!/usr/bin/env python3
"""Loads and builds a given config file in memory.

Can be used for checking that a model can be loaded successfully, or for
generating a vocabulary from a dataset, without the need to run the model.
"""

import argparse

from neuralmonkey.config.parsing import parse_file
from neuralmonkey.config.builder import build_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="a configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        _, config_dict = parse_file(f)

    build_config(config_dict, ignore_names=set())


if __name__ == '__main__':
    main()
