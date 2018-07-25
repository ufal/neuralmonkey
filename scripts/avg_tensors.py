#!/usr/bin/env python3
"""Compute the mean over a set of tensors.

The tensors can be spread over multiple npz files. The mean is computed
over the first dimension (supposed to be a batch).

"""

import argparse
import os
import re
import glob

import numpy as np

from neuralmonkey.logging import log as _log


def log(message: str, color: str = "blue") -> None:
    _log(message, color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file_prefix", type=str,
                        help="prefix of the npz files to be averaged")
    parser.add_argument("--output_path", type=str,
                        help="Path to output the averaged checkpoint to.")
    args = parser.parse_args()

    output_dict = {}
    n = 0
    for file in glob.glob("{}.*npz".format(args.file_prefix)):
        log("Processing {}".format(file))
        tensors = np.load(file)

        # first dimension must be equal for all tensors (batch)
        shapes = [tensors[f].shape for f in tensors.files]
        assert all([x[0] == shapes[0][0] for x in shapes])

        for varname in tensors.files:
            res = np.sum(tensors[varname], 0)
            if varname in output_dict:
                output_dict[varname] += res
            else:
                output_dict[varname] = res
        n += shapes[0][0]

    for name in output_dict:
        output_dict[name] /= n

    np.savez(args.output_path, **output_dict)


if __name__ == "__main__":
    main()
