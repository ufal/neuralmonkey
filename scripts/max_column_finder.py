#!/usr/bin/env python3.5
"""
For each line in N input files with scores, outputs the index of the file
which has maximum value on that line.
Output can be processed with the column_selector.py script.

These two scripts are particularly useful for hypotheses rescoring.
"""
import argparse
import numpy as np


def main() -> None:
    # pylint: disable=no-member
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_files", nargs="+",
                        metavar="SCORE_FILES", type=argparse.FileType("r"),
                        help="the files to traverse")
    args = parser.parse_args()

    for lines in zip(*args.score_files):
        numbers = np.array(lines, dtype=float)
        best_index = np.argmax(numbers)
        # best_score = np.amax(numbers)

        print(best_index)


if __name__ == "__main__":
    main()
