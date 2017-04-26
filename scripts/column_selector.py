#!/usr/bin/env python3.5
"""
Select lines from N files according to an index file. This work nicely in
combination with the max_column_finder.py script.

These two scripts are particularly useful for hypotheses rescoring.
"""
import argparse


def main() -> None:
    # pylint: disable=no-member
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", metavar="SELECTOR",
                        type=argparse.FileType("r"),
                        help="file with column indices")
    parser.add_argument("input_files", nargs="+",
                        metavar="INPUT_FILES", type=argparse.FileType("r"),
                        help="the files to traverse")
    args = parser.parse_args()

    for lines in zip(*([args.selector] + args.input_files)):
        index = int(lines[0])
        print(lines[index + 1].strip())


if __name__ == "__main__":
    main()
