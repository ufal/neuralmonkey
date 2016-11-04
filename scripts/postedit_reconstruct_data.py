#!/usr/bin/env python3

"""
This a script that takes the result of automatic postediting encoded as a
sequence of <keep>, <delete> and insert operations and applies them on the
original text being post-edited.

The inverse script to this one is 'postedit_prepare_data.py'.
"""

# tests: lint, mypy

import argparse
from neuralmonkey.processors.german import GermanPreprocessor
from neuralmonkey.processors.german import GermanPostprocessor
from postedit_prepare_data import load_tokenized

# TODO make reconstruct a postprocessor


def reconstruct(source, edits):
    index = 0
    target = []

    for edit in edits:
        if edit == '<keep>':
            if index < len(source):
                target.append(source[index])
            index += 1

        elif edit == '<delete>':
            index += 1

        else:
            target.append(edit)

    # we may have created a shorter sequence of edit ops due to the
    # decoder limitations -> now copy the rest of source
    if index < len(source):
        target.extend(source[index:])

    return target


def main():
    parser = argparse.ArgumentParser(
        description="Convert postediting target data to sequence of edits")
    parser.add_argument("--edits", type=argparse.FileType('r'), required=True)
    parser.add_argument("--translated-sentences",
                        type=argparse.FileType('r'), required=True)
    parser.add_argument("--target-german", type=bool, default=False)

    args = parser.parse_args()

    postprocess = lambda x: x
    preprocess = None  # type: GermanPreprocessor
    if args.target_german:
        # pylint: disable=redefined-variable-type
        postprocess = GermanPostprocessor()
        preprocess = GermanPreprocessor()

    trans_sentences = load_tokenized(
        args.translated_sentences, preprocess=preprocess)
    edit_sequences = load_tokenized(args.edits, preprocess=None)

    for trans, edits in zip(trans_sentences, edit_sequences):
        target = reconstruct(trans, edits)
        # TODO refactor this (change postprocessor api)
        print(" ".join(postprocess([target])[0]))


if __name__ == '__main__':
    #edits = ['<keep>', 'ahoj', '<delete>', 'proc?']
    #source = ['Karle', 'co', 'kdy']
    # print reconstruct(source, edits)

    main()
