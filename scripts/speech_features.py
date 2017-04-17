#!/usr/bin/env python3
"""
This script precomputes speech features for a given set of recordings and
saves them as a list of NumPy arrays.

usage example:
  %(prog)s src.train src.train.npy -t mfcc -o delta_order 2
"""

import argparse
import os
from typing import Union

import numpy as np

from neuralmonkey.processors.speech import SpeechFeaturesPreprocessor
from neuralmonkey.readers.audio_reader import audio_reader


def try_parse_number(str_value: str) -> Union[str, float, int]:
    value = str_value
    try:
        value = float(str_value)
        value = int(str_value)
    except ValueError:
        pass
    
    return value


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='a file with a list of audio files')
    parser.add_argument('output', help='the .npy output file')
    parser.add_argument('-f', '--format',
                        default='wav',
                        help='the audio format (default: %(default)s)')
    parser.add_argument('-p', '--prefix',
                        help='the prefix for the audio file paths (default: '
                        'the input file location)')
    parser.add_argument('-t', '--type',
                        default='mfcc',
                        help='the feature type (default: %(default)s)')
    parser.add_argument('-o', '--option',
                        nargs=2, action='append', default=[],
                        metavar=('OPTION', 'VALUE'),
                        help='other arguments for SpeechFeaturesPreprocessor')

    args = parser.parse_args()


    prefix = args.prefix
    if prefix is None:
        prefix = os.path.dirname(os.path.abspath(args.input))

    feats_kwargs = {k: try_parse_number(v) for k, v in args.option}

    read = audio_reader(prefix=prefix, audio_format=args.format)
    process = SpeechFeaturesPreprocessor(
        feature_type=args.type, **feats_kwargs)

    output = [process(audio) for audio in read([args.input])]
    
    np.save(args.output, output)


if __name__ == '__main__':
    main()
