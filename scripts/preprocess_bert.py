#!/usr/bin/env python3
"""Creates training data for the BERT network training
(noisified + masked gold predictions) using the input corpus.

The masked Gold predictions use Neural Monkey's PAD_TOKEN to indicate
tokens that should not be classified during training.

We only leave `coverage` percent of symbols for classification. These
symbols are left unchanged on input with a probability of `1 - mask_prob`.
If they are being changed, they are replaced by the `mask_token` with a
probability of `1 - replace_prob` and by a random vocabulary token otherwise.
"""

import argparse
import os

import numpy as np

from neuralmonkey.logging import log as _log
from neuralmonkey.vocabulary import (
    Vocabulary, PAD_TOKEN, UNK_TOKEN, from_wordlist)


def log(message: str, color: str = "blue") -> None:
    _log(message, color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=str, default="/dev/stdin")
    parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--mask_token", type=str, default=UNK_TOKEN,
                        help="token used to mask the tokens")
    parser.add_argument("--coverage", type=float, default=0.15,
                        help=("percentage of tokens that should be left "
                              "for classification during training"))
    parser.add_argument("--mask_prob", type=float, default=0.8,
                        help=("probability of the classified token being "
                             "replaced by a different token on input"))
    parser.add_argument("--replace_prob", type=float, default=0.1,
                        help=("probability of the classified token being "
                              "replaced by a random token instead of "
                              "mask_token"))
    parser.add_argument("--vocab_contains_header", type=bool, default=True)
    parser.add_argument("--vocab_contains_frequencies",
                        type=bool, default=True)
    args = parser.parse_args()

    assert (args.coverage <= 1 and args.coverage >= 0)
    assert (args.mask_prob <= 1 and args.mask_prob >= 0)
    assert (args.replace_prob <= 1 and args.replace_prob >= 0)

    log("Loading vocabulary.")
    vocabulary = from_wordlist(
        args.vocabulary,
        contains_header=args.vocab_contains_header,
        contains_frequencies=args.vocab_contains_frequencies)

    mask_prob = args.mask_prob
    replace_prob = args.replace_prob
    keep_prob = 1 - mask_prob - replace_prob
    sample_probs = (keep_prob, mask_prob, replace_prob)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.input_file
    out_f_noise = "{}.noisy".format(output_prefix)
    out_f_mask = "{}.mask".format(output_prefix)

    out_noise_h = open(out_f_noise, "w", encoding="utf-8")
    out_mask_h = open(out_f_mask, "w", encoding="utf-8")
    log("Processing data.")
    with open(args.input_file, "r", encoding="utf-8") as input_h:
        # TODO: performance optimizations
        for line in input_h:
            line = line.strip().split(" ")
            num_samples = int(args.coverage * len(line))
            sampled_indices = np.random.choice(len(line), num_samples, False)

            output_noisy = list(line)
            output_masked = [PAD_TOKEN] * len(line)
            for i in sampled_indices:
                random_token = np.random.choice(vocabulary.index_to_word[4:])
                new_token = np.random.choice(
                    [line[i], args.mask_token, random_token], p=sample_probs)
                output_noisy[i] = new_token
                output_masked[i] = line[i]
            out_noise_h.write(str(" ".join(output_noisy)) + "\n")
            out_mask_h.write(str(" ".join(output_masked)) + "\n")


if __name__ == "__main__":
    main()
