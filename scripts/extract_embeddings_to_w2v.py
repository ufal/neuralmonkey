#!/usr/bin/env python3

"""Extract embeddings to Word2Vec format.

This script loads a checkpoint (only variables without the model) and extract
embeddings from a given model part. Note that for model imported from Nematus,
the length of the vocabulary JSON and embeddings might differ. In this case,
turn off the check whether the embeddings and vocabulary have the same length.
"""

import argparse
import os
import sys

import tensorflow as tf

from neuralmonkey.logging import log as _log
from neuralmonkey.vocabulary import (
    from_wordlist, from_nematus_json, from_t2t_vocabulary)


POSSIBLE_EMBEDDINGS_NAMES = [
    "word_embeddings", "embedding_matrix_0",
    "input_projection/word_embeddings"]


def log(message: str, color: str = "blue") -> None:
    _log(message, color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_checkpoint", metavar="MODEL-CHECKPOINT",
        help="Path to the model checkpoint.")
    parser.add_argument(
        "model_part_name", metavar="MODEL-PART",
        help="Name of model part with embeddings.")
    parser.add_argument(
        "vocabulary", metavar="VOCABULARY", help="Vocabulary file.")
    parser.add_argument(
        "--output-file", metavar="OUTPUT", default=sys.stdout,
        type=argparse.FileType('w'), required=False,
        help="Output file in Word2Vec format.")
    parser.add_argument(
        "--vocabulary-format", type=str,
        choices=["tsv", "word_list", "nematus_json", "t2t_vocabulary"],
        default="tsv",
        help="Vocabulary format (see functions in the vocabulary module).")
    parser.add_argument(
        "--validate-length", type=bool, default=True,
        help=("Check if the vocabulary and the embeddings have the "
              "same length."))
    args = parser.parse_args()

    if args.vocabulary_format == "word_list":
        vocabulary = from_wordlist(
            args.vocabulary, contains_header=False, contains_frequencies=False)
    elif args.vocabulary_format == "tsv":
        vocabulary = from_wordlist(
            args.vocabulary, contains_header=True, contains_frequencies=True)
    elif args.vocabulary_format == "nematus_json":
        vocabulary = from_nematus_json(args.vocabulary)
    elif args.vocabulary_format == "t2t_vocabulary":
        vocabulary = from_t2t_vocabulary(args.vocabulary)
    else:
        raise ValueError("Unknown type of vocabulary file: {}".format(
            args.vocabulary_format))

    if not os.path.exists("{}.index".format(args.model_checkpoint)):
        log("Checkpoint '{}' does not exist.".format(
            args.model_checkpoint), color="red")
        exit(1)

    embeddings_name = None
    for model_part in [args.model_part_name,
                       "{}_input".format(args.model_part_name)]:
        log("Getting list of variables in '{}'.".format(model_part))
        var_list = [
            name for name, shape in
            tf.contrib.framework.list_variables(args.model_checkpoint)
            if name.startswith("{}/".format(model_part))]

        for name in POSSIBLE_EMBEDDINGS_NAMES:
            candidate_name = "{}/{}".format(model_part, name)
            if candidate_name in var_list:
                embeddings_name = candidate_name
                break

    if embeddings_name is None:
        log("No embeddings found in the model part.", color="red")
        exit(1)

    reader = tf.contrib.framework.load_checkpoint(args.model_checkpoint)
    embeddings = reader.get_tensor(embeddings_name)

    word_count, dimension = embeddings.shape

    if word_count != len(vocabulary):
        if args.validate_length:
            log(("Vocabulary has length of {}, but there are {} "
                 "embeddings.").format(len(vocabulary), word_count),
                color="red")
            exit(1)
        else:
            word_count = min(word_count, len(vocabulary))

    print("{}\t{}".format(word_count, dimension), file=args.output_file)
    for word, vector in zip(vocabulary.index_to_word, embeddings):
        formatted_vector = "\t".join(["{:.8f}".format(x) for x in vector])
        print("{}\t{}".format(word, formatted_vector), file=args.output_file)

    log("Done")


if __name__ == "__main__":
    main()
