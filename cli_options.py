#!/usr/bin/env python

import argparse

def add_common_cli_arguments(parser):
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--embeddings-size", type=int, default=256)
    parser.add_argument("--encoder-rnn-size", type=int, default=256)
    parser.add_argument("--decoder-rnn-size", type=int, default=256)

    parser.add_argument("--maximum-output", type=int, default=20)
    parser.add_argument("--dropout-keep-prob", type=float, default=1.0)
    parser.add_argument("--l2-regularization", type=float, default=0.0)

    parser.add_argument("--use-attention", type=bool, default=False)
    parser.add_argument("--scheduled-sampling", type=float, default=None)
    parser.add_argument("--use-noisy-activations", type=bool, default=False)

    parser.add_argument("--beamsearch", type=bool, default=False)
    parser.add_argument("--target-german", type=bool, default=False)

    parser.add_argument("--initial-variables", type=str, default=None,
            help="File with saved variables for initialization.")

    #parser.add_argument("--character-based", type=bool, default=False)
    # TODO

    return parser



def add_captioning_arguments(parser):
    def shape(string):
        res_shape = [int(s) for s in string.split("x")]
        return res_shape

    parser.add_argument("--train-images", type=argparse.FileType('rb'),
                        help="File with training images features", required=True)
    parser.add_argument("--val-images", type=argparse.FileType('rb'),
                        help="File with validation images features.", required=True)


    parser.add_argument("--tokenized-train-text", type=argparse.FileType('r'),
                        help="File with tokenized training target sentences.", required=True)
    parser.add_argument("--tokenized-val-text", type=argparse.FileType('r'), required=True)


    parser.add_argument("--img-features-shape", type=shape, default='14x14x256', required=True)

    return parser



def add_translation_arguments(parser):
    parser.add_argument("--train-source-sentences", type=argparse.FileType('r'),
                        help="File with training source sentences", required=True)
    parser.add_argument("--train-target-sentences", type=argparse.FileType('r'),
                        help="File with training target sentences.", required=True)


    parser.add_argument("--val-source-sentences", type=argparse.FileType('r'),
                        help="File with validation source sentences.", required=True)
    parser.add_argument("--val-target-sentences", type=argparse.FileType('r'),
                        help="File with validation target sentences.", required=True)

    def mixer_values(string):
        values = [int(s) for s in string.split(",")]
        assert(len(values) == 2)
        return values

    parser.add_argument("--mixer", type=mixer_values, default=None)
    parser.add_argument("--gru-bidi-depth", type=int, default=None)

    return parser



def add_postediting_arguments(parser):
    parser.add_argument("--train-source-sentences", type=argparse.FileType('r'),
                        help="File with training source sentences", required=True)
    parser.add_argument("--train-translated-sentences", type=argparse.FileType('r'),
                        help="File with training source sentences", required=True)
    parser.add_argument("--train-target-sentences", type=argparse.FileType('r'),
                        help="File with training target sentences.", required=True)


    parser.add_argument("--val-source-sentences", type=argparse.FileType('r'),
                        help="File with validation source sentences.", required=True)
    parser.add_argument("--val-translated-sentences", type=argparse.FileType('r'),
                        help="File with validation source sentences.", required=True)
    parser.add_argument("--val-target-sentences", type=argparse.FileType('r'),
                        help="File with validation target sentences.", required=True)


    parser.add_argument("--test-source-sentences", type=argparse.FileType('r'),
                        help="File with tokenized test source sentences.")
    parser.add_argument("--test-translated-sentences", type=argparse.FileType('r'),
                        help="File with tokenized test translated sentences.")
    parser.add_argument("--test-output-file", type=argparse.FileType('w'),
                        help="Output file for translated test set")


    parser.add_argument("--use-copy-net", type=bool, default=False)

    parser.add_argument("--shared-embeddings", type=bool, default=False,
                        help="Share word embeddings between encoders of the same language")

    return parser



def get_postediting_parser():
    parser = get_parser('Trains the postediting')
    add_postediting_arguments(parser)
    return parser

def get_translation_parser():
    parser = get_parser('Trains the translation')
    add_translation_arguments(parser)
    return parser


def get_captioning_parser():
    parser = get_parser('Trains the image captioning')
    add_captioning_arguments(parser)
    return parser



def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    add_common_cli_arguments(parser)
    return parser
