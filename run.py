#!/usr/bin/env python

import sys
import codecs

import tensorflow as tf

from utils import print_header, log
from configuration import Configuration
from learning_utils import initialize_tf, run_on_dataset
from dataset import Dataset

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: train.py <ini_file>"
        exit(1)

    config = Configuration()
    config.add_argument('encoders', list, cond=lambda l: len(l) > 0)
    config.add_argument('decoder')
    config.add_argument('batch_size', int, cond=lambda x: x > 0)
    config.add_argument('postprocess')
    config.add_argument('evaluation', cond=list)
    config.add_argument('runner')
    config.add_argument('test_datasets', list, required=False, default=[])
    config.add_argument('initial_variables', str, required=False, default=[])

    try:
        ini_file = sys.argv[1]
        log("Loading ini file: \"{}\"".format(ini_file), color='cyan')
        config_f = codecs.open(ini_file, 'r', 'utf-8')
        args = config.load_file(config_f)
        log("ini file loded.", color='cyan')
    except Exception as e:
        log(e.message, color='red')
        exit(1)

    print ""

    sess, _ = initialize_tf(args.initial_variables)
    for dataset in args.test_datasets:
        _, evaluation = run_on_dataset(sess, args.runner, args.encoders + [args.decoder],
                                       args.decoder, dataset,
                                       args.evaluation, write_out=True)
