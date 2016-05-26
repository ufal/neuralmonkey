#!/usr/bin/env python

import sys
import codecs

from utils import log
from configuration import Configuration
from learning_utils import initialize_tf, run_on_dataset, print_dataset_evaluation
from dataset import Dataset

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: run.py <run_ini_file> <test_datasets>"
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

    test_datasets = Configuration()
    test_datasets.add_argument('test_datasets')

    args = config.load_file(sys.argv[1])
    datasets_args = test_datasets.load_file(sys.argv[2])
    print ""

    sess, _ = initialize_tf(args.initial_variables)
    for dataset in datasets_args.test_datasets:
        _, evaluation = run_on_dataset(sess, args.runner, args.encoders + [args.decoder],
                                       args.decoder, dataset,
                                       args.evaluation, write_out=True)
        if evaluation:
            print_dataset_evaluation(dataset.name, evaluation)

