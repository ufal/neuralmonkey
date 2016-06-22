#!/usr/bin/env python

import sys

from utils import log
from configuration import Configuration
from learning_utils import initialize_tf, run_on_dataset, print_dataset_evaluation
from checking import check_dataset_and_coders

CONFIG = Configuration()
CONFIG.add_argument('encoders', list, cond=lambda l: len(l) > 0)
CONFIG.add_argument('decoder')
CONFIG.add_argument('batch_size', int, cond=lambda x: x > 0)
CONFIG.add_argument('postprocess')
CONFIG.add_argument('evaluation', cond=list)
CONFIG.add_argument('runner')
CONFIG.add_argument('test_datasets', list, required=False, default=[])
CONFIG.add_argument('initial_variables', str, required=False, default=[])
CONFIG.add_argument('threads', int, required=False, default=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: run.py <run_ini_file> <test_datasets>"
        exit(1)

    test_datasets = Configuration()
    test_datasets.add_argument('test_datasets')

    args = CONFIG.load_file(sys.argv[1])
    datasets_args = test_datasets.load_file(sys.argv[2])
    print ""

    try:
        for dataset in datasets_args.test_datasets:
            check_dataset_and_coders(dataset, args.encoders)
    except Exception as exc:
        log(exc.message, color='red')
        exit(1)

    sess, _ = initialize_tf(args.initial_variables, args.threads)
    for dataset in datasets_args.test_datasets:
        _, evaluation = run_on_dataset(sess, args.runner, args.encoders + [args.decoder],
                                       args.decoder, dataset,
                                       args.evaluation, postprocess, write_out=True)
        if evaluation:
            print_dataset_evaluation(dataset.name, evaluation)
