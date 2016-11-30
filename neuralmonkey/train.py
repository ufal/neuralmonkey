"""
This is a training script for sequence to sequence learning.
"""
# tests: lint, mypy

import sys
import random
import os
from shutil import copyfile

import numpy as np
import tensorflow as tf

from neuralmonkey.checking import CheckingException, check_dataset_and_coders
from neuralmonkey.logging import Logging, log
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.learning_utils import training_loop
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_manager import TensorFlowManager

def create_config(config_file):
    config = Configuration()

    # training loop arguments
    config.add_argument('tf_manager', TensorFlowManager)
    config.add_argument('epochs', int, cond=lambda x: x >= 0)
    config.add_argument('trainer')
    config.add_argument('batch_size', int, cond=lambda x: x > 0)
    config.add_argument('train_dataset', Dataset)
    config.add_argument('val_dataset', Dataset)
    config.add_argument('output', str)
    config.add_argument('evaluation', list)
    config.add_argument('runners', list)
    config.add_argument('test_datasets', list, required=False, default=[])
    config.add_argument('save_n_best', int, required=False, default=1)
    config.add_argument('logging_period', int, required=False, default=20)
    config.add_argument('validation_period', int, required=False, default=500)
    config.add_argument('runners_batch_size', int, required=False, default=None)
    config.add_argument('minimize', bool, required=False, default=False)
    config.add_argument('postprocess')
    config.add_argument('name', str)
    config.add_argument('initial_variables', str, required=False, default=[])
    config.add_argument('overwrite_output_dir', bool, required=False,
                        default=False)

    return config.load_file(config_file)

def main():
    if len(sys.argv) != 2:
        print("Usage: train.py <ini_file>")
        exit(1)

    # random seeds have to be set before anything is created in the graph
    random.seed(2574600)
    np.random.seed(2574600)
    tf.set_random_seed(2574600)

    args = create_config(sys.argv[1])

    print("")

    #pylint: disable=no-member
    if os.path.isdir(args.output) and \
            os.path.exists(os.path.join(args.output, "experiment.ini")):
        if args.overwrite_output_dir:
            # we do not want to delete the directory contents
            log("Directory with experiment.ini '{}' exists, "
                "overwriting enabled, proceeding."
                .format(args.output))
        else:
            log("Directory with experiment.ini '{}' exists, "
                "overwriting disabled."
                .format(args.output), color='red')
            exit(1)

    try:
        check_dataset_and_coders(args.train_dataset,
                                 args.runners)
        check_dataset_and_coders(args.val_dataset,
                                 args.runners)
    except CheckingException as exc:
        log(str(exc), color='red')
        exit(1)

    #pylint: disable=broad-except
    if not os.path.isdir(args.output):
        try:
            os.mkdir(args.output)
        except Exception as exc:
            log("Failed to create experiment directory: {}. Exception: {}"
                .format(args.output, exc), color='red')
            exit(1)

    log_file = "{}/experiment.log".format(args.output)
    ini_file = "{}/experiment.ini".format(args.output)
    git_commit_file = "{}/git_commit".format(args.output)
    git_diff_file = "{}/git_diff".format(args.output)
    variables_file_prefix = "{}/variables.data".format(args.output)

    cont_index = 0

    while (os.path.exists(log_file)
           or os.path.exists(ini_file)
           or os.path.exists(git_commit_file)
           or os.path.exists(git_diff_file)
           or os.path.exists(variables_file_prefix)
           or os.path.exists("{}.0".format(variables_file_prefix))):
        cont_index += 1

        log_file = "{}/experiment.log.cont-{}".format(args.output, cont_index)
        ini_file = "{}/experiment.ini.cont-{}".format(args.output, cont_index)
        git_commit_file = "{}/git_commit.cont-{}".format(
            args.output, cont_index)
        git_diff_file = "{}/git_diff.cont-{}".format(args.output, cont_index)
        variables_file_prefix = "{}/variables.data.cont-{}".format(
            args.output, cont_index)

    copyfile(sys.argv[1], ini_file)
    Logging.set_log_file(log_file)
    Logging.print_header(args.name)

    # this points inside the neuralmonkey/ dir inside the repo, but
    # it does not matter for git.
    repodir = os.path.dirname(os.path.realpath(__file__))

    os.system("cd {}; git log -1 --format=%H > {}"
              .format(repodir, git_commit_file))

    os.system("cd {}; git --no-pager diff --color=always > {}"
              .format(repodir, git_diff_file))

    link_best_vars = "{}.best".format(variables_file_prefix)

    training_loop(tf_manager=args.tf_manager,
                  epochs=args.epochs,
                  trainer=args.trainer,
                  batch_size=args.batch_size,
                  train_dataset=args.train_dataset,
                  val_dataset=args.val_dataset,
                  log_directory=args.output,
                  evaluators=args.evaluation,
                  runners=args.runners,
                  test_datasets=args.test_datasets,
                  save_n_best_vars=args.save_n_best,
                  link_best_vars=link_best_vars,
                  vars_prefix=variables_file_prefix,
                  logging_period=args.logging_period,
                  validation_period=args.validation_period,
                  postprocess=args.postprocess,
                  minimize_metric=args.minimize)
