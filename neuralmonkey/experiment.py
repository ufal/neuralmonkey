import os
import random
from shutil import copyfile
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from typeguard import check_argument_types

from neuralmonkey.checking import (check_dataset_and_coders,
                                   check_unused_initializers)
from neuralmonkey.logging import Logging, log
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.learning_utils import (training_loop, evaluation,
                                         run_on_dataset,
                                         print_final_evaluation)
from neuralmonkey.dataset import Dataset
from neuralmonkey.model.sequence import EmbeddedFactorSequence
from neuralmonkey.runners.base_runner import ExecutionResult
from neuralmonkey.tf_manager import get_default_tf_manager


_TRAIN_ARGS = [
    "val_dataset", "trainer", "name", "train_dataset", "epochs",
    "test_datasets", "initial_variables", "validation_period",
    "val_preview_input_series", "val_preview_output_series",
    "val_preview_num_examples", "logging_period", "visualize_embeddings",
    "random_seed", "overwrite_output_dir"
]


_EXPERIMENT_FILES = ["experiment.log", "experiment.ini", "original.ini",
                     "git_commit", "git_diff", "variables.data"]


def create_config(train_mode: bool = True) -> Configuration:
    config = Configuration()
    config.add_argument("tf_manager", required=False, default=None)
    config.add_argument("batch_size", cond=lambda x: x > 0)
    config.add_argument("output")
    config.add_argument("runners")
    config.add_argument("runners_batch_size", required=False, default=None)

    if train_mode:
        config.add_argument("epochs", cond=lambda x: x >= 0)
        config.add_argument("trainer")
        config.add_argument("train_dataset")
        config.add_argument("val_dataset")
        config.add_argument("postprocess")
        config.add_argument("evaluation")
        config.add_argument("test_datasets", required=False, default=[])
        config.add_argument("logging_period", required=False, default=20)
        config.add_argument("validation_period", required=False, default=500)
        config.add_argument("visualize_embeddings", required=False,
                            default=None)
        config.add_argument("val_preview_input_series",
                            required=False, default=None)
        config.add_argument("val_preview_output_series",
                            required=False, default=None)
        config.add_argument("val_preview_num_examples",
                            required=False, default=15)
        config.add_argument("train_start_offset", required=False, default=0)
        config.add_argument("name", required=False,
                            default="Neural Monkey Experiment")
        config.add_argument("random_seed", required=False, default=2574600)
        config.add_argument("initial_variables", required=False, default=None)
        config.add_argument("overwrite_output_dir", required=False,
                            default=False)
    else:
        config.add_argument("postprocess", required=False, default=None)
        config.add_argument("evaluation", required=False, default=None)
        for argument in _TRAIN_ARGS:
            config.ignore_argument(argument)

    return config


def save_git_info(git_commit_file: str, git_diff_file: str,
                  branch: str = "HEAD", repo_dir: str = None) -> None:
    if repo_dir is None:
        # This points inside the neuralmonkey/ dir inside the repo, but
        # it does not matter for git.
        repo_dir = os.path.dirname(os.path.realpath(__file__))

    with open(git_commit_file, "wb") as file:
        subprocess.run(["git", "log", "-1", "--format=%H", branch],
                       cwd=repo_dir, stdout=file)

    with open(git_diff_file, "wb") as file:
        subprocess.run(["git", "--no-pager", "diff", "--color=always", branch],
                       cwd=repo_dir, stdout=file)


def visualize_embeddings(sequences: List[EmbeddedFactorSequence],
                         output_dir: str) -> None:
    check_argument_types()

    tb_projector = projector.ProjectorConfig()

    for sequence in sequences:
        sequence.tb_embedding_visualization(output_dir, tb_projector)

    summary_writer = tf.summary.FileWriter(output_dir)
    projector.visualize_embeddings(summary_writer, tb_projector)


class Experiment(object):
    # pylint: disable=no-member

    def __init__(self,
                 config_path: str,
                 config_changes: List[str] = None,
                 train_mode: bool = True,
                 overwrite_output_dir: bool = False) -> None:
        self.train_mode = train_mode
        self._config_path = config_path

        self.graph = tf.Graph()
        self.cont_index = None
        self._model_built = False
        self._vars_loaded = False

        self.config = create_config(train_mode)
        self.config.load_file(config_path, config_changes)
        args = self.config.args

        if self.train_mode:
            # We may need to create the experiment directory.
            if (os.path.isdir(args.output) and
                    os.path.exists(os.path.join(args.output,
                                                "experiment.ini"))):
                if args.overwrite_output_dir or overwrite_output_dir:
                    # we do not want to delete the directory contents
                    log("Directory with experiment.ini '{}' exists, "
                        "overwriting enabled, proceeding.".format(args.output))
                else:
                    raise RuntimeError(
                        "Directory with experiment.ini '{}' exists, "
                        "overwriting disabled.".format(args.output))

            if not os.path.isdir(args.output):
                os.mkdir(args.output)

        # Find how many times the experiment has been continued.
        self.cont_index = -1
        while any(os.path.exists(self.get_path(f, cont_index + 1))
                  for f in _EXPERIMENT_FILES):
            self.cont_index += 1

    def build_model(self) -> None:
        if self._model_built:
            raise RuntimeError("build_model() called twice")

        random.seed(self.config.args.random_seed)
        np.random.seed(self.config.args.random_seed)

        with self.graph.as_default():
            self.config.build_model(warn_unused=self.train_mode)
            model = self.config.model
            self._model_built = True

            if model.runners_batch_size is None:
                model.runners_batch_size = model.batch_size

            if model.tf_manager is None:
                model.tf_manager = get_default_tf_manager()

            if self.train_mode:
                check_dataset_and_coders(model.train_dataset, model.runners)
                if isinstance(model.val_dataset, Dataset):
                    check_dataset_and_coders(model.val_dataset, model.runners)
                else:
                    for val_dataset in model.val_dataset:
                        check_dataset_and_coders(val_dataset, model.runners)

            check_unused_initializers()

            if self.train_mode and model.visualize_embeddings:
                visualize_embeddings(model.visualize_embeddings, model.output)

    def train(self) -> None:
        if not self.train_mode:
            raise RuntimeError("train() was called, but the experiment was"
                               "not created in training mode")
        if not self._model_built:
            self.build_model()

        self.cont_index += 1

        # Initialize the experiment directory.
        self.config.save_file(self.get_path("experiment.ini"))
        copyfile(self._config_path, self.get_path("original.ini"))
        save_git_info(self.get_path("git_commit"), self.get_path("git_diff"))
        Logging.set_log_file(self.get_path("experiment.log"))

        Logging.print_header(self.config.model.name, self.config.args.output)

        with self.graph.as_default():
            tf.set_random_seed(self.config.args.random_seed)

            model = self.config.model

            model.tf_manager.init_saving(self.get_path("variables.data"))

            training_loop(
                tf_manager=model.tf_manager,
                epochs=model.epochs,
                trainer=model.trainer,
                batch_size=model.batch_size,
                log_directory=model.output,
                evaluators=model.evaluation,
                runners=model.runners,
                train_dataset=model.train_dataset,
                val_dataset=model.val_dataset,
                test_datasets=model.test_datasets,
                logging_period=model.logging_period,
                validation_period=model.validation_period,
                val_preview_input_series=model.val_preview_input_series,
                val_preview_output_series=model.val_preview_output_series,
                val_preview_num_examples=model.val_preview_num_examples,
                postprocess=model.postprocess,
                train_start_offset=model.train_start_offset,
                runners_batch_size=model.runners_batch_size,
                initial_variables=model.initial_variables)

    def load_variables(self, variable_files: List[str] = None):
        if not self._model_built:
            raise RuntimeError("load_variables() called before build_model()")

        if variable_files is None:
            variable_files = [self.get_path("variables.data")]

            log("Default variable file '{}' will be used for loading "
                "variables.".format(variable_files[0]))

        for vfile in variable_files:
            if not os.path.exists("{}.index".format(vfile)):
                raise RuntimeError(
                    "Index file for var prefix {} does not exist"
                    .format(vfile))

        with self.graph.as_default():
            self.config.model.tf_manager.restore(variable_files)
        self._vars_loaded = True

    def run_model(self,
                  dataset: Dataset,
                  write_out: bool = False,
                  batch_size: Optional[int] = None,
                  log_progress: int = 0) -> Tuple[List[ExecutionResult],
                                                  Dict[str, List[Any]]]:
        if not self._model_built:
            self.build_model()
        if not self._vars_loaded:
            self.load_variables()

        model = self.config.model
        with self.graph.as_default():
            # TODO: check_dataset_and_coders(dataset, model.runners)
            return run_on_dataset(
                model.tf_manager, model.runners, dataset, model.postprocess,
                write_out=write_out, log_progress=log_progress,
                batch_size=batch_size or model.runners_batch_size)

    def evaluate(self,
                 dataset: Dataset,
                 write_out: bool = False,
                 batch_size: Optional[int] = None,
                 log_progress: int = 0):
        execution_results, output_data = self.run_model(
            dataset, write_out, batch_size, log_progress)

        evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                      for e in self.config.model.evaluation]
        with self.graph.as_default():
            eval_result = evaluation(
                evaluators, dataset, self.config.model.runners,
                execution_results, output_data)
        if eval_result and write_out:
            print_final_evaluation(dataset.name, eval_result)

        return eval_result

    def get_path(self, filename: str, cont_index: int = None) -> str:
        if cont_index is None:
            cont_index = self.cont_index
        cont_suffix = ".cont-{}".format(cont_index) if cont_index > 0 else ""
        return os.path.join(self.config.args.output, filename + cont_suffix)
