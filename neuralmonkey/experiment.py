"""Provides a high-level API for training and using a model."""

from argparse import Namespace  # pylint: disable=unused-import
import os
import random
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from typing import Set  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from typeguard import check_argument_types

from neuralmonkey.checking import (check_dataset_and_coders,
                                   CheckingException)
from neuralmonkey.logging import Logging, log, debug, warn
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
                     "git_commit", "git_diff", "variables.data.best"]


class Experiment:
    # pylint: disable=no-member

    _current_experiment = None

    def __init__(self,
                 config_path: str,
                 train_mode: bool = False,
                 overwrite_output_dir: bool = False,
                 config_changes: List[str] = None) -> None:
        """Initialize a Neural Monkey experiment.

        Arguments:
            config_path: The path to the experiment configuration file.
            train_mode: Indicates whether the model should be prepared for
                training.
            overwrite_output_dir: Indicates whether an existing experiment
                should be reused. If `True`, this overrides the setting in
                the configuration file.
            config_changes: A list of modifications that will be made to the
                loaded configuration file before parsing.
        """
        self.train_mode = train_mode
        self._config_path = config_path

        self.graph = tf.Graph()
        self._initializers = {}  # type: Dict[str, Callable]
        self._initialized_variables = set()  # type: Set[str]
        self.cont_index = -1
        self._model_built = False
        self._vars_loaded = False
        self._model = None  # type: Optional[Namespace]

        self.config = create_config(train_mode)
        self.config.load_file(config_path, config_changes)
        args = self.config.args

        if self.train_mode:
            # We may need to create the experiment directory.
            if (os.path.isdir(args.output)
                    and os.path.exists(
                        os.path.join(args.output, "experiment.ini"))):
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
        while any(os.path.exists(self.get_path(f, self.cont_index + 1))
                  for f in _EXPERIMENT_FILES):
            self.cont_index += 1

    @property
    def model(self) -> Namespace:
        if self._model is None:
            raise RuntimeError("Experiment argument model not initialized")

        return self._model

    def build_model(self) -> None:
        if self._model_built:
            raise RuntimeError("build_model() called twice")

        random.seed(self.config.args.random_seed)
        np.random.seed(self.config.args.random_seed)

        with self.graph.as_default():
            tf.set_random_seed(self.config.args.random_seed)

            # Enable the created model parts to find this experiment.
            type(self)._current_experiment = self  # type: ignore
            self.config.build_model(warn_unused=self.train_mode)
            type(self)._current_experiment = None

            self._model = self.config.model
            self._model_built = True

            if self.model.runners_batch_size is None:
                self.model.runners_batch_size = self.model.batch_size

            if self.model.tf_manager is None:
                self.model.tf_manager = get_default_tf_manager()

            if self.train_mode:
                check_dataset_and_coders(self.model.train_dataset,
                                         self.model.runners)
                if isinstance(self.model.val_dataset, Dataset):
                    check_dataset_and_coders(self.model.val_dataset,
                                             self.model.runners)
                else:
                    for val_dataset in self.model.val_dataset:
                        check_dataset_and_coders(val_dataset,
                                                 self.model.runners)

            if self.train_mode and self.model.visualize_embeddings:
                visualize_embeddings(self.model.visualize_embeddings,
                                     self.model.output)

        self._check_unused_initializers()

    def train(self) -> None:
        if not self.train_mode:
            raise RuntimeError("train() was called, but the experiment was "
                               "created with train_mode=False")
        if not self._model_built:
            self.build_model()

        self.cont_index += 1

        # Initialize the experiment directory.
        self.config.save_file(self.get_path("experiment.ini"))
        shutil.copyfile(self._config_path, self.get_path("original.ini"))
        save_git_info(self.get_path("git_commit"), self.get_path("git_diff"))
        Logging.set_log_file(self.get_path("experiment.log"))

        Logging.print_header(self.model.name, self.model.output)

        with self.graph.as_default():
            self.model.tf_manager.init_saving(self.get_path("variables.data"))

            training_loop(
                tf_manager=self.model.tf_manager,
                epochs=self.model.epochs,
                trainer=self.model.trainer,
                batch_size=self.model.batch_size,
                batch_bucket_span=self.model.batch_bucket_span,
                token_level_batching=self.model.token_level_batching,
                log_directory=self.model.output,
                evaluators=self.model.evaluation,
                runners=self.model.runners,
                train_dataset=self.model.train_dataset,
                val_dataset=self.model.val_dataset,
                test_datasets=self.model.test_datasets,
                logging_period=self.model.logging_period,
                validation_period=self.model.validation_period,
                val_preview_input_series=self.model.val_preview_input_series,
                val_preview_output_series=self.model.val_preview_output_series,
                val_preview_num_examples=self.model.val_preview_num_examples,
                postprocess=self.model.postprocess,
                train_start_offset=self.model.train_start_offset,
                runners_batch_size=self.model.runners_batch_size,
                initial_variables=self.model.initial_variables)

            self._vars_loaded = True

    def load_variables(self, variable_files: List[str] = None) -> None:
        if not self._model_built:
            self.build_model()

        if variable_files is None:
            if os.path.exists(self.get_path("variables.data.avg-0.index")):
                variable_files = [self.get_path("variables.data.avg-0")]
            elif os.path.exists(self.get_path("variables.data.avg.index")):
                variable_files = [self.get_path("variables.data.avg")]
            else:
                best_var_file = self.get_path("variables.data.best")
                with open(best_var_file, "r") as f_best:
                    var_path = f_best.read().rstrip()
                variable_files = [os.path.join(self.config.args.output,
                                               var_path)]

            log("Default variable file '{}' will be used for loading "
                "variables.".format(variable_files[0]))

        for vfile in variable_files:
            if not os.path.exists("{}.index".format(vfile)):
                raise RuntimeError(
                    "Index file for var prefix {} does not exist"
                    .format(vfile))

        self.model.tf_manager.restore(variable_files)
        self._vars_loaded = True

    def run_model(self,
                  dataset: Dataset,
                  write_out: bool = False,
                  batch_size: int = None,
                  log_progress: int = 0) -> Tuple[List[ExecutionResult],
                                                  Dict[str, List[Any]]]:
        """Run the model on a given dataset.

        Args:
            dataset: The dataset on which the model will be executed.
            write_out: Flag whether the outputs should be printed to a file
                defined in the dataset object.
            batch_size: size of the minibatch
            log_progress: log progress every X seconds

        Returns:
            A list of `ExecutionResult`s and a dictionary of the output series.
        """
        if not self._model_built:
            self.build_model()
        if not self._vars_loaded:
            self.load_variables()

        with self.graph.as_default():
            # TODO: check_dataset_and_coders(dataset, self.model.runners)
            return run_on_dataset(
                self.model.tf_manager, self.model.runners, dataset,
                self.model.postprocess,
                write_out=write_out, log_progress=log_progress,
                batch_size=batch_size or self.model.runners_batch_size)

    def evaluate(self,
                 dataset: Dataset,
                 write_out: bool = False,
                 batch_size: int = None,
                 log_progress: int = 0) -> Dict[str, Any]:
        """Run the model on a given dataset and evaluate the outputs.

        Args:
            dataset: The dataset on which the model will be executed.
            write_out: Flag whether the outputs should be printed to a file
                defined in the dataset object.
            batch_size: size of the minibatch
            log_progress: log progress every X seconds

        Returns:
            Dictionary of evaluation names and their values which includes the
            metrics applied on respective series loss and loss values from the
            run.
        """
        execution_results, output_data = self.run_model(
            dataset, write_out, batch_size, log_progress)

        evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                      for e in self.model.evaluation]
        with self.graph.as_default():
            eval_result = evaluation(
                evaluators, dataset, self.model.runners,
                execution_results, output_data)
        if eval_result:
            print_final_evaluation(dataset.name, eval_result)

        return eval_result

    def get_path(self, filename: str, cont_index: int = None) -> str:
        """Return the path to the most recent version of the given file."""
        if cont_index is None:
            cont_index = self.cont_index
        cont_suffix = ".cont-{}".format(cont_index) if cont_index > 0 else ""

        if filename.startswith("variables.data"):
            new_filename = "variables.data" + cont_suffix + filename[14:]
        else:
            new_filename = filename + cont_suffix

        return os.path.join(self.config.args.output, new_filename)

    def update_initializers(
            self, initializers: Iterable[Tuple[str, Callable]]) -> None:
        """Update the dictionary mapping variable names to initializers."""
        self._initializers.update(initializers)

    def get_initializer(self, var_name: str,
                        default: Callable = None) -> Optional[Callable]:
        """Return the initializer associated with the given variable name.

        Calling the method marks the given initializer as used.
        """
        initializer = self._initializers.get(var_name, default)
        if initializer is not default:
            debug("Using {} for variable {}".format(initializer, var_name))
        self._initialized_variables.add(var_name)
        return initializer

    def _check_unused_initializers(self) -> None:
        unused_initializers = [name for name in self._initializers
                               if name not in self._initialized_variables]
        if unused_initializers:
            raise CheckingException(
                "Initializers were specified for the following non-existent "
                "variables: " + ", ".join(unused_initializers))

    @classmethod
    def get_current(cls) -> "Experiment":
        """Return the experiment that is currently being built."""
        return cls._current_experiment or _DUMMY_EXPERIMENT


def create_config(train_mode: bool = True) -> Configuration:
    config = Configuration()
    config.add_argument("tf_manager", required=False, default=None)
    config.add_argument("batch_size", cond=lambda x: x > 0)
    config.add_argument("batch_bucket_span", default=-1)
    config.add_argument("token_level_batching", default=False)
    config.add_argument("output")
    config.add_argument("postprocess", required=False, default=None)
    config.add_argument("runners")
    config.add_argument("runners_batch_size", required=False, default=None)

    if train_mode:
        config.add_argument("epochs", cond=lambda x: x >= 0)
        config.add_argument("trainer")
        config.add_argument("train_dataset")
        config.add_argument("val_dataset")
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
        config.add_argument("evaluation", required=False, default=None)
        for argument in _TRAIN_ARGS:
            config.ignore_argument(argument)

    return config


class _DummyExperiment(Experiment):
    """A dummy Experiment.

    An instance of this class takes care of initializers when no other
    experiment is the current experiment. This is needed when someone creates
    a model part outside an experiment (e.g. in a unit test).
    """

    def __init__(self):
        # pylint: disable=super-init-not-called
        self._initializers = {}  # type: Dict[str, Callable]
        self._initialized_variables = set()  # type: Set[str]
        self._warned = False

    def update_initializers(
            self, initializers: Iterable[Tuple[str, Callable]]) -> None:
        self._warn()
        super().update_initializers(initializers)

    def get_initializer(self, var_name: str,
                        default: Callable = None) -> Optional[Callable]:
        """Return the initializer associated with the given variable name."""
        self._warn()
        return super().get_initializer(var_name, default)

    def _warn(self) -> None:
        if not self._warned:
            log("Warning: Creating a model part outside of an experiment.",
                color="red")
            self._warned = True


_DUMMY_EXPERIMENT = _DummyExperiment()


def save_git_info(git_commit_file: str, git_diff_file: str,
                  branch: str = "HEAD", repo_dir: str = None) -> None:
    if shutil.which("git") is not None:
        if repo_dir is None:
            # This points inside the neuralmonkey/ dir inside the repo, but
            # it does not matter for git.
            repo_dir = os.path.dirname(os.path.realpath(__file__))

        with open(git_commit_file, "wb") as file:
            subprocess.run(["git", "log", "-1", "--format=%H", branch],
                           cwd=repo_dir, stdout=file)

        with open(git_diff_file, "wb") as file:
            subprocess.run(
                ["git", "--no-pager", "diff", "--color=always", branch],
                cwd=repo_dir, stdout=file
            )
    else:
        warn("No git executable found. Not storing git commit and diffs")


def visualize_embeddings(sequences: List[EmbeddedFactorSequence],
                         output_dir: str) -> None:
    check_argument_types()

    tb_projector = projector.ProjectorConfig()

    for sequence in sequences:
        sequence.tb_embedding_visualization(output_dir, tb_projector)

    summary_writer = tf.summary.FileWriter(output_dir)
    projector.visualize_embeddings(summary_writer, tb_projector)
