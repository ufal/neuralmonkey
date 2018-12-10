"""Provides a high-level API for training and using a model."""
# pylint: disable=too-many-lines

from argparse import Namespace
import os
import random
import shutil
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from typing import Set  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from neuralmonkey.checking import CheckingException
from neuralmonkey.dataset import BatchingScheme, Dataset
from neuralmonkey.logging import Logging, log, debug, warn
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.config.normalize import normalize_configuration
from neuralmonkey.learning_utils import (training_loop, evaluation,
                                         run_on_dataset,
                                         print_final_evaluation)
from neuralmonkey.runners.base_runner import ExecutionResult
from neuralmonkey.runners.dataset_runner import DatasetRunner


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
        """Get configuration namespace of the experiment.

        The `Experiment` stores the configuration recipe in `self.config`.
        When the configuration is built (meaning the classes referenced from
        the config file are instantiated), it is saved in the `model` property
        of the experiment.

        Returns:
            The built namespace config object.

        Raises:
            `RuntimeError` when the configuration model has not been built.
        """
        if self._model is None:
            raise RuntimeError("Experiment argument model not initialized")

        return self._model

    def _bless_graph_executors(self) -> None:
        """Pre-compute the tensors referenced by the graph executors.

        Due to the lazy nature of the computational graph related components,
        nothing is actually added to the graph until it is "blessed" (
        referenced, and therefore, executed).

        "Blessing" is usually implemented in the form of a log or a debug call
        with the blessed tensor as parameter. Referencing a `Tensor` causes the
        whole computational graph that is needed to evaluate the tensor to be
        built.

        This function "blesses" all tensors that could be potentially used
        using the `fetches` property of the provided runner objects.

        If the experiment runs in the training mode, this function also
        blesses the tensors fetched by the trainer(s).
        """
        log("Building TF Graph")
        if hasattr(self.model, "trainer"):
            if isinstance(self.model.trainer, List):
                trainers = self.model.trainer
            else:
                trainers = [self.model.trainer]

            for trainer in trainers:
                debug("Trainer fetches: {}".format(trainer.fetches), "bless")

        for runner in self.model.runners:
            debug("Runner fetches: {}".format(runner.fetches), "bless")
        log("TF Graph built")

    def register_inputs(self) -> None:
        feedables = set.union(*[ex.feedables for ex in self.model.runners])
        if self.train_mode:
            feedables |= set.union(
                *[ex.feedables for ex in self.model.trainers])

        for feedable in feedables:
            feedable.register_input()

        self.model.dataset_runner.register_input()

    def build_model(self) -> None:
        """Build the configuration and the computational graph.

        This function is invoked by all of the main entrypoints of the
        `Experiment` class (`train`, `evaluate`, `run`). It manages the
        building of the TensorFlow graph.

        The bulding procedure is executed as follows:
        1. Random seeds are set.
        2. Configuration is built (instantiated) and normalized.
        3. TODO(tf-data) tf.data.Dataset instance is created and registered
            in the model parts. (This is not implemented yet!)
        4. Graph executors are "blessed". This causes the rest of the TF Graph
            to be built.
        5. Sessions are initialized using the TF Manager object.

        Raises:
            `RuntimeError` when the model is already built.
        """
        if self._model_built:
            raise RuntimeError("build_model() called twice")

        random.seed(self.config.args.random_seed)
        np.random.seed(self.config.args.random_seed)

        with self.graph.as_default():
            tf.set_random_seed(self.config.args.random_seed)

            # Enable the created model parts to find this experiment.
            type(self)._current_experiment = self  # type: ignore

            self.config.build_model(warn_unused=self.train_mode)
            normalize_configuration(self.config.model, self.train_mode)

            self._model = self.config.model
            self._model_built = True

            # prepare dataset runner
            self.model.dataset_runner = DatasetRunner()

            # build dataset
            self.register_inputs()

            self._bless_graph_executors()
            self.model.tf_manager.initialize_sessions()

            type(self)._current_experiment = None

            if self.train_mode and self.model.visualize_embeddings is not None:
                self.visualize_embeddings()

        self._check_unused_initializers()

    def train(self) -> None:
        """Train model specified by this experiment.

        This function is one of the main functions (entrypoints) called on
        the experiment. It builds the model (if needed) and runs the training
        procedure.

        Raises:
            `RuntimeError` when the experiment is not intended for training.
        """
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

            training_loop(cfg=self.model)

            final_variables = self.get_path("variables.data.final")
            log("Saving final variables in {}".format(final_variables))
            self.model.tf_manager.save(final_variables)

            if self.model.test_datasets:
                if self.model.tf_manager.best_score_index is not None:
                    self.model.tf_manager.restore_best_vars()

                runner_batch = self.model.runners_batching_scheme.batch_size

                for test_id, dataset in enumerate(self.model.test_datasets):
                    self.evaluate(dataset, write_out=True,
                                  batch_size=runner_batch,
                                  name="test_{}".format(test_id))

            log("Finished.")
            self._vars_loaded = True

    def load_variables(self, variable_files: List[str] = None) -> None:
        """Load variables of the built model from file(s).

        When variable files are not provided, Neural Monkey will try to infer
        the name of a default checkpoint file using the following key:
        1. Look for the averaged checkpoints named `variables.data.avg` or
           `variables.data.avg-0`.
        2. Look for file `variables.data.best` file which usually contains the
           best scoring checkpoint from the run.
        3. Look for the final checkpoint saved in `variables.data.final`.

        Arguments:
            variable_files: A list of variable files to load. The length of
                this list should match the number of sessions.
        """
        if not self._model_built:
            self.build_model()

        if variable_files is None:
            if os.path.exists(self.get_path("variables.data.avg-0.index")):
                variable_files = [self.get_path("variables.data.avg-0")]
            elif os.path.exists(self.get_path("variables.data.avg.index")):
                variable_files = [self.get_path("variables.data.avg")]
            elif os.path.exists(self.get_path("variables.data.best")):
                best_var_file = self.get_path("variables.data.best")
                with open(best_var_file, "r") as f_best:
                    var_path = f_best.read().rstrip()
                variable_files = [os.path.join(self.config.args.output,
                                               var_path)]
            elif os.path.exists(self.get_path("variables.data.final.index")):
                variable_files = [self.get_path("variables.data.final")]
            else:
                raise RuntimeError("Cannot infer default variables file")

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
                  log_progress: int = 0) -> Tuple[
                      List[ExecutionResult], Dict[str, List], Dict[str, List]]:
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

        toklevel = self.model.runners_batching_scheme.token_level_batching
        assert self.model.runners_batching_scheme.batch_bucket_span is None

        batching_scheme = BatchingScheme(
            batch_size=batch_size or self.model.runners_batch_size,
            batch_bucket_span=None,
            token_level_batching=toklevel,
            bucketing_ignore_series=[])

        with self.graph.as_default():
            return run_on_dataset(
                self.model.tf_manager,
                self.model.runners,
                self.model.dataset_runner,
                dataset,
                self.model.postprocess,
                write_out=write_out,
                log_progress=log_progress,
                batching_scheme=batching_scheme)

    def evaluate(self,
                 dataset: Dataset,
                 write_out: bool = False,
                 batch_size: int = None,
                 log_progress: int = 0,
                 name: str = None) -> Dict[str, Any]:
        """Run the model on a given dataset and evaluate the outputs.

        Args:
            dataset: The dataset on which the model will be executed.
            write_out: Flag whether the outputs should be printed to a file
                defined in the dataset object.
            batch_size: size of the minibatch
            log_progress: log progress every X seconds
            name: The name of the evaluated dataset

        Returns:
            Dictionary of evaluation names and their values which includes the
            metrics applied on respective series loss and loss values from the
            run.
        """
        execution_results, output_data, f_dataset = self.run_model(
            dataset, write_out, batch_size, log_progress)

        evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                      for e in self.model.evaluation]
        with self.graph.as_default():
            eval_result = evaluation(
                evaluators, f_dataset, execution_results, output_data)
        if eval_result:
            print_final_evaluation(eval_result, name)

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

    def visualize_embeddings(self) -> None:
        """Visualize embeddings of sequences in `main.visualize_embeddings`."""
        tb_projector = projector.ProjectorConfig()

        for sequence in self.model.visualize_embeddings:
            for i, (vocabulary, emb_matrix) in enumerate(
                    zip(sequence.vocabularies, sequence.embedding_matrices)):

                # TODO when vocabularies will have name parameter, change it
                path = self.get_path("seq.{}-{}.tsv".format(sequence.name, i))
                vocabulary.save_wordlist(path)

                embedding = tb_projector.embeddings.add()
                # pylint: disable=unsubscriptable-object
                embedding.tensor_name = emb_matrix.name
                embedding.metadata_path = path
                # pylint: enable=unsubscriptable-object

        summary_writer = tf.summary.FileWriter(self.model.output)
        projector.visualize_embeddings(summary_writer, tb_projector)

    @classmethod
    def get_current(cls) -> "Experiment":
        """Return the experiment that is currently being built."""
        return cls._current_experiment or _DUMMY_EXPERIMENT


def create_config(train_mode: bool = True) -> Configuration:
    config = Configuration()
    config.add_argument("tf_manager", required=False, default=None)
    config.add_argument("batch_size", required=False, default=None,
                        cond=lambda x: x is None or x > 0)
    config.add_argument("batching_scheme", required=False, default=None)
    config.add_argument("output")
    config.add_argument("postprocess", required=False, default=None)
    config.add_argument("runners")
    config.add_argument("runners_batch_size", required=False, default=None)

    if train_mode:
        config.add_argument("epochs", cond=lambda x: x >= 0)
        config.add_argument("trainer")
        config.add_argument("train_dataset")
        config.add_argument("val_dataset", required=False, default=[])
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
