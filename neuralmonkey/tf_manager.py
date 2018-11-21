"""TensorFlow Manager.

TensorFlow manager is a helper object in Neural Monkey which manages TensorFlow
sessions, execution of the computation graph, and saving and restoring of model
variables.

"""
# pylint: disable=unused-import
from typing import Any, List, Union, Optional, Set, Sequence
# pylint: enable=unused-import

import os

import numpy as np
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python import debug as tf_debug
# pylint: enable=no-name-in-module
from typeguard import check_argument_types

from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.model.feedable import Feedable
# pylint: disable=unused-import
from neuralmonkey.runners.base_runner import FeedDict
# pylint: enable=unused-import
from neuralmonkey.runners.base_runner import (
    BaseRunner, ExecutionResult, Executable)
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer

# pylint: disable=invalid-name
Trainer = Union[GenericTrainer, MultitaskTrainer]
# pylint: enable=invalid-name


class TensorFlowManager:
    """Inteface between computational graph, data and TF sessions.

    Attributes:
        sessions: List of active Tensorflow sessions.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_sessions: int,
                 num_threads: int,
                 save_n_best: int = 1,
                 minimize_metric: bool = False,
                 variable_files: Optional[List[str]] = None,
                 gpu_allow_growth: bool = True,
                 per_process_gpu_memory_fraction: float = 1.0,
                 enable_tf_debug: bool = False) -> None:
        """Initialize a TensorflowManager.

        At this moment the graph must already exist. This method initializes
        required number of TensorFlow sessions and initializes them with
        provided variable files if they are provided.

        Args:
            num_sessions: Number of sessions to be initialized.
            num_threads: Number of threads sessions will run in.
            save_n_best: How many best models to keep
            minimize_metric: Whether the best model is the one with the lowest
                or the highest score
            variable_files: List of variable files.
            gpu_allow_growth: TF to allocate incrementally, not all at once.
            per_process_gpu_memory_fraction: Limit TF memory use.
        """
        check_argument_types()

        session_cfg = tf.ConfigProto()
        session_cfg.inter_op_parallelism_threads = num_threads
        session_cfg.intra_op_parallelism_threads = num_threads
        session_cfg.allow_soft_placement = True  # needed for multiple GPUs
        # pylint: disable=no-member
        session_cfg.gpu_options.allow_growth = gpu_allow_growth
        session_cfg.gpu_options.per_process_gpu_memory_fraction = \
            per_process_gpu_memory_fraction
        # pylint: enable=no-member

        if save_n_best < 1:
            raise Exception("save_n_best parameter must be greater than zero")
        self.saver_max_to_keep = save_n_best
        self.minimize_metric = minimize_metric

        self.sessions = [tf.Session(config=session_cfg)
                         for _ in range(num_sessions)]

        if enable_tf_debug:
            self.sessions = [tf_debug.LocalCLIDebugWrapperSession(sess)
                             for sess in self.sessions]

        init_op = tf.global_variables_initializer()
        for sess in self.sessions:
            sess.run(init_op)
        self.saver = tf.train.Saver(max_to_keep=None,
                                    var_list=[g for g in tf.global_variables()
                                              if "reward_" not in g.name])

        if variable_files:
            if len(variable_files) != num_sessions:
                raise Exception(("The number of provided variable files ({}) "
                                 "is different than a number sessions ({})")
                                .format(len(variable_files), num_sessions))
            self.restore(variable_files)

        self.best_score_index = 0
        self.best_score_epoch = 0
        self.best_score_batch = 0

        init_score = np.inf if self.minimize_metric else -np.inf
        self.saved_scores = [init_score for _ in range(self.saver_max_to_keep)]
        self.best_score = init_score

        self.variables_files = []  # type: List[str]
        self._best_vars_file = None  # type: Optional[str]
    # pylint: enable=too-many-arguments

    @property
    def best_vars_file(self) -> str:
        if self._best_vars_file is None:
            raise RuntimeError("Saving not initialized yet.")

        return self._best_vars_file

    def _is_better(self, score1: float, score2: float) -> bool:
        if self.minimize_metric:
            return score1 < score2

        return score1 > score2

    def _argworst(self, scores: List[float]) -> int:
        if self.minimize_metric:
            return np.argmax(scores)

        return np.argmin(scores)

    def _update_best_vars(self, var_index: int) -> None:
        best_vars_prefix = os.path.basename(self.variables_files[var_index])

        with open(self.best_vars_file, "w") as var_file:
            var_file.write(best_vars_prefix)

    def init_saving(self, vars_prefix: str) -> None:
        if self.saver_max_to_keep == 1:
            self.variables_files = [vars_prefix]
        else:
            self.variables_files = ["{}.{}".format(vars_prefix, i)
                                    for i in range(self.saver_max_to_keep)]

        self._best_vars_file = "{}.best".format(vars_prefix)
        self._update_best_vars(var_index=0)

    def validation_hook(self, score: float, epoch: int, batch: int) -> None:
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.best_score_epoch = epoch
            self.best_score_batch = batch

        worst_index = self._argworst(self.saved_scores)
        worst_score = self.saved_scores[worst_index]

        if self._is_better(score, worst_score):
            # we need to save this score instead the worst score
            worst_var_file = self.variables_files[worst_index]
            self.save(worst_var_file)
            self.saved_scores[worst_index] = score
            log("Variable file saved in {}".format(worst_var_file))

            # update symlink and best score index
            if self.best_score == score:
                self._update_best_vars(worst_index)
                self.best_score_index = worst_index

            log("Best scores saved so far: {}".format(
                self.saved_scores))

    # pylint: disable=too-many-locals
    def _run_executables(self,
                         batch: Dataset,
                         executables: List[Executable],
                         train: bool) -> None:
        all_feedables = set()  # type: Set[Any]
        all_tensors_to_execute = {}

        # We might want to feed different values to each session
        # E.g. when executing only step at a time during ensembling
        feed_dicts = [{} for _ in range(len(self.sessions))] \
            # type: List[FeedDict]

        tensor_list_lengths = []  # type: List[int]

        for executable in executables:
            if executable.result is None:
                (feedables,
                 tensors_to_execute,
                 add_feed_dicts) = executable.next_to_execute()
                all_feedables = all_feedables.union(feedables)
                all_tensors_to_execute[executable] = tensors_to_execute
                if add_feed_dicts:
                    for fdict, add_fd in zip(feed_dicts, add_feed_dicts):
                        fdict.update(add_fd)
                tensor_list_lengths.append(len(tensors_to_execute))
            else:
                tensor_list_lengths.append(0)

        feed_dict = _feed_dicts(batch, all_feedables, train=train)

        for fdict in feed_dicts:
            fdict.update(feed_dict)

        session_results = [sess.run(all_tensors_to_execute,
                                    feed_dict=fd)
                           for sess, fd in zip(self.sessions, feed_dicts)]

        for executable in executables:
            if executable.result is None:
                executable.collect_results(
                    [res[executable] for res in session_results])

    # pylint: disable=too-many-locals
    def execute(self,
                batch: Dataset,
                runners: Sequence[Union[BaseRunner, Trainer]],
                train: bool = False,
                compute_losses: bool = True,
                summaries: bool = True) -> List[ExecutionResult]:
        """Execute runners on a batch of data.

        First, extract executables from the provided runners, telling the
        runners whether to compute also losses and summaries. Second, until
        all executables are satisfied (have the `result` attribute set),
        run the executables on the batch.

        Arguments:
            batch: A batch of data.
            execution_scripts: List of runners to execute.
            train: Training mode flag (this value is fed to the `train_mode`
                 placeholders in model parts).
            compute_losses: Flag to runners whether run loss operations.
            summaries: Flag to runners whether to run summary operations.

        Returns:
            A list of `ExecutionResult` tuples, one for each executable
            (runner).
        """
        executables = [runner.get_executable(compute_losses=compute_losses,
                                             summaries=summaries,
                                             num_sessions=len(self.sessions))
                       for runner in runners]

        # TODO refactor runner results to properties
        while not all(getattr(ex, "result") is not None for ex in executables):
            self._run_executables(batch, executables, train)

        return [getattr(ex, "result") for ex in executables]

    def save(self, variable_files: Union[str, List[str]]) -> None:
        if isinstance(variable_files, str) and len(self.sessions) == 1:
            self.saver.save(self.sessions[0], variable_files)
            return

        if isinstance(variable_files, str):
            variable_files = ["{}.{}".format(
                variable_files, i) for i in range(len(self.sessions))]

        if len(variable_files) != len(self.sessions):
            raise Exception(
                "Provided {} files for saving {} sessions.".format(
                    len(variable_files), len(self.sessions)))

        for sess, file_name in zip(self.sessions, variable_files):
            self.saver.save(sess, file_name)

    def restore(self, variable_files: Union[str, List[str]]) -> None:
        if isinstance(variable_files, str):
            variable_files = [variable_files]
        if len(variable_files) != len(self.sessions):
            raise Exception(
                "Provided {} files for restoring {} sessions.".format(
                    len(variable_files), len(self.sessions)))

        for sess, file_name in zip(self.sessions, variable_files):
            log("Loading variables from {}".format(file_name))
            self.saver.restore(sess, file_name)
            log("Variables loaded from {}".format(file_name))

    def restore_best_vars(self) -> None:
        # TODO warn when link does not exist
        self.restore(self.variables_files[self.best_score_index])

    def initialize_model_parts(
            self, runners: List[Any], save: bool = False) -> None:
        """Initialize model parts variables from their checkpoints."""

        if any(not hasattr(r, "parameterizeds") for r in runners):
            raise TypeError(
                "Args to initialize_model_parts must be trainers or runners")

        parameterizeds = set.union(*[rnr.parameterizeds for rnr in runners])
        for coder in parameterizeds:
            for session in self.sessions:
                coder.load(session)

        if save:
            self.save(self.variables_files[0])


def _feed_dicts(dataset: Dataset, coders: Set[Feedable], train: bool = False):
    """Feed the coders with data from dataset.

    This function ensures all encoder and decoder objects feed their the data
    they need from the dataset.
    """
    res = {}

    for coder in coders:
        res.update(coder.feed_dict(dataset, train=train))

    return res


def get_default_tf_manager() -> TensorFlowManager:
    return TensorFlowManager(num_sessions=1, num_threads=4)
