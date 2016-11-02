"""Interface between the data and TF session.

This module impelements the TensorFlow manager which encapsulates the graph
execution in existing sessions.

"""

from typing import List, Union

import numpy as np
import tensorflow as tf
from neuralmonkey.logging import log

# tests: pylint,mypy

# pylint: disable=invalid-name
RunResult = Union[float, np.Array, tf.Summary]

# pylint: disable=too-few-public-methods
class TensorFlowManager(object):
    """Inteface between computational graph, data and TF sessions.

    Attributes:
        sessions: List of active Tensorflow sessions.
    """

    def __init__(self, num_sessions, num_threads, variable_files=None,
                 gpu_allow_growth=True):
        """Initialize a TensorflowManager.

        At this moment the graph must already exist. This method initializes
        required number of TensorFlow sessions and initializes them with
        provided variable files if they are provided.

        Args:
            num_sessions: Number of sessions to be initialized.
            num_threads: Number of threads sessions will run in.
            variable_files: List of variable files.
        """

        session_cfg = tf.ConfigProto()
        session_cfg.inter_op_parallelism_threads = num_threads
        session_cfg.intra_op_parallelism_threads = num_threads
        session_cfg.allow_soft_placement = True # needed for multiple GPUs
        session_cfg.gpu_options.allow_growth = gpu_allow_growth

        self.sessions = [tf.Session(config=session_cfg)
                         for _ in range(num_sessions)]
        self.saver = tf.train.Saver()

        if variable_files:
            if len(variable_files) != num_sessions:
                raise Exception(("The number of provided variable files ({}) "
                                 "is different than a number sessions ({})")
                                .format(len(variable_files), num_sessions))
        for sess, var_file in zip(self.sessions, variable_files):
            log("Loading variables from {}".format(var_file))
            self.saver.restore(sess, var_file)

    # pylint: disable=too-many-locals
    def execute(self, dataset, execution_scripts, train=False, batch_size=None):
        if batch_size is None:
            batch_size = len(dataset)
        batched_dataset = dataset.batch_dataset(batch_size)

        batch_results = [[] for _ in execution_scripts]
        for batch in batched_dataset:
            executables = [s.get_executable(train=train)
                           for s in execution_scripts]
            while not all(ex.is_finished for ex in executables):
                all_feedables = set()
                all_tensors_to_execute = []
                tensor_list_lengths = []

                for executable in executables:
                    if not executable.is_finished():
                        feedables, tensors_to_execute = executable.next_to_execute()
                        all_feedables = all_feedables.union(feedables)
                        all_tensors_to_execute.extend(tensors_to_execute)
                        tensor_list_lengths.append(len(tensors_to_execute))
                    else:
                        tensor_list_lengths.append(0)

                feed_dict = _feed_dicts(batch, all_feedables, train=train)

                session_results = [sess.run(all_tensors_to_execute,
                                            feed_dict=feed_dict) for sess in self.sessions]

                results_by_executable = _partition_results(
                    session_results, tensor_list_lengths)

                for executable, results in zip(executables, results_by_executable):
                    executable.collect(results)

            for script_list, executable in zip(batch_results, executables):
                script_list.append(executable.results)

        results = []
        for result_list, script in zip(executables, execution_scripts):
            results.append(script.collect_finished(result_list))

        return results

    def save(self, variable_files: Union[str, List[str]]):
        if isinstance(variable_files, str):
            variable_files = ["{}.{}".format(
                variable_files, i) for i in range(len(self.sessions()))]

        if len(variable_files) != len(self.sessions):
            raise Exception("Provided {} files for restoring {} sessions.".format(
                len(variable_files), len(self.sessions)))

        for sess, file_name in zip(self.sessions, variable_files):
            self.saver.save(sess, file_name)

    def restore(self, variable_files: List[str]) -> None:
        if len(variable_files) != len(self.sessions):
            raise Exception("Provided {} files for restoring {} sessions.".format(
                len(variable_files), len(self.sessions)))

        for sess, file_name in zip(self.sessions, variable_files):
            self.saver.restore(sess, file_name)


def _partition_results(session_results: List[RunResult],
                       tensor_list_lengths: List[int]) -> List[RunResult]:
    """Split the session run results back for their executables."""
    results_by_executable = []
    res_start = 0
    for length in tensor_list_lengths:
        this_executable_results = []
        for results in session_results:
            this_executable_results.append(
                results[res_start:res_start + length])
        res_start += length
        results_by_executable.append(this_executable_results)
    return results_by_executable


def _feed_dicts(dataset, coders, train=False):
    """
    This function ensures all encoder and decoder objects feed their the data
    they need from the dataset.
    """
    res = {}

    for coder in coders:
        res.update(coder.feed_dict(dataset, train=train))

    return res
