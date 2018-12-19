# pylint: disable=too-many-lines
# TODO de-clutter this file!

from argparse import Namespace
import time
# pylint: disable=unused-import
from typing import (Any, Callable, Dict, List, Tuple, Optional, Union,
                    Iterable, Iterator, Set)
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf
from termcolor import colored

from neuralmonkey.logging import log, log_print, warn
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.runners.base_runner import (
    BaseRunner, ExecutionResult, FeedDict, GraphExecutor, OutputSeries)
from neuralmonkey.runners.dataset_runner import DatasetRunner
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer
from neuralmonkey.trainers.delayed_update_trainer import DelayedUpdateTrainer
from neuralmonkey.training_profiler import TrainingProfiler
from neuralmonkey.writers.plain_text_writer import Writer

# pylint: disable=invalid-name
Evaluation = Dict[str, float]
SeriesName = str
EvalConfiguration = List[Union[Tuple[SeriesName, Any],
                               Tuple[SeriesName, SeriesName, Any]]]
Postprocess = Optional[List[Tuple[SeriesName, Callable]]]
Trainer = Union[GenericTrainer, MultitaskTrainer, DelayedUpdateTrainer]
# pylint: enable=invalid-name


# pylint: disable=too-many-nested-blocks,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-arguments
def training_loop(cfg: Namespace,
                  handle: tf.Tensor,
                  train_handle: str,
                  val_handles: List[str]) -> None:
    """Execute the training loop for given graph and data.

    Arguments:
        cfg: Experiment configuration namespace.
    """
    _check_series_collisions(cfg.runners, cfg.postprocess)
    _log_model_variables(cfg.trainers)
    _initialize_model(cfg.tf_manager, cfg.initial_variables,
                      cfg.runners + cfg.trainers)

    log("Initializing TensorBoard summary writer.")
    tb_writer = tf.summary.FileWriter(cfg.output,
                                      cfg.tf_manager.sessions[0].graph)
    log("TensorBoard writer initialized.")

    runtime_feedables = set.union(*[ex.feedables for ex in cfg.runners])
    runtime_feedables |= cfg.dataset_runner.feedables
    train_feedables = set.union(*[ex.feedables for ex in cfg.trainers])

    parameterizeds = set.union(*[ex.parameterizeds for ex in cfg.runners])
    parameterizeds |= set.union(*[ex.parameterizeds for ex in cfg.trainers])

    log("Starting training")
    profiler = TrainingProfiler()
    profiler.training_start()

    step = 0
    seen_instances = 0
    last_seen_instances = 0
    interrupt = None

    def train_step() -> None:
        nonlocal seen_instances

        res = cfg.tf_manager.execute(
            {handle: train_handle}, train_feedables, cfg.trainers, train=True,
            summaries=False)

        seen_instances += res[0].size

    def train_step_with_log(epoch_n: int) -> None:
        nonlocal seen_instances

        f_batch = prefetch_dataset(
            cfg.tf_manager, {handle: train_handle}, cfg.dataset_runner)

        batch_feed_dict = {}
        for s_id, data in f_batch.outputs.items():
            batch_feed_dict[cfg.dataset_runner.dataset[s_id]] = data

        exec_result, _ = run_batch(
            cfg.tf_manager, batch_feed_dict, cfg.runners, cfg.dataset_runner,
            runtime_feedables, cfg.postprocess, compute_losses=True)

        trainer_result = cfg.tf_manager.execute(
            batch_feed_dict, train_feedables, cfg.trainers, train=True,
            summaries=True)

        seen_instances += trainer_result[0].size

        # Overwrite f_batch with a version with textual data represented as a
        # list of strings instead of numpy array of bytes
        exec_result, f_batch = run_batch(
            cfg.tf_manager, batch_feed_dict, cfg.runners, cfg.dataset_runner,
            runtime_feedables, cfg.postprocess, compute_losses=True)

        train_evaluation = evaluation(
            cfg.evaluation, f_batch.outputs, exec_result)

        _log_continuous_evaluation(
            tb_writer, cfg.main_metric, train_evaluation, seen_instances,
            epoch_n, cfg.epochs, trainer_result[0], train=True)

        profiler.log_done()

    def validate(epoch_n: int, batch_n: int) -> None:
        nonlocal seen_instances
        nonlocal last_seen_instances

        log_print("")
        profiler.validation_start()
        val_examples = 0

        cfg.tf_manager.init_validation()

        valheader = "Validation (epoch {}, batch number {}):".format(epoch_n,
                                                                     batch_n)
        log(valheader, color="blue")

        for val_id, valhand in enumerate(val_handles):

            val_result, f_valset = run_on_dataset(
                cfg.tf_manager, {handle: valhand}, cfg.runners,
                cfg.dataset_runner, runtime_feedables, cfg.postprocess,
                compute_losses=True)

            val_examples += val_result.size

            _print_examples(
                f_valset.outputs, val_result.outputs, f_valset.size,
                cfg.val_preview_input_series, cfg.val_preview_output_series,
                cfg.val_preview_num_examples)
            log_print("")
            log(valheader, color="blue")

            val_evaluation = evaluation(
                cfg.evaluation, f_valset.outputs, val_result)

            # The last validation set is selected to be the main
            if val_id == len(cfg.val_datasets) - 1:
                this_score = val_evaluation[cfg.main_metric]
                cfg.tf_manager.validation_hook(this_score, epoch_n, batch_n)

                if this_score == cfg.tf_manager.best_score:
                    best_score_str = colored(
                        "{:.4g}".format(cfg.tf_manager.best_score),
                        attrs=["bold"])

                    # store also graph parts
                    for coder in parameterizeds:
                        for session in cfg.tf_manager.sessions:
                            coder.save(session)
                else:
                    best_score_str = "{:.4g}".format(cfg.tf_manager.best_score)

                    log("best {} on validation: {} (in epoch {}, after batch "
                        "number {})".format(cfg.main_metric, best_score_str,
                                            cfg.tf_manager.best_score_epoch,
                                            cfg.tf_manager.best_score_batch),
                        color="blue")

            v_name = "val_{}".format(
                val_id) if len(cfg.val_datasets) > 1 else None

            _log_continuous_evaluation(
                tb_writer, cfg.main_metric, val_evaluation, seen_instances,
                epoch_n, cfg.epochs, val_result, train=False,
                dataset_name=v_name)

        profiler.validation_done()
        profiler.log_after_validation(
            val_examples, seen_instances - last_seen_instances)
        last_seen_instances = seen_instances

        log_print("")

    def epoch(epoch_n: int) -> None:
        nonlocal step

        cfg.tf_manager.init_training()

        batch_n = 0

        log_print("")
        log("Epoch {} begins".format(epoch_n), color="red")
        profiler.epoch_start()

        while True:
            try:
                batch_n += 1
                step += 1

                if cfg.log_timer(step, profiler.last_log_time):
                    train_step_with_log(epoch_n)
                else:
                    train_step()

                if cfg.val_timer(step, profiler.last_val_time):
                    validate(epoch_n, batch_n)

            except tf.errors.OutOfRangeError:
                break

    try:
        for epoch_n in range(1, cfg.epochs + 1):
            epoch(epoch_n)
    except KeyboardInterrupt as ex:
        interrupt = ex

    log("Training finished. Maximum {} on validation data: {:.4g}, epoch {}"
        .format(cfg.main_metric, cfg.tf_manager.best_score,
                cfg.tf_manager.best_score_epoch))

    if interrupt is not None:
        raise interrupt  # pylint: disable=raising-bad-type


def _log_model_variables(trainers: List[Trainer]) -> None:

    var_list = list(set().union(*[t.var_list for t in trainers])) \
               # type: List[tf.Variable]

    trainable_vars = tf.trainable_variables()
    if not var_list:
        var_list = trainable_vars

    assert var_list is not None
    fixed_vars = [var for var in trainable_vars if var not in var_list]

    total_params = 0

    logstr = "The model has {} trainable variables{}:\n\n".format(
        len(trainable_vars),
        " ({} {})".format(len(fixed_vars), colored("fixed", on_color="on_red"))
        if fixed_vars else "")

    logstr += colored(
        "{: ^80}{: ^20}{: ^10}\n".format("Variable name", "Shape", "Size"),
        color="yellow", attrs=["bold"])

    for var in trainable_vars:

        shape = var.get_shape().as_list()
        params_in_var = int(np.prod(shape))
        total_params += params_in_var

        name = var.name
        if var not in var_list:
            name = colored(name, on_color="on_red")
        # Pad and compensate for control characters:
        name = name.ljust(80 + (len(name) - len(var.name)))
        log_entry = "{}{: <20}{: >10}".format(name, str(shape), params_in_var)
        logstr += "\n{}".format(log_entry)

    logstr += "\n"

    log(logstr)
    log("Total number of all parameters: {}".format(total_params))


def _initialize_model(tf_manager: TensorFlowManager,
                      initial_variables: Optional[List[str]],
                      executables: List[GraphExecutor]):

    if initial_variables is None:
        # Assume we don't look at coder checkpoints when global
        # initial variables are supplied
        tf_manager.initialize_model_parts(executables)
    else:
        try:
            tf_manager.restore(initial_variables)
        except tf.errors.NotFoundError:
            warn("Some variables were not found in checkpoint.)")


def _check_series_collisions(runners: List[BaseRunner],
                             postprocess: Postprocess) -> None:
    """Check if output series names do not collide."""
    runners_outputs = set()  # type: Set[str]
    for runner in runners:
        series = runner.output_series
        if series in runners_outputs:
            raise Exception(("Output series '{}' is multiple times among the "
                             "runners' outputs.").format(series))
        else:
            runners_outputs.add(series)
    if postprocess is not None:
        for series, _ in postprocess:
            if series in runners_outputs:
                raise Exception(("Postprocess output series '{}' "
                                 "already exists.").format(series))
            else:
                runners_outputs.add(series)


def prefetch_dataset(tf_manager: TensorFlowManager,
                     data_feed_dict: FeedDict,
                     dataset_runner: DatasetRunner) -> ExecutionResult:
    """Pre-fetch a single batch from the dataset.

    Use this function for pre-fetching a batch as a feed dictionary to
    be able to evaluate the model on a single batch multiple times.
    """
    data_result = tf_manager.execute(data_feed_dict, {dataset_runner},
                                     [dataset_runner])

    return data_result[0]


def run_batch(tf_manager: TensorFlowManager,
              data_feed_dict: FeedDict,
              runners: List[BaseRunner],
              dataset_runner: DatasetRunner,
              feedables: Set[Feedable],
              postprocess: Postprocess,
              compute_losses: bool = False) -> Tuple[ExecutionResult,
                                                     ExecutionResult]:
    """Run a single batch of computation.

    The goal of this function is to run a batch and merge execution results
    from different graph executors.

    Arguments:
        tf_manager: The TF manager instance.
        data_feed_dict: Feed dict containing the dataset handle (or the
            pre-fetched dataset itself).
        runners: A list of runners to execute.
        dataset_runner: The dataset runner instance.
        feedables: A set of feedables from the model.
        postprocess: A dataset-level postprocess function.
        compute_losses: Flag for executors whether to compute loss tensors.

    Returns:
        A pair of execution results - the result and the input batch.
    """
    executors = []  # type: List[GraphExecutor]
    executors.extend(runners)
    executors.append(dataset_runner)

    execution_results = tf_manager.execute(
        data_feed_dict, feedables, executors, compute_losses=compute_losses)

    sizes = set(ex.size for ex in execution_results)
    assert len(sizes) == 1
    processed_examples = next(iter(sizes))

    results = execution_results[:-1]
    dataset = execution_results[-1]

    # Join execution results from different runners
    result_data = {}  # type: Dict[str, OutputSeries]
    for s_id, data in (
            pair for res in results for pair in res.outputs.items()):

        # for s_id, data in output.items():
        if s_id in result_data:
            raise ValueError("Overwriting output series forbidden.")
        result_data[s_id] = data

    # Replace string arrays with lists in the fetched dataset
    for s_id, data in dataset.outputs.items():
        if isinstance(data, np.ndarray) and data.dtype == np.object:
            dataset.outputs[s_id] = tf.contrib.framework.nest.map_structure(
                tf.compat.as_text, data.tolist())

    # Run dataset-level postprocessing.
    if postprocess is not None:
        for s_id, postprocessor in postprocess:
            result_data[s_id] = postprocessor(dataset.outputs, result_data)

    # Check output series lengths.
    for s_id, data in result_data.items():
        if len(data) != processed_examples:
            warn("Output '{}' has length {}, but input dataset size is {}"
                 .format(s_id, len(data), processed_examples))

    losses = {}  # type: Dict[str, float]
    for loss_dict in [res.losses for res in results]:
        if any(l in losses for l in loss_dict):
            raise ValueError("Overwriting losses forbidden.")
        losses.update(loss_dict)

    flat_summaries = [s for res in execution_results for s in res.summaries]

    return ExecutionResult(
        result_data, losses, processed_examples, flat_summaries), dataset


def run_on_dataset(tf_manager: TensorFlowManager,
                   data_feed_dict: FeedDict,
                   runners: List[BaseRunner],
                   dataset_runner: DatasetRunner,
                   feedables: Set[Feedable],
                   postprocess: Postprocess,
                   write_out: Dict[str, Tuple[str, Writer]] = None,
                   log_progress: int = 0,
                   compute_losses: bool = False) -> Tuple[
                       ExecutionResult, ExecutionResult]:
    """Apply the model on a dataset and optionally write outputs to files.

    This function processes the dataset in batches and optionally prints out
    the execution progress.

    Args:
        tf_manager: TensorFlow manager with initialized sessions.
        runners: A function that runs the code
        dataset_runner: A runner object that fetches the data inputs
        data_feed_dict: Feed dict that provides the dataset.
        evaluators: List of evaluators that are used for the model
            evaluation if the target data are provided.
        postprocess: Dataset-level postprocessors
        write_out: An optional mapping of series IDs to paths/writer tuples.
        log_progress: log progress every X seconds

    Returns:
        Tuple of resulting sentences/numpy arrays, and evaluation results if
        they are available which are dictionary function -> value.

    """
    last_log_time = time.process_time()
    batch_results = []
    batch_inputs = []

    processed_examples = 0

    while True:
        try:
            if 0 < log_progress < time.process_time() - last_log_time:
                log("Processed {} examples.".format(processed_examples))
                last_log_time = time.process_time()

            result, dataset = run_batch(tf_manager, data_feed_dict, runners,
                                        dataset_runner, feedables, postprocess,
                                        compute_losses)
            batch_results.append(result)
            batch_inputs.append(dataset)
            processed_examples += result.size

        except tf.errors.OutOfRangeError:
            break

    # Join execution results from different batches. Note that the arrays can
    # differ in both batch and time dimensions.
    joined_result = join_execution_results(batch_results)
    joined_inputs = join_execution_results(batch_inputs)

    # POZOR TOHLE SE NESMI SMAZAT Z run_on_dataset !!! (anebo se to dá dál)
    # TODO(tf-data)
    # if write_out and dataset.outputs is not None:
    #     for series_id, data in result_data.items():
    #         if series_id in dataset.outputs:
    #             path, writer = dataset.outputs[series_id]
    #             writer(path, data)
    #         else:
    #             log("There is no file for output series '{}' in dataset: "
    #                 "'{}'".format(series_id, dataset.name), color="red")
    # elif write_out:
    #     log("Dataset does not have any outputs, nothing to write out.",
    #         color="red")

    return joined_result, joined_inputs


# TODO(tf-data) add unit tests!
def join_execution_results(
        execution_results: List[ExecutionResult]) -> ExecutionResult:
    """Aggregate batch of execution results from a single runner."""

    losses_sum = {loss: 0. for loss in execution_results[0].losses}

    def join(output_series: List[OutputSeries]) -> OutputSeries:
        """Join a list of batches of results into a flat list of outputs."""
        joined = []  # type: List[Any]

        for item in output_series:
            joined.extend(item)

        # If the list is a list of np.arrays, concatenate the list along first
        # dimension (batch). Otherwise, return the list.
        if joined and isinstance(joined[0], np.ndarray):
            return np.array(joined)

        return joined

    outputs = {}  # type: Dict[str, OutputSeries]
    for key in execution_results[0].outputs.keys():
        outputs[key] = join([res.outputs[key] for res in execution_results])

    for result in execution_results:
        for l_id, loss in result.losses.items():
            losses_sum[l_id] += loss * result.size

    total_size = sum(res.size for res in execution_results)
    losses = {l_id: loss / total_size for l_id, loss in losses_sum.items()}

    all_summaries = [
        summ for res in execution_results if res.summaries is not None
        for summ in res.summaries]

    return ExecutionResult(outputs, losses, total_size, all_summaries)


def evaluation(evaluators, batch, execution_result):
    """Evaluate the model outputs.

    Args:
        evaluators: List of tuples of series and evaluation functions.
        batch: Batch of data against which the evaluation is done.
        execution_result: Execution result that includes the loss values.

    Returns:
        Dictionary of evaluation names and their values which includes the
        metrics applied on respective series loss and loss values from the run.
    """
    # losses
    eval_result = execution_result.losses

    # evaluation metrics
    for hyp_id, ref_id, evaluator in evaluators:
        if ref_id not in batch or hyp_id not in execution_result.outputs:
            continue

        references = batch[ref_id]
        hypotheses = execution_result.outputs[hyp_id]

        eval_key = "{}/{}".format(hyp_id, evaluator.name)
        eval_result[eval_key] = evaluator(hypotheses, references)

    return eval_result


def _log_continuous_evaluation(tb_writer: tf.summary.FileWriter,
                               main_metric: str,
                               eval_result: Evaluation,
                               seen_instances: int,
                               epoch: int,
                               max_epochs: int,
                               execution_result: ExecutionResult,
                               train: bool = False,
                               dataset_name: str = None) -> None:
    """Log the evaluation results and the TensorBoard summaries."""

    color, prefix = ("yellow", "train") if train else ("blue", "val")

    if dataset_name is not None:
        prefix += "_" + dataset_name

    eval_string = _format_evaluation_line(eval_result, main_metric)
    eval_string = "Epoch {}/{}  Instances {}  {}".format(epoch, max_epochs,
                                                         seen_instances,
                                                         eval_string)
    log(eval_string, color=color)

    if tb_writer:
        for summaries in execution_result.summaries:
            tb_writer.add_summary(summaries, seen_instances)

        external_str = \
            tf.Summary(value=[tf.Summary.Value(tag=prefix + "_" + name,
                                               simple_value=value)
                              for name, value in eval_result.items()])
        tb_writer.add_summary(external_str, seen_instances)


def _format_evaluation_line(evaluation_res: Evaluation,
                            main_metric: str) -> str:
    """Format the evaluation metric for stdout with last one bold."""
    eval_string = "    ".join("{}: {:.4g}".format(name, value)
                              for name, value in evaluation_res.items()
                              if name != main_metric)

    eval_string += colored(
        "    {}: {:.4g}".format(main_metric,
                                evaluation_res[main_metric]),
        attrs=["bold"])

    return eval_string


def print_final_evaluation(eval_result: Evaluation, name: str = None) -> None:
    """Print final evaluation from a test dataset."""
    line_len = 22

    if name is not None:
        log("Model evaluated on '{}'".format(name))

    for eval_name, value in eval_result.items():
        space = "".join([" " for _ in range(line_len - len(eval_name))])
        log("... {}:{} {:.4g}".format(eval_name, space, value))

    log_print("")


def _data_item_to_str(item: Any) -> str:
    if isinstance(item, list):
        return " ".join([_data_item_to_str(i) for i in item])

    if isinstance(item, dict):
        return "{\n      " + "\n      ".join(
            ["{}: {}".format(_data_item_to_str(key), _data_item_to_str(val))
             for key, val in item.items()]) + "\n    }"

    if isinstance(item, np.ndarray) and len(item.shape) > 1:
        return "[numpy tensor, shape {}]".format(item.shape)

    return str(item)


def _print_examples(dataset: Dict[str, List],
                    outputs: Dict[str, List],
                    dataset_size: int,
                    val_preview_input_series: Optional[List[str]] = None,
                    val_preview_output_series: Optional[List[str]] = None,
                    num_examples=15) -> None:
    """Print examples of the model output.

    Arguments:
        dataset: The dataset from which to take examples
        outputs: A mapping from the output series ID to the list of its
            contents
        val_preview_input_series: An optional list of input series to include
            in the preview. An input series is a data series that is present in
            the dataset. It can be either a target series (one that is also
            present in the outputs, i.e. reference), or a source series (one
            that is not among the outputs). In the validation preview, source
            input series and preprocessed target series are yellow and target
            (reference) series are red. If None, all series are written.
        val_preview_output_series: An optional list of output series to include
            in the preview. An output series is a data series that is present
            among the outputs. In the preview, magenta is used as the font
            color for output series
    """
    log_print(colored("Examples:", attrs=["bold"]))

    source_series_names = [s for s in dataset if s not in outputs]
    target_series_names = [s for s in dataset if s in outputs]
    output_series_names = list(outputs.keys())

    assert outputs

    if val_preview_input_series is not None:
        target_series_names = [s_id for s_id in target_series_names
                               if s_id in val_preview_input_series]
        source_series_names = [s_id for s_id in source_series_names
                               if s_id in val_preview_input_series]

    if val_preview_output_series is not None:
        output_series_names = [s for s in output_series_names
                               if s in val_preview_output_series]

    # for further indexing we need to make sure, all relevant
    # dataset series are lists
    target_series = {s_id: dataset[s_id] for s_id in target_series_names}
    source_series = {s_id: dataset[s_id] for s_id in source_series_names}

    for i in range(min(num_examples, dataset_size)):
        log_print(colored("  [{}]".format(i + 1), color="magenta",
                          attrs=["bold"]))

        def print_line(prefix, color, content):
            colored_prefix = colored(prefix, color=color)
            formatted = _data_item_to_str(content)
            log_print("  {}: {}".format(colored_prefix, formatted))

        # Input source series = yellow
        for series_id, data in sorted(source_series.items(),
                                      key=lambda x: x[0]):
            print_line(series_id, "yellow", data[i])

        # Output series = magenta
        for series_id in sorted(output_series_names):
            data = list(outputs[series_id])
            model_output = data[i]
            print_line(series_id, "magenta", model_output)

        # Input target series (a.k.a. references) = red
        for series_id in sorted(target_series_names):
            data = outputs[series_id]
            desired_output = target_series[series_id][i]
            print_line(series_id + " (ref)", "red", desired_output)

        log_print("")


def _skip_lines(start_offset: int,
                batches: Iterator[Dataset]) -> None:
    """Skip training instances from the beginning.

    Arguments:
        start_offset: How many training instances to skip (minimum)
        batches: Iterator over batches to skip
    """
    log("Skipping first {} instances in the dataset".format(start_offset))

    skipped_instances = 0
    while skipped_instances < start_offset:
        try:
            skipped_instances += len(next(batches))  # type: ignore
        except StopIteration:
            raise ValueError("Trying to skip more instances than "
                             "the size of the dataset")

    if skipped_instances > 0:
        log("Skipped {} instances".format(skipped_instances))
