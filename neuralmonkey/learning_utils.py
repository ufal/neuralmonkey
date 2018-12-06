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
from neuralmonkey.dataset import Dataset, BatchingScheme
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.runners.base_runner import (
    BaseRunner, ExecutionResult, reduce_execution_results, GraphExecutor)
from neuralmonkey.runners.dataset_runner import DatasetRunner
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer
from neuralmonkey.trainers.delayed_update_trainer import DelayedUpdateTrainer
from neuralmonkey.training_profiler import TrainingProfiler

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
def training_loop(cfg: Namespace) -> None:
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

    feedables = set.union(*[ex.feedables for ex in cfg.runners + cfg.trainers])

    log("Starting training")
    profiler = TrainingProfiler()
    profiler.training_start()

    step = 0
    seen_instances = 0
    last_seen_instances = 0
    interrupt = None

    try:
        for epoch_n in range(1, cfg.epochs + 1):
            train_batches = cfg.train_dataset.batches(cfg.batching_scheme)

            if epoch_n == 1 and cfg.train_start_offset:
                if cfg.train_dataset.shuffled and not cfg.train_dataset.lazy:
                    warn("Not skipping training instances with shuffled "
                         "non-lazy dataset")
                else:
                    _skip_lines(cfg.train_start_offset, train_batches)

            log_print("")
            log("Epoch {} begins".format(epoch_n), color="red")
            profiler.epoch_start()

            for batch_n, batch in enumerate(train_batches):
                step += 1
                seen_instances += len(batch)

                if cfg.log_timer(step, profiler.last_log_time):
                    trainer_result = cfg.tf_manager.execute(
                        batch, feedables, cfg.trainers, train=True,
                        summaries=True)
                    train_results, train_outputs, f_batch = run_on_dataset(
                        cfg.tf_manager, cfg.runners, cfg.dataset_runner, batch,
                        cfg.postprocess, write_out=False,
                        batching_scheme=cfg.runners_batching_scheme)
                    # ensure train outputs are iterable more than once
                    train_outputs = {
                        k: list(v) for k, v in train_outputs.items()}
                    train_evaluation = evaluation(
                        cfg.evaluation, f_batch, cfg.runners, train_results,
                        train_outputs)

                    _log_continuous_evaluation(
                        tb_writer, cfg.main_metric, train_evaluation,
                        seen_instances, epoch_n, cfg.epochs, trainer_result,
                        train=True)

                    profiler.log_done()

                else:
                    cfg.tf_manager.execute(
                        batch, feedables, cfg.trainers, train=True,
                        summaries=False)

                if cfg.val_timer(step, profiler.last_val_time):

                    log_print("")
                    profiler.validation_start()

                    val_examples = 0
                    for val_id, valset in enumerate(cfg.val_datasets):
                        val_examples += len(valset)

                        val_results, val_outputs, f_valset = run_on_dataset(
                            cfg.tf_manager, cfg.runners, cfg.dataset_runner,
                            valset, cfg.postprocess, write_out=False,
                            batching_scheme=cfg.runners_batching_scheme)
                        # ensure val outputs are iterable more than once
                        val_outputs = {k: list(v)
                                       for k, v in val_outputs.items()}
                        val_evaluation = evaluation(
                            cfg.evaluation, f_valset, cfg.runners, val_results,
                            val_outputs)

                        valheader = ("Validation (epoch {}, batch number {}):"
                                     .format(epoch_n, batch_n))
                        log(valheader, color="blue")
                        _print_examples(
                            f_valset, val_outputs,
                            cfg.val_preview_input_series,
                            cfg.val_preview_output_series,
                            cfg.val_preview_num_examples)
                        log_print("")
                        log(valheader, color="blue")

                        # The last validation set is selected to be the main
                        if val_id == len(cfg.val_datasets) - 1:
                            this_score = val_evaluation[cfg.main_metric]
                            cfg.tf_manager.validation_hook(this_score, epoch_n,
                                                           batch_n)

                            if this_score == cfg.tf_manager.best_score:
                                best_score_str = colored(
                                    "{:.4g}".format(cfg.tf_manager.best_score),
                                    attrs=["bold"])

                                # store also graph parts
                                rnrs = cfg.runners + cfg.trainers
                                # TODO: refactor trainers/runners so that they
                                # have the same API predecessor
                                parameterizeds = set.union(
                                    *[rnr.parameterizeds
                                      for rnr in rnrs])
                                for coder in parameterizeds:
                                    for session in cfg.tf_manager.sessions:
                                        coder.save(session)
                            else:
                                best_score_str = "{:.4g}".format(
                                    cfg.tf_manager.best_score)

                            log("best {} on validation: {} (in epoch {}, "
                                "after batch number {})"
                                .format(cfg.main_metric, best_score_str,
                                        cfg.tf_manager.best_score_epoch,
                                        cfg.tf_manager.best_score_batch),
                                color="blue")

                        v_name = "val_{}".format(val_id) if len(
                            cfg.val_datasets) > 1 else None
                        _log_continuous_evaluation(
                            tb_writer, cfg.main_metric, val_evaluation,
                            seen_instances, epoch_n, cfg.epochs, val_results,
                            train=False, dataset_name=v_name)

                    profiler.validation_done()
                    profiler.log_after_validation(
                        val_examples, seen_instances - last_seen_instances)
                    last_seen_instances = seen_instances

                    log_print("")

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


def run_on_dataset(tf_manager: TensorFlowManager,
                   runners: List[BaseRunner],
                   dataset_runner: DatasetRunner,
                   dataset: Dataset,
                   postprocess: Postprocess,
                   batching_scheme: BatchingScheme,
                   write_out: bool = False,
                   log_progress: int = 0) -> Tuple[
                       List[ExecutionResult],
                       Dict[str, List],
                       Dict[str, List]]:
    """Apply the model on a dataset and optionally write outputs to files.

    This function processes the dataset in batches and optionally prints out
    the execution progress.

    Args:
        tf_manager: TensorFlow manager with initialized sessions.
        runners: A function that runs the code
        dataset_runner: A runner object that fetches the data inputs
        dataset: The dataset on which the model will be executed.
        evaluators: List of evaluators that are used for the model
            evaluation if the target data are provided.
        postprocess: Dataset-level postprocessors
        write_out: Flag whether the outputs should be printed to a file defined
            in the dataset object.
        batching_scheme: Scheme used for batching.
        log_progress: log progress every X seconds

        extra_fetches: Extra tensors to evaluate for each batch.

    Returns:
        Tuple of resulting sentences/numpy arrays, and evaluation results if
        they are available which are dictionary function -> value.

    """
    # If the dataset contains the target series, compute also losses.
    contains_targets = all(runner.decoder_data_id in dataset
                           for runner in runners
                           if runner.decoder_data_id is not None)

    last_log_time = time.process_time()
    batch_results = [[] for _ in runners]  # type: List[List[ExecutionResult]]
    batch_results.append([])  # For dataset runner

    feedables = set.union(*[runner.feedables for runner in runners])
    feedables |= dataset_runner.feedables

    processed_examples = 0
    for batch in dataset.batches(batching_scheme):
        if 0 < log_progress < time.process_time() - last_log_time:
            log("Processed {} examples.".format(processed_examples))
            last_log_time = time.process_time()

        executors = []  # type: List[GraphExecutor]
        executors.extend(runners)
        executors.append(dataset_runner)

        execution_results = tf_manager.execute(
            batch, feedables, executors, compute_losses=contains_targets)

        processed_examples += len(batch)

        for script_list, ex_result in zip(batch_results, execution_results):
            script_list.append(ex_result)

    # Transpose runner interim results.
    all_results = [reduce_execution_results(res) for res in batch_results[:-1]]

    # TODO uncomment this when dataset runner starts outputting the dataset
    # input_transposed = reduce_execution_results(batch_results[-1]).outputs
    # fetched_input = {
    #     k: [dic[k] for dic in input_transposed] for k in input_transposed[0]}

    fetched_input = {s: list(dataset.get_series(s)) for s in dataset.series}
    fetched_input_lengths = {s: len(fetched_input[s]) for s in dataset.series}

    if len(set(fetched_input_lengths.values())) != 1:
        warn("Fetched input dataset series are not of the same length: {}"
             .format(str(fetched_input_lengths)))

    dataset_len = fetched_input_lengths[dataset.series[0]]

    # Convert execution results to dictionary.
    result_data = {runner.output_series: result.outputs
                   for runner, result in zip(runners, all_results)}

    # Run dataset-level postprocessing.
    if postprocess is not None:
        for series_name, postprocessor in postprocess:
            postprocessed = postprocessor(fetched_input, result_data)
            if not hasattr(postprocessed, "__len__"):
                postprocessed = list(postprocessed)

            result_data[series_name] = postprocessed

    # Check output series lengths.
    for series_id, data in result_data.items():
        if len(data) != dataset_len:
            warn("Output '{}' for dataset '{}' has length {}, but input "
                 "dataset size is {}".format(series_id, dataset.name,
                                             len(data), dataset_len))

    if write_out and dataset.outputs is not None:
        for series_id, data in result_data.items():
            if series_id in dataset.outputs:
                path, writer = dataset.outputs[series_id]
                writer(path, data)
            else:
                log("There is no file for output series '{}' in dataset: '{}'"
                    .format(series_id, dataset.name), color="red")
    elif write_out:
        log("Dataset does not have any outputs, nothing to write out.",
            color="red")

    return all_results, result_data, fetched_input


def evaluation(evaluators, batch, runners, execution_results, result_data):
    """Evaluate the model outputs.

    Args:
        evaluators: List of tuples of series and evaluation functions.
        batch: Batch of data against which the evaluation is done.
        runners: List of runners (contains series ids and loss names).
        execution_results: Execution results that include the loss values.
        result_data: Dictionary from series names to list of outputs.

    Returns:
        Dictionary of evaluation names and their values which includes the
        metrics applied on respective series loss and loss values from the run.
    """
    eval_result = {}

    # losses
    for runner, result in zip(runners, execution_results):
        for name, value in zip(runner.loss_names, result.losses):
            eval_result["{}/{}".format(runner.output_series, name)] = value

    # evaluation metrics
    for hypothesis_id, reference_id, function in evaluators:
        if reference_id not in batch or hypothesis_id not in result_data:
            continue

        desired_output = batch[reference_id]
        model_output = result_data[hypothesis_id]
        eval_result["{}/{}".format(hypothesis_id, function.name)] = function(
            model_output, desired_output)

    return eval_result


def _log_continuous_evaluation(tb_writer: tf.summary.FileWriter,
                               main_metric: str,
                               eval_result: Evaluation,
                               seen_instances: int,
                               epoch: int,
                               max_epochs: int,
                               execution_results: List[ExecutionResult],
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
        for result in execution_results:
            for summaries in [result.scalar_summaries,
                              result.histogram_summaries,
                              result.image_summaries]:
                if summaries is not None:
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


def _print_examples(dataset: Dict[str, List[Any]],
                    outputs: Dict[str, List[Any]],
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
        target_series_names = [s for s in target_series_names
                               if s in val_preview_input_series]
        source_series_names = [s for s in source_series_names
                               if s in val_preview_input_series]

    if val_preview_output_series is not None:
        output_series_names = [s for s in output_series_names
                               if s in val_preview_output_series]

    # for further indexing we need to make sure, all relevant
    # dataset series are lists
    target_series = {series_id: list(dataset[series_id])
                     for series_id in target_series_names}
    source_series = {series_id: list(dataset[series_id])
                     for series_id in source_series_names}

    dataset_length = len(next(iter(dataset.values())))
    num_examples = min(dataset_length, num_examples)

    for i in range(num_examples):
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
