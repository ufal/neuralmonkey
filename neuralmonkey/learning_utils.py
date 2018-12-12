# pylint: disable=too-many-lines
# TODO de-clutter this file!

# pylint: disable=unused-import
from typing import (Any, Callable, Dict, List, Tuple, Optional, Union,
                    Iterable, Iterator, Set)
# pylint: enable=unused-import

import time
import numpy as np
import tensorflow as tf
from termcolor import colored
from typeguard import check_argument_types

from neuralmonkey.logging import log, log_print, warn, notice
from neuralmonkey.dataset import Dataset, BatchingScheme
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.runners.base_runner import (
    BaseRunner, ExecutionResult, reduce_execution_results)
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer
from neuralmonkey.trainers.delayed_update_trainer import DelayedUpdateTrainer

# pylint: disable=invalid-name
Evaluation = Dict[str, float]
SeriesName = str
EvalConfiguration = List[Union[Tuple[SeriesName, Any],
                               Tuple[SeriesName, SeriesName, Any]]]
Postprocess = Optional[List[Tuple[SeriesName, Callable]]]
Trainer = Union[GenericTrainer, MultitaskTrainer, DelayedUpdateTrainer]
# pylint: enable=invalid-name


# pylint: disable=too-many-arguments,too-many-locals,too-many-nested-blocks
# pylint: disable=too-many-branches,too-many-statements
def training_loop(tf_manager: TensorFlowManager,
                  epochs: int,
                  trainers: List[Trainer],
                  batching_scheme: BatchingScheme,
                  runners_batching_scheme: BatchingScheme,
                  log_directory: str,
                  evaluators: EvalConfiguration,
                  main_metric: str,
                  runners: List[BaseRunner],
                  train_dataset: Dataset,
                  val_datasets: List[Dataset],
                  test_datasets: Optional[List[Dataset]],
                  log_timer: Callable[[int, float], bool],
                  val_timer: Callable[[int, float], bool],
                  val_preview_input_series: Optional[List[str]],
                  val_preview_output_series: Optional[List[str]],
                  val_preview_num_examples: int,
                  postprocess: Optional[Postprocess],
                  train_start_offset: int,
                  initial_variables: Optional[Union[str, List[str]]],
                  final_variables: str) -> None:
    """Execute the training loop for given graph and data.

    Args:
        tf_manager: TensorFlowManager with initialized sessions.
        epochs: Number of epochs for which the algoritm will learn.
        trainer: The trainer object containg the TensorFlow code for computing
            the loss and optimization operation.
        batch_size: Number of examples in one mini-batch.
        batching_scheme: Batching scheme specification. Cannot be provided when
            batch_size is specified.
        log_directory: Directory where the TensordBoard log will be generated.
            If None, nothing will be done.
        evaluators: List of evaluators. The last evaluator is used as the main.
            An evaluator is a tuple of the name of the generated
            series, the name of the dataset series the generated one is
            evaluated with and the evaluation function. If only one
            series names is provided, it means the generated and
            dataset series have the same name.
        runners: List of runners for logging and evaluation runs
        train_dataset: Dataset used for training
        val_dataset: used for validation. Can be Dataset or a list of datasets.
            The last dataset is used as the main one for storing best results.
            When using multiple datasets. It is recommended to name them for
            better Tensorboard visualization.
        test_datasets: List of datasets used for testing
        logging_period: after how many batches should the logging happen. It
            can also be defined as a time period in format like: 3s; 4m; 6h;
            1d; 3m15s; 3seconds; 4minutes; 6hours; 1days
        validation_period: after how many batches should the validation happen.
            It can also be defined as a time period in same format as logging
        val_preview_input_series: which input series to preview in validation
        val_preview_output_series: which output series to preview in validation
        val_preview_num_examples: how many examples should be printed during
            validation
        train_start_offset: how many lines from the training dataset should be
            skipped. The training starts from the next batch.
        runners_batch_size: batch size of runners. Reuses the training batching
            scheme with bucketing turned off.
        initial_variables: variables used for initialization, for example for
            continuation of training. Provide it with a path to your model
            directory and its checkpoint file group common prefix, e.g.
            "variables.data", or "variables.data.3" in case of multiple
            checkpoints per experiment.
        postprocess: A function which takes the dataset with its output series
            and generates additional series from them.
    """
    check_argument_types()

    _check_series_collisions(runners, postprocess)

    _log_model_variables(
        var_list=list(set().union(*[t.var_list for t in trainers])))

    step = 0
    seen_instances = 0
    last_seen_instances = 0

    if initial_variables is None:
        # Assume we don't look at coder checkpoints when global
        # initial variables are supplied
        tf_manager.initialize_model_parts(
            runners + trainers, save=True)  # type: ignore
    else:
        try:
            tf_manager.restore(initial_variables)
        except tf.errors.NotFoundError:
            warn("Some variables were not found in checkpoint.)")

    # Ignoring type. Mypy complains about summing runner and trainer lists.
    feedables = set.union(
        *[ex.feedables for ex in runners + trainers])  # type: ignore

    if log_directory:
        log("Initializing TensorBoard summary writer.")
        tb_writer = tf.summary.FileWriter(
            log_directory, tf_manager.sessions[0].graph)
        log("TensorBoard writer initialized.")

    log("Starting training")
    last_log_time = time.process_time()
    last_val_time = time.process_time()
    interrupt = None
    try:
        for epoch_n in range(1, epochs + 1):
            log_print("")
            log("Epoch {} begins".format(epoch_n), color="red")

            train_batches = train_dataset.batches(batching_scheme)

            if epoch_n == 1 and train_start_offset:
                if train_dataset.shuffled and not train_dataset.lazy:
                    warn("Not skipping training instances with shuffled "
                         "non-lazy dataset")
                else:
                    _skip_lines(train_start_offset, train_batches)

            for batch_n, batch in enumerate(train_batches):
                step += 1
                seen_instances += len(batch)

                if log_timer(step, last_log_time):
                    trainer_result = tf_manager.execute(
                        batch, feedables, trainers, train=True, summaries=True)
                    train_results, train_outputs = run_on_dataset(
                        tf_manager, runners, batch, postprocess,
                        write_out=False,
                        batching_scheme=runners_batching_scheme)
                    # ensure train outputs are iterable more than once
                    train_outputs = {
                        k: list(v) for k, v in train_outputs.items()}
                    train_evaluation = evaluation(
                        evaluators, batch, runners, train_results,
                        train_outputs)

                    _log_continuous_evaluation(
                        tb_writer, main_metric, train_evaluation,
                        seen_instances, epoch_n, epochs, trainer_result,
                        train=True)
                    last_log_time = time.process_time()
                else:
                    tf_manager.execute(batch, feedables, trainers, train=True,
                                       summaries=False)

                if val_timer(step, last_val_time):
                    log_print("")
                    val_duration_start = time.process_time()
                    val_examples = 0
                    for val_id, valset in enumerate(val_datasets):
                        val_examples += len(valset)

                        val_results, val_outputs = run_on_dataset(
                            tf_manager, runners, valset,
                            postprocess, write_out=False,
                            batching_scheme=runners_batching_scheme)
                        # ensure val outputs are iterable more than once
                        val_outputs = {k: list(v)
                                       for k, v in val_outputs.items()}
                        val_evaluation = evaluation(
                            evaluators, valset, runners, val_results,
                            val_outputs)

                        valheader = ("Validation (epoch {}, batch number {}):"
                                     .format(epoch_n, batch_n))
                        log(valheader, color="blue")
                        _print_examples(
                            valset, val_outputs, val_preview_input_series,
                            val_preview_output_series,
                            val_preview_num_examples)
                        log_print("")
                        log(valheader, color="blue")

                        # The last validation set is selected to be the main
                        if val_id == len(val_datasets) - 1:
                            this_score = val_evaluation[main_metric]
                            tf_manager.validation_hook(this_score, epoch_n,
                                                       batch_n)

                            if this_score == tf_manager.best_score:
                                best_score_str = colored(
                                    "{:.4g}".format(tf_manager.best_score),
                                    attrs=["bold"])

                                # store also graph parts
                                rnrs = runners + trainers  # type: ignore
                                # TODO: refactor trainers/runners so that they
                                # have the same API predecessor
                                parameterizeds = set.union(
                                    *[rnr.parameterizeds
                                      for rnr in rnrs])
                                for coder in parameterizeds:
                                    for session in tf_manager.sessions:
                                        coder.save(session)
                            else:
                                best_score_str = "{:.4g}".format(
                                    tf_manager.best_score)

                            log("best {} on validation: {} (in epoch {}, "
                                "after batch number {})"
                                .format(main_metric, best_score_str,
                                        tf_manager.best_score_epoch,
                                        tf_manager.best_score_batch),
                                color="blue")

                        v_name = valset.name if len(val_datasets) > 1 else None
                        _log_continuous_evaluation(
                            tb_writer, main_metric, val_evaluation,
                            seen_instances, epoch_n, epochs, val_results,
                            train=False, dataset_name=v_name)

                    # how long was the training between validations
                    training_duration = val_duration_start - last_val_time
                    val_duration = time.process_time() - val_duration_start

                    # the training should take at least twice the time of val.
                    steptime = (training_duration
                                / (seen_instances - last_seen_instances))
                    valtime = val_duration / val_examples
                    last_seen_instances = seen_instances
                    log("Validation time: {:.2f}s, inter-validation: {:.2f}s, "
                        "per-instance (train): {:.2f}s, per-instance (val): "
                        "{:.2f}s".format(val_duration, training_duration,
                                         steptime, valtime), color="blue")
                    if training_duration < 2 * val_duration:
                        notice("Validation period setting is inefficient.")

                    log_print("")
                    last_val_time = time.process_time()

    except KeyboardInterrupt as ex:
        interrupt = ex

    log("Training finished. Maximum {} on validation data: {:.4g}, epoch {}"
        .format(main_metric, tf_manager.best_score,
                tf_manager.best_score_epoch))

    log("Saving final variables in {}".format(final_variables))
    tf_manager.save(final_variables)

    if test_datasets:
        tf_manager.restore_best_vars()

        for dataset in test_datasets:
            test_results, test_outputs = run_on_dataset(
                tf_manager, runners, dataset, postprocess,
                write_out=True, batching_scheme=runners_batching_scheme)
            # ensure test outputs are iterable more than once
            test_outputs = {k: list(v) for k, v in test_outputs.items()}
            eval_result = evaluation(evaluators, dataset, runners,
                                     test_results, test_outputs)
            print_final_evaluation(dataset.name, eval_result)

    log("Finished.")

    if interrupt is not None:
        raise interrupt  # pylint: disable=raising-bad-type


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
                   dataset: Dataset,
                   postprocess: Postprocess,
                   batching_scheme: BatchingScheme,
                   write_out: bool = False,
                   log_progress: int = 0) -> Tuple[
                       List[ExecutionResult], Dict[str, List[Any]]]:
    """Apply the model on a dataset and optionally write outputs to files.

    This function processes the dataset in batches and optionally prints out
    the execution progress.

    Args:
        tf_manager: TensorFlow manager with initialized sessions.
        runners: A function that runs the code
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
    contains_targets = all(dataset.has_series(runner.decoder_data_id)
                           for runner in runners
                           if runner.decoder_data_id is not None)

    last_log_time = time.process_time()
    batch_results = [[] for _ in runners]  # type: List[List[ExecutionResult]]

    feedables = set.union(*[runner.feedables for runner in runners])

    processed_examples = 0
    for batch in dataset.batches(batching_scheme):
        if 0 < log_progress < time.process_time() - last_log_time:
            log("Processed {} examples.".format(processed_examples))
            last_log_time = time.process_time()

        execution_results = tf_manager.execute(
            batch, feedables, runners, compute_losses=contains_targets)
        processed_examples += len(batch)

        for script_list, ex_result in zip(batch_results, execution_results):
            script_list.append(ex_result)

    # Transpose runner interim results.
    all_results = [reduce_execution_results(res) for res in batch_results]

    # Convert execution results to dictionary.
    result_data = {runner.output_series: result.outputs
                   for runner, result in zip(runners, all_results)}

    # Run dataset-level postprocessing.
    if postprocess is not None:
        for series_name, postprocessor in postprocess:
            postprocessed = postprocessor(dataset, result_data)
            if not hasattr(postprocessed, "__len__"):
                postprocessed = list(postprocessed)

            result_data[series_name] = postprocessed

    # Check output series lengths.
    for series_id, data in result_data.items():
        if len(data) != len(dataset):
            warn("Output '{}' for dataset '{}' has length {}, but "
                 "len(dataset) == {}".format(series_id, dataset.name,
                                             len(data), len(dataset)))

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

    return all_results, result_data


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
        if (not batch.has_series(reference_id)
                or hypothesis_id not in result_data):
            continue

        desired_output = list(batch.get_series(reference_id))
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


def print_final_evaluation(name: str, eval_result: Evaluation) -> None:
    """Print final evaluation from a test dataset."""
    line_len = 22
    log("Evaluating model on '{}'".format(name))

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


def _print_examples(dataset: Dataset,
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

    source_series_names = [s for s in dataset.series if s not in outputs]
    target_series_names = [s for s in dataset.series if s in outputs]
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
    target_series = {series_id: list(dataset.get_series(series_id))
                     for series_id in target_series_names}
    source_series = {series_id: list(dataset.get_series(series_id))
                     for series_id in source_series_names}

    if not dataset.lazy:
        num_examples = min(len(dataset), num_examples)

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


def _log_model_variables(var_list: List[tf.Variable] = None) -> None:
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
