# pylint: disable=too-many-lines
# TODO de-clutter this file!

from typing import Any, Callable, Dict, List, Tuple, Optional, Union, Iterable
import time
import re
from datetime import timedelta
import numpy as np
import tensorflow as tf
from termcolor import colored

from neuralmonkey.logging import log, log_print, warn, notice
from neuralmonkey.dataset import Dataset, LazyDataset
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.runners.base_runner import BaseRunner, ExecutionResult
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.tf_utils import gpu_memusage
from typeguard import check_argument_types

# pylint: disable=invalid-name
Evaluation = Dict[str, float]
SeriesName = str
EvalConfiguration = List[Union[Tuple[SeriesName, Any],
                               Tuple[SeriesName, SeriesName, Any]]]
Postprocess = Optional[List[Tuple[SeriesName, Callable]]]
# pylint: enable=invalid-name


# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
# pylint: disable=too-many-statements, too-many-nested-blocks
def training_loop(tf_manager: TensorFlowManager,
                  epochs: int,
                  trainer: GenericTrainer,  # TODO better annotate
                  batch_size: int,
                  log_directory: str,
                  evaluators: EvalConfiguration,
                  runners: List[BaseRunner],
                  train_dataset: Dataset,
                  val_dataset: Union[Dataset, List[Dataset]],
                  test_datasets: Optional[List[Dataset]] = None,
                  logging_period: Union[str, int] = 20,
                  validation_period: Union[str, int] = 500,
                  val_preview_input_series: Optional[List[str]] = None,
                  val_preview_output_series: Optional[List[str]] = None,
                  val_preview_num_examples: int = 15,
                  train_start_offset: int = 0,
                  runners_batch_size: Optional[int] = None,
                  initial_variables: Optional[Union[str, List[str]]] = None,
                  postprocess: Postprocess = None) -> None:
    """
    Performs the training loop for given graph and data.
    Args:
        tf_manager: TensorFlowManager with initialized sessions.
        epochs: Number of epochs for which the algoritm will learn.
        trainer: The trainer object containg the TensorFlow code for computing
            the loss and optimization operation.
        batch_size: number of examples in one mini-batch
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
        runners_batch_size: batch size of runners. It is the same as batch_size
            if not specified
        initial_variables: variables used for initialization, for example for
            continuation of training
        postprocess: A function which takes the dataset with its output series
            and generates additional series from them.
    """
    check_argument_types()

    if isinstance(val_dataset, Dataset):
        val_datasets = [val_dataset]
    else:
        val_datasets = val_dataset

    log_period_batch, log_period_time = _resolve_period(logging_period)
    val_period_batch, val_period_time = _resolve_period(validation_period)

    _check_series_collisions(runners, postprocess)

    _log_model_variables()

    if tf_manager.report_gpu_memory_consumption:
        log("GPU memory usage: {}".format(gpu_memusage()))

    if runners_batch_size is None:
        runners_batch_size = batch_size

    evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                  for e in evaluators]

    if evaluators:
        main_metric = "{}/{}".format(evaluators[-1][0],
                                     evaluators[-1][-1].name)
    else:
        main_metric = "{}/{}".format(runners[-1].decoder_data_id,
                                     runners[-1].loss_names[0])

        if not tf_manager.minimize_metric:
            raise ValueError("minimize_metric must be set to True in "
                             "TensorFlowManager when using loss as "
                             "the main metric")

    step = 0
    seen_instances = 0
    last_seen_instances = 0

    if initial_variables is None:
        # Assume we don't look at coder checkpoints when global
        # initial variables are supplied
        tf_manager.initialize_model_parts(
            runners + [trainer], save=True)  # type: ignore
    else:
        tf_manager.restore(initial_variables)

    if log_directory:
        log("Initializing TensorBoard summary writer.")
        tb_writer = tf.summary.FileWriter(
            log_directory, tf_manager.sessions[0].graph)
        log("TensorBoard writer initialized.")

    log("Starting training")
    last_log_time = time.process_time()
    last_val_time = time.process_time()
    try:
        for epoch_n in range(1, epochs + 1):
            log_print("")
            log("Epoch {} starts".format(epoch_n), color='red')

            train_dataset.shuffle()
            train_batched_datasets = train_dataset.batch_dataset(batch_size)

            if epoch_n == 1 and train_start_offset:
                if not isinstance(train_dataset, LazyDataset):
                    warn("Not skipping training instances with "
                         "shuffled in-memory dataset")
                else:
                    _skip_lines(train_start_offset, train_batched_datasets)

            for batch_n, batch_dataset in enumerate(train_batched_datasets):
                step += 1
                seen_instances += len(batch_dataset)
                if _is_logging_time(step, log_period_batch,
                                    last_log_time, log_period_time):
                    trainer_result = tf_manager.execute(
                        batch_dataset, [trainer], train=True,
                        summaries=True)
                    train_results, train_outputs = run_on_dataset(
                        tf_manager, runners, batch_dataset,
                        postprocess, write_out=False,
                        batch_size=runners_batch_size)
                    # ensure train outputs are iterable more than once
                    train_outputs = {k: list(v) for k, v
                                     in train_outputs.items()}
                    train_evaluation = evaluation(
                        evaluators, batch_dataset, runners,
                        train_results, train_outputs)

                    _log_continuous_evaluation(
                        tb_writer, tf_manager, main_metric, train_evaluation,
                        seen_instances, epoch_n, epochs, trainer_result,
                        train=True)
                    last_log_time = time.process_time()
                else:
                    tf_manager.execute(batch_dataset, [trainer],
                                       train=True, summaries=False)

                if _is_logging_time(step, val_period_batch,
                                    last_val_time, val_period_time):
                    log_print("")
                    val_duration_start = time.process_time()
                    val_examples = 0
                    for val_id, valset in enumerate(val_datasets):
                        val_examples += len(valset)

                        val_results, val_outputs = run_on_dataset(
                            tf_manager, runners, valset,
                            postprocess, write_out=False,
                            batch_size=runners_batch_size)
                        # ensure val outputs are iterable more than once
                        val_outputs = {k: list(v)
                                       for k, v in val_outputs.items()}
                        val_evaluation = evaluation(
                            evaluators, valset, runners, val_results,
                            val_outputs)

                        valheader = ("Validation (epoch {}, batch number {}):"
                                     .format(epoch_n, batch_n))
                        log(valheader, color='blue')
                        _print_examples(
                            valset, val_outputs, val_preview_input_series,
                            val_preview_output_series,
                            val_preview_num_examples)
                        log_print("")
                        log(valheader, color='blue')

                        # The last validation set is selected to be the main
                        if val_id == len(val_datasets) - 1:
                            this_score = val_evaluation[main_metric]
                            tf_manager.validation_hook(this_score, epoch_n,
                                                       batch_n)

                            if this_score == tf_manager.best_score:
                                best_score_str = colored(
                                    "{:.4g}".format(tf_manager.best_score),
                                    attrs=['bold'])

                                # store also graph parts
                                all_coders = set.union(
                                    *[rnr.all_coders
                                      for rnr in runners +
                                      [trainer]])  # type: ignore
                                for coder in all_coders:
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
                                color='blue')

                        _log_continuous_evaluation(
                            tb_writer, tf_manager, main_metric, val_evaluation,
                            seen_instances, epoch_n, epochs, val_results,
                            train=False)

                    # how long was the training between validations
                    training_duration = val_duration_start - last_val_time
                    val_duration = time.process_time() - val_duration_start

                    # the training should take at least twice the time of val.
                    steptime = (training_duration /
                                (seen_instances - last_seen_instances))
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

    except KeyboardInterrupt:
        log("Training interrupted by user.")

    log("Training finished. Maximum {} on validation data: {:.4g}, epoch {}"
        .format(main_metric, tf_manager.best_score,
                tf_manager.best_score_epoch))

    if test_datasets:
        tf_manager.restore_best_vars()

    for dataset in test_datasets:
        test_results, test_outputs = run_on_dataset(
            tf_manager, runners, dataset, postprocess,
            write_out=True, batch_size=runners_batch_size)
        # ensure test outputs are iterable more than once
        test_outputs = {k: list(v) for k, v in test_outputs.items()}
        eval_result = evaluation(evaluators, dataset, runners,
                                 test_results, test_outputs)
        print_final_evaluation(dataset.name, eval_result)

    log("Finished.")


def _is_logging_time(step: int, logging_period_batch: int,
                     last_log_time: float, logging_period_time: int):
    if logging_period_batch is not None:
        return step % logging_period_batch == logging_period_batch - 1
    return last_log_time + logging_period_time < time.process_time()


def _resolve_period(period):
    if isinstance(period, int):
        return period, None
    else:
        regex = re.compile(
            r'((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?'
            r'((?P<seconds>\d+?)s)?')
        parts = regex.match(period)

        if not parts:
            raise ValueError(
                "Validation or logging period have incorrect format. "
                "It should be in format: 3h; 5m; 14s")

        parts = parts.groupdict()
        time_params = {}
        for (name, param) in parts.items():
            if param:
                time_params[name] = int(param)

        delta_seconds = timedelta(**time_params).total_seconds()
        if delta_seconds <= 0:
            raise ValueError(
                "Validation or logging period must be bigger than 0")
        return None, delta_seconds


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
                   write_out: bool = False,
                   batch_size: Optional[int] = None) -> Tuple[
                       List[ExecutionResult], Dict[str, List[Any]]]:
    """Apply the model on a dataset and optionally write outputs to files.

    Args:
        tf_manager: TensorFlow manager with initialized sessions.
        runners: A function that runs the code
        dataset: The dataset on which the model will be executed.
        evaluators: List of evaluators that are used for the model
            evaluation if the target data are provided.
        postprocess: an object to use as postprocessing of the
        write_out: Flag whether the outputs should be printed to a file defined
            in the dataset object.

        extra_fetches: Extra tensors to evaluate for each batch.

    Returns:
        Tuple of resulting sentences/numpy arrays, and evaluation results if
        they are available which are dictionary function -> value.

    """
    contains_targets = all(dataset.has_series(runner.decoder_data_id)
                           for runner in runners
                           if runner.decoder_data_id is not None)

    all_results = tf_manager.execute(dataset, runners,
                                     compute_losses=contains_targets,
                                     batch_size=batch_size)

    result_data = {runner.output_series: result.outputs
                   for runner, result in zip(runners, all_results)}

    if postprocess is not None:
        for series_name, postprocessor in postprocess:
            postprocessed = postprocessor(dataset, result_data)
            if not hasattr(postprocessed, '__len__'):
                postprocessed = list(postprocessed)

            result_data[series_name] = postprocessed

    # check output series lengths
    for series_id, data in result_data.items():
        if len(data) != len(dataset):
            warn("Output '{}' for dataset '{}' has length {}, but "
                 "len(dataset) == {}".format(series_id, dataset.name,
                                             len(data), len(dataset)))

    if write_out:
        for series_id, data in result_data.items():
            if series_id in dataset.series_outputs:
                path = dataset.series_outputs[series_id]
                if isinstance(data, np.ndarray):
                    np.save(path, data)
                    log('Result saved as numpy array to "{}"'.format(path))
                else:
                    with open(path, 'w', encoding='utf-8') as f_out:
                        f_out.writelines(
                            [" ".join(sent) + "\n" for sent in data])
                    log("Result saved as plain text \"{}\"".format(path))
            else:
                log("There is no output file for dataset: {}"
                    .format(dataset.name), color='red')

    return all_results, result_data


def evaluation(evaluators, dataset, runners, execution_results, result_data):
    """Evaluate the model outputs.

    Args:
        evaluators: List of tuples of series and evaluation functions.
        dataset: Dataset against which the evaluation is done.
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
    for generated_id, dataset_id, function in evaluators:
        if (not dataset.has_series(dataset_id) or
                generated_id not in result_data):
            continue

        desired_output = dataset.get_series(dataset_id)
        model_output = result_data[generated_id]
        eval_result["{}/{}".format(generated_id, function.name)] = function(
            model_output, desired_output)

    return eval_result


def _log_continuous_evaluation(tb_writer: tf.summary.FileWriter,
                               tf_manager: TensorFlowManager,
                               main_metric: str,
                               eval_result: Evaluation,
                               seen_instances: int,
                               epoch: int,
                               max_epochs: int,
                               execution_results: List[ExecutionResult],
                               train: bool = False) -> None:
    """Log the evaluation results and the TensorBoard summaries."""

    color, prefix = ("yellow", "train") if train else ("blue", "val")

    if tf_manager.report_gpu_memory_consumption:
        meminfostr = "  " + gpu_memusage()
    else:
        meminfostr = ""

    eval_string = _format_evaluation_line(eval_result, main_metric)
    eval_string = "Epoch {}/{}  Instances {}  {}".format(epoch, max_epochs,
                                                         seen_instances,
                                                         eval_string)
    eval_string = eval_string + meminfostr
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
    """ Format the evaluation metric for stdout with last one bold."""
    eval_string = "    ".join("{}: {:.4g}".format(name, value)
                              for name, value in evaluation_res.items()
                              if name != main_metric)

    eval_string += colored(
        "    {}: {:.4g}".format(main_metric,
                                evaluation_res[main_metric]),
        attrs=['bold'])

    return eval_string


def print_final_evaluation(name: str, eval_result: Evaluation) -> None:
    """Print final evaluation from a test dataset."""
    line_len = 22
    log("Evaluating model on \"{}\"".format(name))

    for eval_name, value in eval_result.items():
        space = "".join([" " for _ in range(line_len - len(eval_name))])
        log("... {}:{} {:.4g}".format(eval_name, space, value))

    log_print("")


def _data_item_to_str(item: Any) -> str:
    if isinstance(item, list):
        return " ".join([str(i) for i in item])

    if isinstance(item, str):
        return item

    if isinstance(item, np.ndarray) and len(item.shape) > 1:
        return "numpy tensor"

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

    source_series_names = [s for s in dataset.series_ids if s not in outputs]
    target_series_names = [s for s in dataset.series_ids if s in outputs]
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

    if not isinstance(dataset, LazyDataset):
        num_examples = min(len(dataset), num_examples)

    for i in range(num_examples):
        log_print(colored("  [{}]".format(i + 1), color="magenta",
                          attrs=["bold"]))

        def print_line(prefix, color, content):
            colored_prefix = colored(prefix, color=color)
            formated = _data_item_to_str(content)
            log_print("  {}: {}".format(colored_prefix, formated))

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
                batched_datasets: Iterable[Dataset]) -> None:
    """Skip training instances from the beginning.

    Arguments:
        start_offset: How many training instances to skip (minimum)
        batched_datasets: From where to throw away batches
    """
    log("Skipping first {} instances in the dataset".format(start_offset))

    skipped_instances = 0
    while skipped_instances < start_offset:
        try:
            skipped_instances += len(next(batched_datasets))  # type: ignore
        except StopIteration:
            raise ValueError("Trying to skip more instances than "
                             "the size of the dataset")

    if skipped_instances > 0:
        log("Skipped {} instances".format(skipped_instances))


def _log_model_variables() -> None:
    trainable_vars = tf.trainable_variables()
    total_params = 0

    logstr = "The model has {} trainable variables:\n\n".format(
        len(trainable_vars))

    logstr += colored(
        "{: ^80}{: ^20}{: ^10}\n".format("Variable name", "Shape", "Size"),
        color="yellow", attrs=["bold"])

    for var in trainable_vars:

        shape = var.get_shape().as_list()
        params_in_var = int(np.prod(shape))
        total_params += params_in_var

        log_entry = "{: <80}{: <20}{: >10}".format(var.name, str(shape),
                                                   params_in_var)
        logstr += "\n{}".format(log_entry)

    logstr += "\n"

    log(logstr)
    log("Total number of all parameters: {}".format(total_params))
