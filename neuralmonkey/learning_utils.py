# tests: mypy

from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import os
import numpy as np
import tensorflow as tf
from termcolor import colored

from neuralmonkey.logging import log, log_print
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.runners.base_runner import BaseRunner, ExecutionResult

# pylint: disable=invalid-name
Evaluation = Dict[str, float]
EvalConfiguration = List[Union[Tuple[str, Any], Tuple[str, str, Any]]]


def training_loop(tf_manager: TensorFlowManager,
                  epochs: int,
                  trainer: BaseRunner, # TODO better annotate
                  batch_size: int,
                  train_dataset: Dataset,
                  val_dataset: Dataset,
                  log_directory: str,
                  evaluators: EvalConfiguration,
                  runners: List[BaseRunner],
                  test_datasets: Optional[List[Dataset]]=None,
                  save_n_best_vars: int=1,
                  link_best_vars="/tmp/variables.data.best",
                  vars_prefix="/tmp/variables.data",
                  logging_period: int=20,
                  validation_period: int=500,
                  runners_batch_size: Optional[int]=None,
                  postprocess: Callable=None,
                  minimize_metric: bool=False):

    # TODO finish the list
    """
    Performs the training loop for given graph and data.

    Args:
        tf_manager: TensorFlowManager with initialized sessions.
        epochs: Number of epochs for which the algoritm will learn.
        trainer: The trainer object containg the TensorFlow code for computing
            the loss and optimization operation.
        train_dataset:
        val_dataset:
        postprocess: Function that takes the output sentence as produced by the
            decoder and transforms into tokenized sentence.
        log_directory: Directory where the TensordBoard log will be generated.
            If None, nothing will be done.
        evaluators: List of evaluators. The last evaluator is used as the main.
            An evaluator is a tuple of the name of the generated series, the
            name of the dataset series the generated one is evaluated with and
            the evaluation function. If only one series names is provided, it
            means the generated and dataset series have the same name.
    """

    evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                  for e in evaluators]

    main_metric = "{}/{}".format(evaluators[-1][0], evaluators[-1][-1].name)
    step = 0
    seen_instances = 0

    if save_n_best_vars < 1:
        raise Exception('save_n_best_vars must be greater than zero')

    if save_n_best_vars == 1:
        variables_files = [vars_prefix]
    elif save_n_best_vars > 1:
        variables_files = ['{}.{}'.format(vars_prefix, i)
                           for i in range(save_n_best_vars)]

    if minimize_metric:
        saved_scores = [np.inf for _ in range(save_n_best_vars)]
        best_score = np.inf
    else:
        saved_scores = [-np.inf for _ in range(save_n_best_vars)]
        best_score = -np.inf

    tf_manager.save(variables_files[0])

    if os.path.islink(link_best_vars):
        # if overwriting output dir
        os.unlink(link_best_vars)
    os.symlink(os.path.basename(variables_files[0]), link_best_vars)

    if log_directory:
        log("Initializing TensorBoard summary writer.")
        tb_writer = tf.train.SummaryWriter(log_directory,
                                           tf_manager.sessions[0].graph)
        log("TesorBoard writer initialized.")

    best_score_epoch = 0
    best_score_batch_no = 0

    log("Starting training")
    try:
        for i in range(epochs):
            log_print("")
            log("Epoch {} starts".format(i + 1), color='red')

            train_dataset.shuffle()
            train_batched_datasets = train_dataset.batch_dataset(batch_size)

            for batch_n, batch_dataset in enumerate(train_batched_datasets):

                step += 1
                seen_instances += len(batch_dataset)
                if step % logging_period == logging_period - 1:
                    trainer_result = tf_manager.execute(
                        batch_dataset, [trainer], train=True,
                        summaries=True)
                    train_results, train_outputs = run_on_dataset(
                        tf_manager, runners, batch_dataset,
                        postprocess, write_out=False)
                    train_evaluation = evaluation(
                        evaluators, batch_dataset, runners,
                        train_results, train_outputs)

                    _log_continuons_evaluation(tb_writer, main_metric,
                                               train_evaluation,
                                               seen_instances,
                                               trainer_result,
                                               train=True)
                else:
                    tf_manager.execute(batch_dataset, [trainer],
                                       train=True, summaries=False)

                if step % validation_period == validation_period - 1:
                    val_results, val_outputs = run_on_dataset(
                        tf_manager, runners, val_dataset,
                        postprocess, write_out=False,
                        batch_size=runners_batch_size)
                    val_evaluation = evaluation(
                        evaluators, val_dataset, runners, val_results,
                        val_outputs)

                    this_score = val_evaluation[main_metric]

                    def is_better(score1, score2, minimize):
                        if minimize:
                            return score1 < score2
                        else:
                            return score1 > score2

                    def argworst(scores, minimize):
                        if minimize:
                            return np.argmax(scores)
                        else:
                            return np.argmin(scores)

                    if is_better(this_score, best_score, minimize_metric):
                        best_score = this_score
                        best_score_epoch = i + 1
                        best_score_batch_no = batch_n

                    worst_index = argworst(saved_scores, minimize_metric)
                    worst_score = saved_scores[worst_index]

                    if is_better(this_score, worst_score, minimize_metric):
                        # we need to save this score instead the worst score
                        worst_var_file = variables_files[worst_index]
                        tf_manager.save(worst_var_file)
                        saved_scores[worst_index] = this_score
                        log("Variable file saved in {}".format(worst_var_file))

                        # update symlink
                        if best_score == this_score:
                            os.unlink(link_best_vars)
                            os.symlink(os.path.basename(worst_var_file),
                                       link_best_vars)

                        log("Best scores saved so far: {}".format(saved_scores))

                    log("Validation (epoch {}, batch number {}):"
                        .format(i + 1, batch_n), color='blue')

                    _log_continuons_evaluation(tb_writer, main_metric,
                                               val_evaluation, seen_instances,
                                               val_results, train=False)

                    if this_score == best_score:
                        best_score_str = colored("{:.2f}".format(best_score),
                                                 attrs=['bold'])
                    else:
                        best_score_str = "{:.2f}".format(best_score)

                    log("best {} on validation: {} (in epoch {}, "
                        "after batch number {})"
                        .format(main_metric, best_score_str,
                                best_score_epoch, best_score_batch_no),
                        color='blue')

                    log_print("")
                    _print_examples(val_dataset, val_outputs)

    except KeyboardInterrupt:
        log("Training interrupted by user.")


    log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}"
        .format(main_metric, best_score, best_score_epoch))

    if test_datasets and os.path.islink(link_best_vars):
        tf_manager.restore(link_best_vars)

    for dataset in test_datasets:
        test_results, test_outputs = run_on_dataset(
            tf_manager, runners, dataset, postprocess,
            write_out=True, batch_size=runners_batch_size)
        eval_result = evaluation(evaluators, dataset, runners,
                                 test_results, test_outputs)
        print_final_evaluation(dataset.name, eval_result)

    log("Finished.")


def run_on_dataset(tf_manager: TensorFlowManager,
                   runners: List[BaseRunner],
                   dataset: Dataset,
                   postprocess: Callable,
                   write_out: bool=False,
                   batch_size: Optional[int]=None) \
                                                -> Tuple[List[ExecutionResult],
                                                         Dict[str, List[Any]]]:
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

    contains_targets = all(dataset.has_series(runner.output_series)
                           for runner in runners)

    all_results = tf_manager.execute(dataset, runners,
                                     train=contains_targets,
                                     batch_size=batch_size)

    result_data_raw = {runner.output_series: result.outputs
                       for runner, result in zip(runners, all_results)}

    if postprocess is not None:
        result_data = postprocess(dataset, result_data_raw)
    else:
        result_data = result_data_raw

    if write_out:
        for series_id, data in result_data.items():
            if series_id in dataset.series_outputs:
                path = dataset.series_outputs[series_id]
                if isinstance(data, np.ndarray):
                    np.save(path, data)
                    log('Result saved as numpy array to "{}"'.format(path))
                else:
                    with open(path, 'w') as f_out:
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


def _log_continuons_evaluation(tb_writer: tf.train.SummaryWriter,
                               main_metric: str,
                               eval_result: Evaluation,
                               seen_instances: int,
                               execution_results: List[ExecutionResult],
                               train: bool=False) -> None:
    """Log the evaluation results and the TensorBoard summaries."""

    color, prefix = ("yellow", "train") if train else ("blue", "val")

    eval_string = _format_evaluation_line(eval_result, main_metric)
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
    eval_string = "    ".join("{}: {:.2f}".format(name, value)
                              for name, value in evaluation_res.items()
                              if name != main_metric)

    eval_string += colored(
        "    {}: {:.2f}".format(main_metric,
                                evaluation_res[main_metric]),
        attrs=['bold'])

    return eval_string


def print_final_evaluation(name: str, eval_result: Evaluation) -> None:
    """Print final evaluation from a test dataset."""
    line_len = 22
    log("Evaluating model on \"{}\"".format(name))

    for name, value in eval_result.items():
        space = "".join([" " for _ in range(line_len - len(name))])
        log("... {}:{} {:.4f}".format(name, space, value))

    log_print("")


def _data_item_to_str(item: Any) -> str:
    if isinstance(item, list):
        return " ".join(item)
    elif isinstance(item, str):
        return item
    elif isinstance(item, np.ndarray):
        return "numpy tensor"
    else:
        return str(item)


def _print_examples(dataset: Dataset,
                    outputs: Dict[str, List[Any]],
                    num_examples=15) -> None:
    """Print examples of the model output."""
    log_print(colored("Examples:", attrs=['bold']))

    # for further indexing we need to make sure, all relevant
    # dataset series are lists
    target_series = {series_id: list(dataset.get_series(series_id))
                     for series_id in outputs.keys()
                     if dataset.has_series(series_id)}
    source_series = {series_id: list(dataset.get_series(series_id))
                     for series_id in dataset.series_ids
                     if series_id not in outputs}

    for i in range(num_examples):
        log_print(colored("  [{}]".format(i + 1), color='magenta',
                          attrs=['bold']))

        def print_line(prefix, color, content):
            colored_prefix = colored(prefix, color=color)
            formated = _data_item_to_str(content)
            log_print("  {}: {}".format(colored_prefix, formated))

        for series_id, data in sorted(source_series.items(),
                                      key=lambda x: x[0]):
            print_line(series_id, 'yellow', data[i])

        for series_id, data in sorted(outputs.items(),
                                      key=lambda x: x[0]):
            model_output = data[i]
            print_line(series_id, 'magenta', model_output)

            if series_id in target_series:
                desired_output = target_series[series_id][i]
                print_line(series_id + " (ref)", "red", desired_output)
        log_print("")
