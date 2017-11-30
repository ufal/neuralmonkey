#!/usr/bin/env python3
import argparse
import collections
import glob
import os
import sys
import shutil
import tempfile
import time
from typing import (Any, Callable, Dict, List, Tuple, Optional, Union,
                    Iterable, Set)

import numpy as np
import tensorflow as tf
from termcolor import colored
from typeguard import check_argument_types

from neuralmonkey import logging
from neuralmonkey.config.builder import ClassSymbol, ObjectRef
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.dataset import Dataset, LazyDataset
from neuralmonkey.decorators import tensor
from neuralmonkey.evaluators.accuracy import AccuracySeqLevel
from neuralmonkey.learning_utils import (
    EvalConfiguration, Evaluation, Postprocess, SeriesName, evaluation,
    run_on_dataset, print_final_evaluation, _check_series_collisions,
    _is_logging_time, _log_model_variables, _log_continuous_evaluation,
    _print_examples, _resolve_period)
from neuralmonkey.logging import log, log_print, warn, notice
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import TemporalStatefulWithOutput, Stateful
from neuralmonkey.runners.base_runner import BaseRunner, ExecutionResult
from neuralmonkey.runners.representation_runner import RepresentationRunner
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.tf_utils import gpu_memusage
from neuralmonkey.trainers.generic_trainer import GenericTrainer


class DummyEncoder(ModelPart, TemporalStatefulWithOutput):

    def __init__(self,
                 name: str,
                 data_id: str,
                 output_size: int,
                 batch_size: int = 1,
                 initializer: Callable = None,
                 norm: float = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a new instance of the encoder."""
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        TemporalStatefulWithOutput.__init__(self)

        self.output_size = output_size
        self.batch_size = batch_size
        self.initializer = initializer
        self.norm = norm
        self.data_id = data_id  # needed for RepresentationRunner
        self.init_value = None

    @tensor
    def variable(self) -> tf.Variable:
        return tf.get_variable("encoder_output",
                               [self.batch_size, self.output_size],
                               initializer=self.initializer)

    @tensor
    def output(self) -> tf.Tensor:
        if self.norm is not None:
            return tf.nn.l2_normalize(self.variable, dim=1) * self.norm
        else:
            return self.variable

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return TemporalStatefulWithOutput.temporal_states(self)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return TemporalStatefulWithOutput.temporal_mask(self)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        del dataset
        del train
        return {}


class GaussianRegularizer(ModelPart):

    def __init__(self,
                 name: str,
                 encoder: Stateful,
                 mean_path: str,
                 cov_path: str) -> None:
        ModelPart.__init__(self, name)

        self.z = encoder.output
        self.mean = np.load(mean_path)
        cov_matrix = np.load(cov_path)
        self.prec_matrix = np.linalg.inv(cov_matrix)

    @tensor
    def cost(self) -> tf.Tensor:
        return tf.reduce_sum((self.z @ self.prec_matrix) * self.z, axis=1)


    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {}


def training_loop(tf_manager: TensorFlowManager,
                  iterations: int,
                  trainer: GenericTrainer,
                  log_directory: str,
                  evaluators: EvalConfiguration,
                  runners: List[BaseRunner],
                  train_batch: Dataset,
                  logging_period: Union[str, int] = 20,
                  long_logging_period: Union[str, int] = 500,
                  val_preview_input_series: Optional[List[str]] = None,
                  val_preview_output_series: Optional[List[str]] = None,
                  val_preview_num_examples: int = 15,
                  initial_variables: Optional[Union[str, List[str]]] = None,
                  postprocess: Postprocess = None) -> None:
    check_argument_types()

    log_period_batch, log_period_time = _resolve_period(logging_period)
    long_log_period_batch, long_log_period_time = _resolve_period(
        long_logging_period)

    _check_series_collisions(runners, postprocess)

    _log_model_variables(var_list=trainer.var_list)

    if tf_manager.report_gpu_memory_consumption:
        log("GPU memory usage: {}".format(gpu_memusage()))

    batch_size = len(train_batch)

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
    epoch_n = 1
    epochs = 1

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
    last_long_log_time = time.process_time()
    interrupt = None
    try:
        for step in range(1, iterations + 1):
            is_log_time = _is_logging_time(
                step, log_period_batch, last_log_time, log_period_time)
            is_long_log_time = _is_logging_time(
                step, long_log_period_batch, last_long_log_time,
                long_log_period_time)
            if is_log_time or is_long_log_time:
                trainer_result = tf_manager.execute(
                    train_batch, [trainer], train=True,
                    summaries=True)
                train_results, train_outputs = run_on_dataset(
                    tf_manager, runners, train_batch,
                    postprocess, write_out=False,
                    batch_size=batch_size)
                # ensure train outputs are iterable more than once
                train_outputs = {k: list(v) for k, v
                                 in train_outputs.items()}
                train_evaluation = evaluation(
                    evaluators, train_batch, runners,
                    train_results, train_outputs)

                this_score = train_evaluation[main_metric]
                tf_manager.validation_hook(this_score, epoch_n, step)

                _log_continuous_evaluation(
                    tb_writer, tf_manager, main_metric, train_evaluation,
                    step, epoch_n, epochs, trainer_result,
                    train=True)
                last_log_time = time.process_time()

                if is_long_log_time:
                    valheader = ("Examples (iteration {}):".format(step))
                    log(valheader, color="blue")
                    _print_examples(
                        train_batch, train_outputs, val_preview_input_series,
                        val_preview_output_series,
                        batch_size)
                    log_print("")
                    log(valheader, color="blue")
                    last_long_log_time = time.process_time()


                if this_score == tf_manager.best_score:
                    best_score_str = colored(
                        "{:.4g}".format(tf_manager.best_score),
                        attrs=["bold"])
                else:
                    best_score_str = "{:.4g}".format(
                        tf_manager.best_score)

                log("best {} on validation: {} (after iteration {})"
                    .format(main_metric, best_score_str,
                            tf_manager.best_score_batch),
                    color="blue")
            else:
                tf_manager.execute(train_batch, [trainer],
                                   train=True, summaries=False)

    except KeyboardInterrupt as ex:
        interrupt = ex

    log("Training finished. Maximum {}: {:.4g}, iteration {}"
        .format(main_metric, tf_manager.best_score,
                tf_manager.best_score_batch))

    tf_manager.restore_best_vars()

    test_results, test_outputs = run_on_dataset(
        tf_manager, runners, train_batch, postprocess,
        write_out=True, batch_size=batch_size)
    # ensure test outputs are iterable more than once
    test_outputs = {k: list(v) for k, v in test_outputs.items()}
    eval_result = evaluation(evaluators, train_batch, runners,
                             test_results, test_outputs)
    print_final_evaluation(train_batch.name, eval_result)

    log("Finished.")

    if interrupt is not None:
        raise interrupt  # pylint: disable=raising-bad-type


def _make_class_dict(clazz: Callable, **kwargs) -> Dict:
    return collections.OrderedDict([
        ("class", ClassSymbol("{}.{}".format(clazz.__module__,
                                             clazz.__qualname__))),
        *kwargs.items()
    ])


def _make_config() -> Configuration:
    config = Configuration()
    config.add_argument("output", required=True)
    config.add_argument("dataset", required=True)
    config.add_argument("num_repeat", default=4)
    config.add_argument("encoder_output_size", required=True)
    config.add_argument("num_iterations", default=500)
    config.add_argument("decoder", required=True)
    config.add_argument("encoder", required=True)
    config.add_argument("logging_period", default=5)
    config.add_argument("long_logging_period", default=15)
    config.add_argument("preview_series")
    config.add_argument("init_with_series")
    config.add_argument("norm")
    config.add_argument("initializer")
    config.add_argument("trainer")
    config.add_argument("runners")
    config.add_argument("postprocess")
    config.add_argument("evaluation")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_config", metavar="MODEL-INI-FILE",
                        help="the configuration file of the model")
    parser.add_argument("config", metavar="INI-FILE",
                        help="the configuration file for the reverse decoder")
    args = parser.parse_args()

    config = _make_config()
    config.load_file(args.model_config)

    model_dir = config.args.output

    config.load_file(args.config)
    cfg_dict = config.config_dict

    output_dir = config.args.output
    os.makedirs(output_dir)

    # Replace original encoder with our dummy encoder
    num_repeat = config.args.num_repeat
    decoder_name = config.args.decoder.name
    orig_encoder_name = config.args.encoder.name
    cfg_dict["dummy_encoder"] = _make_class_dict(
        DummyEncoder, name="dummy_encoder",
        output_size=config.args.encoder_output_size,
        batch_size=num_repeat,
        initializer=config.args.initializer,
        norm=config.args.norm,
        data_id=cfg_dict[orig_encoder_name]["data_id"])
    dummy_encoder_ref = ObjectRef("dummy_encoder")
    cfg_dict[decoder_name].update(dict(
        load_checkpoint=os.path.join(model_dir, "variables.data"),
        encoders=[dummy_encoder_ref],
        dropout_keep_prob=1.,
    ))

    with tempfile.TemporaryDirectory(prefix="reverse_decoder") as tmp_dir:
        tmp_output_dir = os.path.join(tmp_dir, "output")
        cfg_dict["main"]["output"] = tmp_output_dir
        os.mkdir(tmp_output_dir)

        config.build_model()

        full_dataset = config.model.dataset
        dummy_encoder = dummy_encoder_ref.target
        init_value = tf.placeholder(shape=dummy_encoder.variable.shape,
                                    dtype=dummy_encoder.variable.dtype)
        init_op = tf.assign(dummy_encoder.variable, init_value)

        for i in range(len(full_dataset)):
            # Clean up output directory
            shutil.rmtree(tmp_output_dir)
            os.mkdir(tmp_output_dir)
            shutil.copymode(output_dir, tmp_output_dir)

            logging.Logging.set_log_file(
                os.path.join(tmp_output_dir, "experiment.log"))

            tf_manager = TensorFlowManager(num_sessions=1, num_threads=4)
            tf_manager.init_saving(
                os.path.join(tmp_output_dir, "variables.data"))

            # Get i-th sentence from dataset and repeat it num_repeat times
            dataset = full_dataset.subset(i, 1)
            dataset = Dataset("{}x{}".format(dataset.name, num_repeat),
                              {key: dataset.get_series(key) * num_repeat
                               for key in dataset.series_ids},
                              dataset.series_outputs)

            if config.model.init_with_series:
                init_data = dataset.get_series(config.model.init_with_series)
                tf_manager.sessions[0].run(init_op,
                                           feed_dict={init_value: init_data})

            # Train and evaluate
            try:
                training_loop(
                    tf_manager=tf_manager,
                    iterations=config.model.num_iterations,
                    trainer=config.model.trainer,
                    log_directory=config.model.output,
                    evaluators=config.model.evaluation,
                    runners=config.model.runners,
                    train_batch=dataset,
                    logging_period=config.model.logging_period,
                    long_logging_period=config.model.long_logging_period,
                    val_preview_output_series=config.model.preview_series,
                    val_preview_input_series=config.model.preview_series,
                    postprocess=config.model.postprocess)
            finally:
                tf_manager.sessions[0].close()
                for fname in glob.glob(os.path.join(tmp_output_dir,
                                                    "variables.data*")):
                    os.remove(fname)
                sentence_output_dir = os.path.join(output_dir,
                                                   "s{:08}".format(i))
                shutil.copytree(tmp_output_dir, sentence_output_dir)

if __name__ == "__main__":
    sys.exit(main())
