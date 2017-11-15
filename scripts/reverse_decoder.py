#!/usr/bin/env python3
import argparse
import collections
import glob
import os
import sys
import shutil
import tempfile
from typing import Callable, Dict

import tensorflow as tf

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.config.builder import ClassSymbol, ObjectRef
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.evaluators.accuracy import AccuracySeqLevel
from neuralmonkey.learning_utils import training_loop
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import TemporalStatefulWithOutput
from neuralmonkey.runners.representation_runner import RepresentationRunner
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey import logging


class DummyEncoder(ModelPart, TemporalStatefulWithOutput):

    def __init__(self,
                 name: str,
                 data_id: str,
                 output_size: int,
                 batch_size: int = 1,
                 initializer: Callable = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a new instance of the encoder."""
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        TemporalStatefulWithOutput.__init__(self)

        self.output_size = output_size
        self.batch_size = batch_size
        self.initializer = initializer
        self.data_id = data_id  # needed for RepresentationRunner
        self.init_value = None

    @tensor
    def output(self) -> tf.Tensor:
        return tf.get_variable("encoder_output",
                               [self.batch_size, self.output_size],
                               initializer=self.initializer)

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


def make_class_dict(clazz: Callable, **kwargs) -> Dict:
    return collections.OrderedDict([
        ("class", ClassSymbol("{}.{}".format(clazz.__module__,
                                             clazz.__qualname__))),
        *kwargs.items()
    ])


def make_config() -> Configuration:
    config = Configuration()
    config.add_argument("output", required=True)
    config.add_argument("dataset", required=True)
    config.add_argument("num_repeat", default=4)
    config.add_argument("encoder_output_size", required=True)
    config.add_argument("num_iterations", default=500)
    config.add_argument("decoder", required=True)
    config.add_argument("encoder", required=True)
    config.add_argument("logging_period", default=5)
    config.add_argument("validation_period", default=15)
    config.add_argument("preview_series")
    config.add_argument("init_with_series")
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

    config = make_config()
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
    cfg_dict["dummy_encoder"] = make_class_dict(
        DummyEncoder, name="dummy_encoder",
        output_size=config.args.encoder_output_size,
        batch_size=num_repeat,
        initializer=config.args.initializer,
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
        init_value = tf.placeholder(shape=dummy_encoder.output.shape,
                                    dtype=dummy_encoder.output.dtype)
        init_op = tf.assign(dummy_encoder.output, init_value)

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
                    epochs=config.model.num_iterations,
                    trainer=config.model.trainer,
                    batch_size=num_repeat,
                    runners_batch_size=num_repeat,
                    log_directory=config.model.output,
                    evaluators=config.model.evaluation,
                    runners=config.model.runners,
                    train_dataset=dataset,
                    val_dataset=dataset,
                    test_datasets=[dataset],
                    logging_period=config.model.logging_period,
                    validation_period=config.model.validation_period,
                    val_preview_output_series=config.model.preview_series,
                    val_preview_input_series=config.model.preview_series,
                    val_preview_num_examples=num_repeat,
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
