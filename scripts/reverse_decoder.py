#!/usr/bin/env python3
import argparse
import collections
import glob
import os
import sys
import shutil
import tempfile

import tensorflow as tf
from typing import Any, Callable, Dict

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.config.builder import ClassSymbol
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.learning_utils import training_loop
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import TemporalStatefulWithOutput
from neuralmonkey.processors.german import GermanPreprocessor
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

    @tensor
    def output(self) -> tf.Tensor:
        return tf.get_variable('encoder_output',
                               [self.batch_size, self.output_size],
                               initializer=self.initializer)

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return TemporalStatefulWithOutput.temporal_states(self)

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return TemporalStatefulWithOutput.temporal_mask(self)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary for the encoder object.

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run.
        """
        del dataset
        del train
        return {}


def make_class_dict(clazz: Callable, **kwargs) -> Dict:
    return collections.OrderedDict([
        ('class', ClassSymbol('{}.{}'.format(clazz.__module__,
                                             clazz.__qualname__))),
        *kwargs.items()
    ])


def make_model_config() -> Configuration:
    config = Configuration()
    config.ignore_argument('name')
    config.ignore_argument('train_dataset')
    config.ignore_argument('val_dataset')
    config.ignore_argument('test_datasets')
    config.ignore_argument('logging_period')
    config.ignore_argument('validation_period')
    config.ignore_argument('epochs')
    config.ignore_argument('val_preview_input_series')
    config.ignore_argument('val_preview_output_series')
    config.ignore_argument('val_preview_num_examples')
    config.ignore_argument('visualize_embeddings')
    config.ignore_argument('minimize')
    config.ignore_argument('random_seed')
    config.ignore_argument('save_n_best')
    config.ignore_argument('overwrite_output_dir')
    config.ignore_argument('batch_size')
    config.add_argument('evaluation')
    config.add_argument('runners')
    config.add_argument('output')
    config.add_argument('initial_variables')
    config.add_argument('tf_manager')
    config.add_argument('trainer')
    config.add_argument('postprocess')
    return config


def make_config() -> Configuration:
    config = Configuration()
    config.add_argument('output', required=True)
    config.add_argument('dataset', required=True)
    config.add_argument('num_repeat', default=4)
    config.add_argument('encoder_output_size', required=True)
    config.add_argument('num_iterations', default=500)
    config.add_argument('decoder_name', default='decoder')
    config.add_argument('logging_period', default=5)
    config.add_argument('validation_period', default=15)
    config.add_argument('preview_series')
    config.add_argument('initializer')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_config', metavar='MODEL-INI-FILE',
                        help='the configuration file of the model')
    parser.add_argument('config', metavar='INI-FILE',
                        help='the configuration file for the reverse decoder')
    args = parser.parse_args()

    config = make_config()
    config.load_file(args.config)
    config.build_model()

    os.makedirs(config.model.output)

    model_config = make_model_config()
    model_config.load_file(args.model_config)
    model_dict = model_config.config_dict

    model_dir = model_dict['main']['output']

    # Replace original encoder with our dummy encoder
    num_repeat = config.model.num_repeat
    orig_encoder_name = (model_dict[config.model.decoder_name]['encoders'][0]
                         .replace('object:', ''))
    model_dict['dummy_encoder'] = make_class_dict(
        DummyEncoder, name='dummy_encoder',
        output_size=config.model.encoder_output_size,
        batch_size=num_repeat,
        initializer=config.model.initializer,
        data_id=model_dict[orig_encoder_name]['data_id'])
    model_dict[config.model.decoder_name].update(dict(
        load_checkpoint=os.path.join(model_dir, 'variables.data'),
        encoders=['object:dummy_encoder'],
        dropout_keep_prob=1.,
    ))

    # Disable regularization and decoder training
    trainer_name = model_dict['main']['trainer'].replace('object:', '')
    model_dict[trainer_name].update(dict(
        l1_weight=0.,
        l2_weight=0.,
        var_scopes=['dummy_encoder'],
    ))

    # Add a runner that writes the learned representation to a file
    model_dict['main']['runners'].append('object:representation_runner')
    model_dict['representation_runner'] = make_class_dict(
        RepresentationRunner,
        output_series='encoded',
        encoder='object:dummy_encoder')

    with tempfile.TemporaryDirectory(prefix='reverse_decoder') as tmp_dir:
        tmp_output_dir = os.path.join(tmp_dir, 'output')
        model_dict['main']['output'] = tmp_output_dir
        os.mkdir(tmp_output_dir)

        model_config.build_model()

        full_dataset = config.model.dataset

        for i in range(len(full_dataset)):
            # Clean up output directory
            shutil.rmtree(tmp_output_dir)
            os.mkdir(tmp_output_dir)
            shutil.copymode(config.model.output, tmp_output_dir)

            logging.Logging.set_log_file(
                os.path.join(tmp_output_dir, 'experiment.log'))

            tf_manager = TensorFlowManager(num_sessions=1, num_threads=4)
            tf_manager.init_saving(
                os.path.join(tmp_output_dir, 'variables.data'))

            # Get i-th sentence from dataset and repeat it num_repeat times
            dataset = full_dataset.subset(i, 1)
            dataset = Dataset('{}x{}'.format(dataset.name, num_repeat),
                              {key: dataset.get_series(key) * num_repeat
                               for key in dataset.series_ids},
                              dataset.series_outputs)

            # Train and evaluate
            try:
                training_loop(
                    tf_manager=tf_manager,
                    epochs=config.model.num_iterations,
                    trainer=model_config.model.trainer,
                    batch_size=num_repeat,
                    runners_batch_size=num_repeat,
                    log_directory=model_config.model.output,
                    evaluators=model_config.model.evaluation,
                    runners=model_config.model.runners,
                    train_dataset=dataset,
                    val_dataset=dataset,
                    test_datasets=[dataset],
                    logging_period=config.model.logging_period,
                    validation_period=config.model.validation_period,
                    val_preview_output_series=config.model.preview_series,
                    val_preview_input_series=config.model.preview_series,
                    val_preview_num_examples=num_repeat,
                    postprocess=model_config.model.postprocess,
                    initial_variables=model_config.model.initial_variables)
            finally:
                tf_manager.sessions[0].close()
                for fname in glob.glob(os.path.join(tmp_output_dir, 'variables.data*')):
                    os.remove(fname)
                output_dir = os.path.join(config.model.output, 's{:08}'.format(i))
                shutil.copytree(tmp_output_dir, output_dir)

if __name__ == '__main__':
    sys.exit(main())
