"""This module contains helper functions that are suppoosed to be called from
the configuration file because calling the functions or the class constructors
directly would be inconvinent or impossible.
"""

import tensorflow as tf


from neuralmonkey.logging import log
import neuralmonkey.vocabulary as vocabulary
import neuralmonkey.dataset as dataset


def deprecated(func):
    def dep_func(*args, **kwargs):
        log("Warning! Use of deprecated function from "
            + "'neuralmonkey.config.utils'. " +
            "Use '{}' instead.".format(func.__module__[13:]
                                       + '.' + func.__name__),
            color='red')
        return func(*args, **kwargs)
    return dep_func


# pylint: disable=invalid-name
# for backwards compatibility
dataset_from_files = deprecated(dataset.load_dataset_from_files)
vocabulary_from_file = deprecated(vocabulary.from_file)
vocabulary_from_bpe = deprecated(vocabulary.from_bpe)
vocabulary_from_dataset = deprecated(vocabulary.from_dataset)
initialize_vocabulary = vocabulary.initialize_vocabulary


def adam_optimizer(learning_rate=1e-4):
    return tf.train.AdamOptimizer(learning_rate)


def adadelta_optimizer(**kwargs):
    return tf.train.AdadeltaOptimizer(**kwargs)


def variable(initial_value=0, trainable=False, **kwargs):
    return tf.Variable(initial_value, trainable, **kwargs)
