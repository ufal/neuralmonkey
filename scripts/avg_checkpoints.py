#!/usr/bin/env python3
"""Compute the average of each variable in a list of checkpoint files.

Given a list of model checkpoints, it generates a new checkpoint with
parameters which are the arithmetic average of them.

Based on a script from Tensor2Tensor:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/avg_checkpoints.py
"""

import argparse
import os
import re

import numpy as np
import tensorflow as tf

from neuralmonkey.logging import log


IGNORED_PATTERNS = ["global_step"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=str, nargs="+",
                        help="Space-separated list of checkpoints to average.")
    parser.add_argument("output_path", type=str,
                        help="Path to output the averaged checkpoint to.")
    args = parser.parse_args()

    non_existing_chckpoints = []
    for ckpt in args.checkpoints:
        if not os.path.exists("{}.index".format(ckpt)):
            non_existing_chckpoints.append(ckpt)
    if non_existing_chckpoints:
        raise ValueError(
            "Provided checkpoints do not exist: {}".format(
                ", ".join(non_existing_chckpoints)))

    # Read variables from all checkpoints and average them.
    log("Getting list of variables")
    var_list = tf.contrib.framework.list_variables(args.checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if not any(re.match(pat, name) for pat in IGNORED_PATTERNS):
            var_values[name] = np.zeros(shape)
    for checkpoint in args.checkpoints:
        log("Reading from checkpoint {}".format(checkpoint))
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor
    for name in var_values:    # Average.
        var_values[name] /= len(args.checkpoints)

    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(
            0, name="global_step", trainable=False, dtype=tf.int64)
    saver = tf.train.Saver()

    # Build a model only with variables, set them to the average values.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               var_values.items()):
            sess.run(assign_op, {p: value})
        saver.save(sess, args.output_path, global_step=global_step)

    log("Averaged checkpoints saved in {}".format(args.output_path))


if __name__ == "__main__":
    main()
