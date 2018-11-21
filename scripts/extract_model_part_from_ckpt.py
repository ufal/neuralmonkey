#!/usr/bin/env python3

"""Extract variables of one model part into a single checkpoint file.

Can be used to load the model part in a different setup."""

import argparse
import os
import tensorflow as tf

from neuralmonkey.logging import log as _log


def log(message: str, color: str = "blue") -> None:
    _log(message, color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orig_checkpoint", metavar="EXPERIMENT-CHECKPOINT",
                        help="path to the original checkpoint")
    parser.add_argument("model_part_name", metavar="MODEL-PART",
                        help="name of the extracted model part")
    parser.add_argument("output_path", metavar="OUTPUT-CHECKPOINT",
                        help="output checkopint file")
    args = parser.parse_args()

    if not os.path.exists("{}.index".format(args.orig_checkpoint)):
        log("Checkpoint '{}' does not exist.".format(
            args.orig_checkpoint), color="red")
        exit(1)

    log("Getting list of variables.")
    var_list = [
        name for name, shape in
        tf.contrib.framework.list_variables(args.orig_checkpoint)
        if name.startswith("{}/".format(args.model_part_name))
        and "Adam" not in name]

    if not var_list:
        log("No variables for model part '{}' in checkpoint '{}'.".format(
            args.model_part_name, args.orig_checkpoint), color="red")
        exit(1)

    log("Reading variables from the checkpoint: {}".format(
        ", ".join(var_list)))

    var_values, var_dtypes = {}, {}
    reader = tf.contrib.framework.load_checkpoint(args.orig_checkpoint)
    for name in var_list:
        tensor = reader.get_tensor(name)
        var_dtypes[name] = tensor.dtype
        var_values[name] = tensor

    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    saver = tf.train.Saver()

    # Build a model only with variables, set them to the average values.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               var_values.items()):
            sess.run(assign_op, {p: value})
        saver.save(sess, os.path.abspath(args.output_path))

    log("Extracted model part saved to {}".format(args.output_path))


if __name__ == "__main__":
    main()
