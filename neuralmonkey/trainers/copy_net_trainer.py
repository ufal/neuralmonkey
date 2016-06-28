#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tests: mypy

class CopyNetTrainer(object):
    """
    This is a specialized trainer for the CopyNet architecture. In addition to
    standard cross-entropy it adds copy cross-entropy to the loss function in
    such a way that whnenever the network can copy it should copy. Using this
    trainer requires feeding additional information to the computation graph
    (see feed_dict method).
    """
    def __init__(self, decoder, l2_regularization):
        self.decoder = decoder

        self.copy_target_plc = [tf.placeholder(tf.int64, shape=[None]) for _ in decoder.copynet_logits]
        self.copy_w_plc = [tf.placeholder(tf.float32, shape=[None]) for _ in decoder.copynet_logits]

        copy_costs_in_time = [tf.nn.sparse_softmax_cross_entropy_with_logits(l, t) * w \
                for w, l, t in zip(self.copy_w_plc, decoder.copynet_logits, self.copy_target_plc)]

        copy_cost = sum([tf.reduce_sum(c) for c in copy_costs_in_time])
        tf.scalar_summary('train_copy_cost', copy_cost, collections=["summary_train"])
        tf.scalar_summary('val_copy_cost', copy_cost, collections=["summary_val"])

        with tf.variable_scope("l2_regularization"):
            l2_value = sum([tf.reduce_sum(v ** 2) for v in tf.trainable_variables()])
            if l2_regularization > 0:
                l2_cost = l2_regularization * l2_value
            else:
                l2_cost = 0.0

            tf.scalar_summary('train_l2_cost', l2_value, collections=["summary_train"])

        optimizer = tf.train.AdamOptimizer(1e-4)
        gradients = optimizer.compute_gradients(decoder.cost + copy_cost + l2_cost)
        #for (g, v) in gradients:
        #    if g is not None:
        #        tf.histogram_summary('gr_' + v.name, g, collections=["summary_gradients"])
        self.optimize_op = optimizer.apply_gradients(gradients, global_step=decoder.learning_step)
        #self.summary_gradients = tf.merge_summary(tf.get_collection("summary_gradients"))
        self.summary_train = tf.merge_summary(tf.get_collection("summary_train"))
        self.summary_val = tf.merge_summary(tf.get_collection("summary_val"))

    def run(self, sess, fd, references, verbose=False):
        if verbose:
            return sess.run([self.optimize_op, self.decoder.loss_with_decoded_ins,
                             self.decoder.loss_with_gt_ins,
                             self.summary_train] + self.decoder.copynet_logits + self.decoder.decoded_seq,
                             #self.summary_train, self.summary_gradients] + self.decoder.copynet_logits + self.decoder.decoded_seq,
                            feed_dict=fd)
        else:
            return sess.run([self.optimize_op], feed_dict=fd)

    def feed_dict(self, trans_sentences, tgt_sentences, batch_size, dicts=None):
        if dicts is None:
            dicts = [{} for _ in
                     range(len(sentences) / batch_size + int(len(sentences) % batch_size > 0))]

        for fd, start in zip(dicts, list(range(0, len(tgt_sentences), batch_size))):
            this_trans_sentences = trans_sentences[start:start + batch_size]
            this_tgt_sentences = tgt_sentences[start:start + batch_size]

            for i, (target_plc, weight_plc) in enumerate(zip(self.copy_target_plc, self.copy_w_plc)):
                weights = np.zeros(len(this_trans_sentences))
                targets = np.zeros(len(this_trans_sentences), dtype=np.int32)
                for n, (tgt_sent, trans_sent) in enumerate(zip(this_tgt_sentences, this_trans_sentences)):
                    if i < len(tgt_sent):
                        tgt_word = tgt_sent[i]
                        copy_index = -float('inf')
                        for j, trans_word in enumerate(trans_sent):
                            if trans_word == tgt_word and abs(j - i) < abs(copy_index - i):
                                copy_index = j
                                weights[n] = 1.0
                                targets[n] = j + 1
                fd[target_plc] = targets
                fd[weight_plc] = weights

        return dicts
