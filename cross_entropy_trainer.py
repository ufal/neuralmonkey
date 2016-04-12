import tensorflow as tf


class CrossEntropyTrainer(object):

    def __init__(self, decoder, l2_regularization):
        self.decoder = decoder

        if l2_regularization > 0:
            with tf.variable_scope("l2_regularization"):
                l2_cost = l2_regularization * \
                    sum([tf.reduce_sum(v ** 2) for v in tf.trainable_variables()])
        else:
            l2_cost = 0.0

        self.optimize_op = tf.train.AdamOptimizer(1e-4).minimize(decoder.cost + l2_cost, \
                                                                     global_step=decoder.learning_step)

    def run(self, sess, fd, references, verbose=False):
        if verbose:
            return sess.run([self.optimize_op, self.decoder.loss_with_decoded_ins,
                             self.decoder.loss_with_gt_ins, self.decoder.summary_train] + self.decoder.decoded_seq,
                            feed_dict=fd)
        else:
            return sess.run([self.optimize_op], feed_dict=fd)

