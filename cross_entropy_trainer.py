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

        optimizer = tf.train.AdamOptimizer(1e-4)
        gradients = optimizer.compute_gradients(decoder.cost + l2_cost)
        for (g, v) in gradients:
            tf.histogram_summary('gr_' + v.name, g, collections=["summary_gradients"])
        self.optimize_op = optimizer.apply_gradients(gradients, global_step=decoder.learning_step)
        self.summary_gradients = tf.merge_summary(tf.get_collection("summary_gradients"))

    def run(self, sess, fd, references, verbose=False):
        if verbose:
            return sess.run([self.optimize_op, self.decoder.loss_with_decoded_ins,
                             self.decoder.loss_with_gt_ins,
                             self.decoder.summary_train, self.summary_gradients] + self.decoder.decoded_seq,
                            feed_dict=fd)
        else:
            return sess.run([self.optimize_op], feed_dict=fd)

