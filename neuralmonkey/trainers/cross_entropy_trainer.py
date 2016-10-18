import tensorflow as tf

from neuralmonkey.logging import log

# tests: mypy


def l2_cost(l2_parameter):
    with tf.variable_scope("l2_regularization"):
        l2_value = sum([tf.reduce_sum(v ** 2)
                        for v in tf.trainable_variables()])

        l2_cost = l2_parameter * l2_value
        tf.scalar_summary('train_l2_cost', l2_value,
                          collections=["summary_train"])
    return l2_cost


class CrossEntropyTrainer(object):
    def __init__(self, decoder, l2_regularization, optimizer=None,
                 clip_norm=None):
        log("Initializing Cross-entropy trainer.")
        self.decoder = decoder

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(1e-4)

        cost = decoder.cost
        if l2_regularization > 0:
            cost += l2_cost(l2_regularization)

        gradients = optimizer.compute_gradients(cost)

        if clip_norm is not None:
            gradients = [(tf.clip_by_norm(grad, clip_norm), var)
                         for grad, var in gradients]

        #for (g, v) in gradients:
        #    if g is not None:
        #        tf.histogram_summary('gr_' + v.name, g, collections=["summary_gradients"])

        self.optimize_op = optimizer.apply_gradients(
            gradients, global_step=decoder.learning_step)

        #self.summary_gradients = tf.merge_summary(tf.get_collection("summary_gradients"))

        self.summary_train = tf.merge_summary(
            tf.get_collection("summary_train"))
        log("Trainer initialized.")

    def run(self, sess, f_dict, summary=False):
        if summary:
            _, summary_str = \
                   sess.run([self.optimize_op,
                             self.summary_train],#, self.summary_gradients
                            feed_dict=f_dict)
            return summary_str
        else:
            sess.run(self.optimize_op, feed_dict=f_dict)
