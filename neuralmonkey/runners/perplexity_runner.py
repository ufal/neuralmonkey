"""

This module contains an implementation of a runner that is supposed to be used
in case we train a language model. Instead of decoding sentences in computes
its perplexities given the decoder.

"""

import numpy as np
import tensorflow as tf

from neuralmonkey.learning_utils import feed_dicts

class PerplexityRunner(object):
    def __init__(self, decoder, batch_size):
        self.decoder = decoder
        self.batch_size = batch_size
        self.vocabulary = decoder.vocabulary

        self.cross_entropies_op = tf.nn.seq2seq.sequence_loss_by_example(
            decoder.gt_logits, decoder.targets, decoder.weights_ins,
            len(decoder.vocabulary))

    def __call__(self, sess, dataset, coders):
        if not dataset.has_series(self.decoder.data_id):
            raise Exception("Dataset must have the target values ({}) for computing perplexity.".\
                    format(self.decoder.data_id))

        batched_dataset = dataset.batch_dataset(self.batch_size)
        losses = [self.decoder.loss_with_gt_ins,
                  self.decoder.loss_with_decoded_ins]
        perplexities = []

        loss_with_gt_ins = 0.0
        loss_with_decoded_ins = 0.0
        batch_count = 0
        for batch in batched_dataset:
            batch_count += 1
            batch_feed_dict = feed_dicts(batch, coders, train=False)
            cross_entropies, opt_loss, dec_loss = \
                sess.run([self.cross_entropies_op] + losses, feed_dict=batch_feed_dict)
            perplexities.extend([2 ** xent for xent in cross_entropies])
            loss_with_gt_ins += opt_loss
            loss_with_decoded_ins += dec_loss

        return perplexities, \
               loss_with_gt_ins / batch_count, \
               loss_with_decoded_ins / batch_count
