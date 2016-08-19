"""
This module contains an implementation of a runner that is supposed to be
used in case we train a language model. Instead of decoding sentences in
computes its perplexities given the decoder.
"""
#tests: lint

from neuralmonkey.learning_utils import feed_dicts

#pylint: disable=too-few-public-methods
class PerplexityRunner(object):
    def __init__(self, decoder, batch_size):
        self.decoder = decoder
        self.batch_size = batch_size
        self.vocabulary = decoder.vocabulary

    def __call__(self, sess, dataset, coders):
        if not dataset.has_series(self.decoder.data_id):
            raise Exception("Dataset must have the target values ({})"
                            "for computing perplexity."
                            .format(self.decoder.data_id))
        perplexities = []
        train_loss = 0.0
        runtime_loss = 0.0
        batch_count = 0

        for batch in dataset.batch_dataset(self.batch_size):
            batch_count += 1
            batch_feed_dict = feed_dicts(batch, coders, train=False)
            cross_entropies, opt_loss, dec_loss = sess.run(
                [self.decoder.cross_entropies,
                 self.decoder.train_loss,
                 self.decoder.runtime_loss],
                feed_dict=batch_feed_dict)

            perplexities.extend([2 ** xent for xent in cross_entropies])
            train_loss += opt_loss
            runtime_loss += dec_loss

        avg_train_loss = train_loss / batch_count
        avg_runtime_loss = runtime_loss / batch_count

        return perplexities, avg_train_loss, avg_runtime_loss
