import tensorflow as tf

from neuralmonkey.learning_utils import feed_dicts

# tests: mypy

class GreedyRunner(object):
    def __init__(self, decoder, batch_size):
        self.decoder = decoder
        self.batch_size = batch_size

    def __call__(self, sess, dataset, coders):
        batched_dataset = dataset.batch_dataset(self.batch_size)
        decoded_sentences = []

        loss_with_gt_ins = 0.0
        loss_with_decoded_ins = 0.0
        batch_count = 0
        for batch in batched_dataset:
            decoded = self.decoder.decoded
            vocab = self.decoder.vocabulary
            batch_feed_dict = feed_dicts(batch, coders, train=False)
            batch_count += 1

            # if is a target sentence, compute also the losses
            # otherwise, just compute zero
            if dataset.has_series(self.decoder.data_id):
                losses = [self.decoder.train_loss,
                          self.decoder.runtime_loss]
            else:
                losses = [tf.zeros([]), tf.zeros([])]

            computation = sess.run(losses + self.decoder.decoded,
                                   feed_dict=batch_feed_dict)
            loss_with_gt_ins += computation[0]
            loss_with_decoded_ins += computation[1]
            decoded_sentences_batch = \
                    vocab.vectors_to_sentences(computation[len(losses):])
            decoded_sentences += decoded_sentences_batch

        return decoded_sentences, \
               loss_with_gt_ins / batch_count, \
               loss_with_decoded_ins / batch_count
