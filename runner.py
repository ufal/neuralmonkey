import tensorflow as tf

from learning_utils import feed_dicts

class GreedyRunner(object):
    def __init__(self, decoder, batch_size, postprocess=None):
        self.decoder = decoder
        self.batch_size = batch_size
        self.postprocess = postprocess
        self.vocabulary = decoder.vocabulary

    def __call__(self, sess, dataset, coders):
        batched_dataset = dataset.batch_dataset(self.batch_size)
        dicts = [feed_dicts(batch, coders, train=False) for batch in batched_dataset]
        decoded_sentences = []

        loss_with_gt_ins = 0.0
        loss_with_decoded_ins = 0.0
        for batch_feed_dict in dicts:
            if self.decoder.data_id in dataset.series:
                losses = [self.decoder.loss_with_gt_ins,
                          self.decoder.loss_with_decoded_ins]
            else:
                losses = [tf.zeros([]), tf.zeros([])]

            computation = sess.run(losses + self.decoder.decoded_seq,
                                   feed_dict=batch_feed_dict)
            loss_with_gt_ins += computation[0]
            loss_with_decoded_ins += computation[1]
            decoded_sentences_batch = \
                self.vocabulary.vectors_to_sentences(computation[-self.decoder.max_output_len - 1:])
            decoded_sentences += decoded_sentences_batch

        if self.postprocess is not None:
            decoded_sentences = self.postprocess(decoded_sentences, dataset)

        return decoded_sentences, \
               loss_with_gt_ins / len(dicts), \
               loss_with_decoded_ins / len(dicts)
