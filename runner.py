import tensorflow as tf

from learning_utils import feed_dicts

# TODO add copynet to the greedy runner

def greedy_runner(decoder, batch_size, postprocess=None):
    def run(sess, dataset, coders):
        dicts = feed_dicts(dataset, batch_size, coders, train=False)
        decoded_sentences = []

        loss_with_gt_ins = 0.0
        loss_with_decoded_ins = 0.0
        for batch_feed_dict in dicts:
            if decoder.data_id in dataset.series:
                losses = [decoder.loss_with_gt_ins,
                          decoder.loss_with_decoded_ins]
            else:
                losses = [tf.zeros([]), tf.zeros([])]

            computation = sess.run(losses + decoder.decoded_seq,
                                   feed_dict=batch_feed_dict)
            loss_with_gt_ins += computation[0]
            loss_with_decoded_ins += computation[1]
            decoded_sentences_batch = \
                decoder.vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])
            decoded_sentences += decoded_sentences_batch

        if postprocess is not None:
            decoded_sentences = postprocess(decoded_sentences, dataset)

        return decoded_sentences, \
               loss_with_gt_ins / len(dicts), \
               loss_with_decoded_ins / len(dicts)
    return run
