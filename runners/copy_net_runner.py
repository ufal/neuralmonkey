import tensorflow as tf
import numpy as np

from learning_utils import feed_dicts

# TODO inherit from greedy runner

def copynet_substitute(decoded_sentences, copy_logits, copy_sentences):
    """
    Substitutes the <unk> tokens with the tokens from the source encoder we are
    copying from.
    """
    #assert len(computation) -(2*decoder.max_output_len)-2 == 5 ## neplati pri validaci
    assert len(decoded_sentences) == len(copy_sentences)

    for i, (dec_sent, copy_sent) in enumerate(zip(decoded_sentences, copy_sentences)):
        for j, wrd in enumerate(dec_sent):
            if wrd == '<unk>':
                selected = np.argmax(copy_logits[j][i])

                ## Copynet can generate <pad> tokens from outside the sentence
                if selected < len(copy_sent) and selected != 0:
                    decoded_sentences[i][j] = copy_sent[selected - 1]

    return decoded_sentences

def copynet_runner(batch_size, postprocess=None):
    def run(sess, dataset, coders, decoder):
        batched_dataset = dataset.batch_dataset(self.batch_size)
        dicts = [feed_dicts(batch, coders, train=False) for batch in batched_dataset]
        decoded_sentences = []

        loss_with_gt_ins = 0.0
        loss_with_decoded_ins = 0.0
        for batch_feed_dict in dicts:
            if decoder.data_id in dataset:
                losses = [decoder.loss_with_gt_ins,
                          decoder.loss_with_decoded_ins]
            else:
                losses = [tf.zeros([]), tf.zeros([])]

            computation = sess.run(losses + decoder.copynet_logits + decoder.decoded_seq,
                                   feed_dict=batch_feed_dict)
            loss_with_gt_ins += computation[0]
            loss_with_decoded_ins += computation[1]
            decoded_sentences_batch = \
                decoder.vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])
            decoded_sentences += decoded_sentences_batch

            copy_logits = computation[-(2*decoder.max_output_len)-2 : -decoder.max_output_len - 1]
            assert len(copy_logits) == decoder.max_output_len + 1

            # TODO do the copynet subsitute here

        if postprocess is not None:
            decoded_sentences = postprocess(decoded_sentences, dataset)

        return decoded_sentences, \
               loss_with_gt_ins / len(feed_dicts), \
               loss_with_decoded_ins / len(feed_dicts)
    return run
