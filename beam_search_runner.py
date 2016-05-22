import numpy as np
import tensorflow as tf

from learning_utils import feed_dicts
from utils import log

def expand(session, decoder, feed_dict, state, hypotheses):
    # type: (tf.Session, Decoder, Feed_dict, np.Array, List[Hypothesis]) -> List[Hypothesis]
    feed_dict[decoder.encoded] = state
    hyp_length = len(hypotheses[0][1])
    hyp_count = len(hypotheses)
    if hyp_length == 2:
        for k in feed_dict:
            shape = k.get_shape()
            #print shape
            #import ipdb; ipdb.set_trace()
            if not shape == tf.TensorShape(None):
                if len(shape) == 1:
                    feed_dict[k] = np.repeat(feed_dict[k], hyp_count)
                elif len(shape) == 2:
                    feed_dict[k] = np.repeat(np.array(feed_dict[k]), hyp_count, axis=0)
                else:
                    log("ERROR in expanding beamsearch hypothesis")
    elif hyp_length > 2:
        feed_dict[decoder.encoded] = np.repeat(state, hyp_count, axis=0)

    for i, n in zip(decoder.gt_inputs, range(hyp_length)):
        for k in range(hyp_count):
            feed_dict[i][k] = hypotheses[k][1][n]
    probs, prob_i = session.run([decoder.top10_probs[hyp_length - 1][0],
                                 decoder.top10_probs[hyp_length - 1][1]],
                                feed_dict=feed_dict)
    beam = []
    for i in range(hyp_count):
        for p, x in zip(probs[i], prob_i[i]):
            beam.append((hypotheses[i][0] + p, hypotheses[i][1] + [x]))
    return beam


def beamsearch(session, decoder, feed_dict):
    # type: (tf.Session, Decoder, Feed_dict) -> List[int]
    beam = [(1.0, [1])]
    state = session.run(decoder.encoded, feed_dict)
    for _ in range(len(decoder.decoded_probs)):
        new_beam = expand(session, decoder, feed_dict, state, beam)
        new_beam.sort(reverse=True)
        beam = new_beam[:10]
    return beam[0][1]


def beam_search_runner(beam_size, postprocess=None):

    # TODO move top10 here from decoder

    def run(sess, dataset, coders, decoder):
        singleton_dicts = feed_dicts(dataset, 1, coders)

        # call beamsearch for each sentence
        decoded_sentences = []
        for sent_dict in singleton_dicts:
            decoded_s = beamsearch(sess, decoder, sent_dict)
            decoded_sentences_batch = \
                    decoder.vocabulary.vectors_to_sentences([np.array([i]) for i in decoded_s[1:]])
            decoded_sentences += decoded_sentences_batch

        if postprocess is not None:
            decoded_sentences = postprocess(decoded_sentences, dataset)

        # return sentences and empty losses
        return decoded_sentences, 0.0, 0.0
    return run
