import numpy as np
import tensorflow as tf

from neuralmonkey.learning_utils import feed_dicts
from neuralmonkey.utils import log

class BeamSearchRunner(object):
    def __init__(self, decoder, beam_size, postprocess=None):
        self.decoder = decoder
        self.beam_size = beam_size
        self.postprocess = postprocess
        self.vocabulary = decoder.vocabulary

        self.decoded_probs = [tf.nn.log_softmax(l) for l in decoder.gt_logits]
        self.top_n_probs = [tf.nn.top_k(p, beam_size) for p in self.decoded_probs]

    def _expand(self, session, feed_dict, state, hypotheses):
        # type: (tf.Session, Decoder, Feed_dict, np.Array, List[Hypothesis]) -> List[Hypothesis]
        feed_dict[self.decoder.encoded] = state
        hyp_length = len(hypotheses[0][1])
        hyp_count = len(hypotheses)
        if hyp_length == 2:
            for k in feed_dict:
                shape = k.get_shape()
                if shape != tf.TensorShape(None):
                    if len(shape) == 1:
                        feed_dict[k] = np.repeat(feed_dict[k], hyp_count)
                    elif len(shape) == 2:
                        feed_dict[k] = np.repeat(np.array(feed_dict[k]), hyp_count, axis=0)
                    else:
                        log("ERROR in expanding beamsearch hypothesis")
        elif hyp_length > 2:
            feed_dict[self.decoder.encoded] = np.repeat(state, hyp_count, axis=0)

        for i, n in zip(self.decoder.gt_inputs, range(hyp_length)):
            for k in range(hyp_count):
                feed_dict[i][k] = hypotheses[k][1][n]
        probs, prob_i = session.run([self.top_n_probs[hyp_length - 1][0],
                                     self.top_n_probs[hyp_length - 1][1]],
                                    feed_dict=feed_dict)
        beam = []
        for i in range(hyp_count):
            for p, x in zip(probs[i], prob_i[i]):
                beam.append((hypotheses[i][0] + p, hypotheses[i][1] + [x]))
        return beam

    def _beamsearch(self, session, feed_dict):
        # type: (tf.Session, Decoder, Feed_dict) -> List[int]
        beam = [(1.0, [1])]
        state = session.run(self.decoder.encoded, feed_dict)
        for _ in range(len(self.decoded_probs)):
            new_beam = self._expand(session, feed_dict, state, beam)
            new_beam.sort(reverse=True)
            beam = new_beam[:10]
        return beam[0][1]

    def __call__(self, sess, dataset, coders):
        singletons = dataset.batch_dataset(1)
        singleton_dicts = [feed_dicts(sing, coders, train=False) for sing in singletons]

        # call beamsearch for each sentence
        decoded_sentences = []
        for sent_dict in singleton_dicts:
            decoded_s = self._beamsearch(sess, sent_dict)
            decoded_sentences_batch = \
                    self.vocabulary.vectors_to_sentences([np.array([i]) for i in decoded_s[1:]])
            decoded_sentences += decoded_sentences_batch

        if self.postprocess is not None:
            decoded_sentences = self.postprocess(decoded_sentences, dataset)

        # return sentences and empty losses
        return decoded_sentences, 0.0, 0.0
