import tensorflow as tf
from bidirectional_rnn_layer import BidirectionalRNNLayer
from tensorflow.models.rnn import rnn_cell
from noisy_gru_cell import NoisyGRUCell
import numpy as np

class SentenceEncoder(object):
    def __init__(self, max_input_len, vocabulary, embedding_size, rnn_size, dropout_placeholder,
                 is_training, use_noisy_activations=False, attention_type=None, attention_fertility=None, name="sentence_encoder"):
        self.vocabulary = vocabulary
        self.max_input_len = max_input_len

        with tf.variable_scope(name):
            self.inputs = \
                    [tf.placeholder(tf.int32, shape=[None], name="input_{}".format(i)) for i in range(max_input_len + 2)]
            self.sentence_lengths = tf.placeholder(tf.int64, shape=[None], name="sequence_lengths")

            self.word_embeddings = tf.Variable(tf.random_uniform([len(vocabulary), embedding_size], -1.0, 1.0))

            embedded_inputs = [tf.nn.embedding_lookup(self.word_embeddings, input_) for input_ in self.inputs]

            dropped_embedded_inputs = [tf.nn.dropout(i, dropout_placeholder) for i in embedded_inputs]

            if use_noisy_activations:
                forward_gru = NoisyGRUCell(rnn_size, is_training, input_size=embedding_size)
                backward_gru = NoisyGRUCell(rnn_size, is_training, input_size=embedding_size)

            else:
                forward_gru = rnn_cell.GRUCell(rnn_size, input_size=embedding_size)
                backward_gru = rnn_cell.GRUCell(rnn_size, input_size=embedding_size)


            bidi_layer = BidirectionalRNNLayer(forward_gru,
                                               backward_gru,
                                               dropped_embedded_inputs,
                                               self.sentence_lengths)

            self.outputs_bidi = bidi_layer.outputs_bidi
            self.encoded = bidi_layer.encoded

            self.attention_tensor = \
                    tf.concat(1, [tf.expand_dims(o, 1) for o in self.outputs_bidi])

            self.attention_object = attention_type(self.attention_tensor,
                                                   scope="attention_{}".format(name),
                                                   dropout_placeholder=dropout_placeholder,
                                                   max_fertility=attention_fertility) if attention_type else None


    def feed_dict(self, sentences, batch_size, train=False, dicts=None):
        if dicts == None:
            dicts = [{} for _ in range(len(sentences) / batch_size + int(len(sentences) % batch_size > 0))]

        batched_sentences = []
        for fd, start in zip(dicts, range(0, len(sentences), batch_size)):
            this_sentences = sentences[start:start + batch_size]
            fd[self.sentence_lengths] = np.array([min(self.max_input_len, len(s)) + 2 for s in this_sentences])
            vectors, _ = \
                    self.vocabulary.sentences_to_tensor(this_sentences, self.max_input_len, train=train)
            for words_plc, words_tensor in zip(self.inputs, vectors):
                fd[words_plc] = words_tensor
            batched_sentences.append(this_sentences)

        return dicts, batched_sentences
