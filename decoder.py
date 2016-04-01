import math
import tensorflow as tf
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

from decoding_function import attention_decoder
from learning_utils import log

class Decoder:
    def __init__(self, encoders, vocabulary, rnn_size, embedding_size=128, use_attention=False,
                 max_out_len=20, use_peepholes=False, scheduled_sampling=None,
                 dropout_placeholder=None):
        """

        A class that collects the part of the computation graph that is needed
        for decoding.

        TensorBoard summaries are collected in this class into the following
        collections:

            * 'summary_train' - collects statistics from the train-time

            * 'sumarry_test' - collects statistics while being tested on the
                 development data

        Arguments:

            encoders: List of encoders. If no encoder is provided, the decoder
                can be used to train a LM.

            vocabulary: Vocabulary used for decoding

            rnn_size: Size of the RNN state.

            embedding_size (int): Dimensionality of the word
                embeddings used during decoding.

            use_attention (bool): Flag whether use attention in the decoder.

            max_out_len (int): Maximum length of the decoder output.

            use_peepholes (bool): Flag whether peephole connections should be
                used in the GRU decoder.

            scheduled_sampling: Parameter k for inverse sigmoid decay in
                scheduled sampling. If set to None, linear combination of the
                decoded and supervised loss is used as a cost function.

            dropoout_placeholder: If not None, dropout with this placeholder's
                keep probablity will be applied on the logits


        Attributes:

            inputs: List of placeholders for the decoder inputs. The i-th
                element of the list contains a batch of i-th symbols in the
                sequence.

            weights_ins: List of placeholders of particular output symbols
                weights.  The i-th elements of the list contains a vector telling
                for each string of the batch how much the i-th word should.
                contirbute to the loss cumputation. In practice it contains 1's
                for words which are parts of the decoded strings and 0's
                for the padding.

            loss_with_gt_ins: Operator computing the sequence loss when the
                decoder always gets the ground truth input.

            loss_with_decoded_ins: Operator computing the sequence loss when
                the decoder receives previously computed outputs on its input.

            decoded_seq: List of batches of decoded words. (When the
                decoder is fed with its own outputs.)

        """

        self.max_output_len = max_out_len

        if len(encoders) == 1 and rnn_size == encoders[0].encoded.get_shape()[1].value:
            encoded = encoders[0].encoded
            log("Using encoder output wihtout projection.")
        elif len(encoders) >= 1:
            with tf.variable_scope("encoders_projection"):
                encoded_concat = tf.concat(1, [e.encoded for e in encoders])
                concat_size = encoded_concat.get_shape()[1].value
                proj = tf.Variable(tf.truncated_normal([concat_size, rnn_size]),
                                   name="project_encoders")
                encoded_concat_dropped = tf.nn.dropout(encoded_concat, dropout_placeholder)
                proj_bias = tf.Variable(tf.zeros([rnn_size]))
                encoded = tf.matmul(encoded_concat_dropped, proj) + proj_bias
#            with tf.variable_scope("encoders_projection"):
#                projected = []
#                encoders_shapes = []
#                for i, encoder in enumerate(encoders):
#                    encoder_shape = encoder.encoded.get_shape()[1].value
#                    encoders_shapes.append(encoder_shape)
#                    proj = tf.Variable(tf.truncated_normal([encoder_shape, rnn_size]),
#                                       name="project_encoder_{}".format(i))
#                    dropped_encoded = tf.nn.dropout(encoder.encoded, dropout_placeholder)
#                    projected.append(tf.matmul(dropped_encoded, proj))
#                proj_bias = tf.Variable(tf.zeros([rnn_size]))
#                encoded = sum(projected) + proj_bias
#            log("Projection {} encoders (dimensions: {}) into single vector (dimension {}).".format(len(encoders), encoders_shapes, rnn_size))
        elif len(encoders) == 0: # if we want to train just LM
            encoded = tf.zeros(rnn_size)
            log("No encoder - language model only.")


        self.learning_step = tf.Variable(0, name="learning_step", trainable=False)
        self.gt_inputs = []

        with tf.variable_scope("decoder_inputs"):
            for i in range(max_out_len + 2):
                dec = tf.placeholder(tf.int32, [None],
                                     name='decoder{0}'.format(i))
                tf.add_to_collection('dec_encoder_ins', dec)
                self.gt_inputs.append(dec)

        targets = self.gt_inputs[1:]

        self.weights_ins = []
        with tf.variable_scope("input_weights"):
            for _ in range(len(targets)):
                self.weights_ins.append(tf.placeholder(tf.float32, [None]))

        with tf.variable_scope('decoder'):
            decoding_W = \
                tf.Variable(tf.random_uniform([rnn_size, len(vocabulary)], -0.5, 0.5),
                        name="state_to_word_W")
            decoding_B = \
                tf.Variable(tf.fill([len(vocabulary)], - math.log(len(vocabulary))),
                        name="state_to_word_b")
            decoding_EM = \
                tf.Variable(tf.random_uniform([len(vocabulary), embedding_size], -0.5, 0.5),
                        name="word_embeddings")

            embedded_gt_inputs = \
                    [tf.nn.embedding_lookup(decoding_EM, o) for o in self.gt_inputs[:-1]]

            if dropout_placeholder:
                embedded_gt_inputs = \
                    [tf.nn.dropout(i, dropout_placeholder) for i in embedded_gt_inputs]

            def loop(prev_state, _):
                # it takes the previous hidden state, finds the word and formats it
                # as input for the next time step ... used in the decoder in the "real decoding scenario"
                if dropout_placeholder:
                    prev_state = tf.nn.dropout(prev_state, dropout_placeholder)
                out_activation = tf.matmul(prev_state, decoding_W) + decoding_B
                prev_word_index = tf.argmax(out_activation, 1)
                next_step_embedding = \
                        tf.nn.embedding_lookup(decoding_EM, prev_word_index)
                if dropout_placeholder:
                    return tf.nn.dropout(next_step_embedding, dropout_placeholder)
                else:
                    return next_step_embedding

            def sampling_loop(prev_state, i):
                threshold = scheduled_sampling / \
                        (scheduled_sampling + tf.exp(tf.to_float(self.learning_step) / scheduled_sampling))
                condition = tf.less_equal(tf.random_uniform(tf.shape(embedded_gt_inputs[0])), threshold)
                return tf.select(condition, embedded_gt_inputs[i], loop(prev_state, i))

            decoder_cell = \
                rnn_cell.GRUCell(rnn_size, embedding_size)

            gt_loop_function = sampling_loop if scheduled_sampling else None

            attention_tensors = [e.attention_tensor for e in encoders]
            if dropout_placeholder:
                encoded = tf.nn.dropout(encoded, dropout_placeholder)
                attention_tensors_dropped = \
                    [tf.nn.dropout(t, dropout_placeholder) for t in attention_tensors]

            if use_attention:
                rnn_outputs_gt_ins, _ = attention_decoder(embedded_gt_inputs, encoded,
                                                      attention_states=attention_tensors_dropped,
                                                      cell=decoder_cell,
                                                      loop_function=gt_loop_function)
            else:
                rnn_outputs_gt_ins, _ = seq2seq.rnn_decoder(embedded_gt_inputs, encoded,
                                                cell=decoder_cell,
                                                loop_function=gt_loop_function)

            tf.get_variable_scope().reuse_variables()

            if use_attention:
                rnn_outputs_decoded_ins, _ = \
                    attention_decoder(embedded_gt_inputs, encoded,
                                              cell=decoder_cell,
                                              attention_states=attention_tensors_dropped,
                                              loop_function=loop)
            else:
                rnn_outputs_decoded_ins, _ = \
                    seq2seq.rnn_decoder(embedded_gt_inputs, encoded,
                                        cell=decoder_cell,
                                        loop_function=loop)

        def loss_and_decoded(rnn_outputs, use_dropout):
            logits = []
            decoded = []
            for o in rnn_outputs:
                if use_dropout and dropout_placeholder:
                    o = tf.nn.dropout(o, dropout_placeholder)
                out_activation = tf.matmul(o, decoding_W) + decoding_B
                logits.append(out_activation)
                decoded.append(tf.argmax(out_activation, 1))
            loss = seq2seq.sequence_loss(logits, targets,
                                         self.weights_ins, len(vocabulary))
            return loss, decoded

        self.loss_with_gt_ins, _ = \
                loss_and_decoded(rnn_outputs_gt_ins, True)

        tf.scalar_summary('loss_on_dev_data_with_gt_input', self.loss_with_gt_ins, collections=["summary_test"])
        tf.scalar_summary('loss_on_train_data_with_gt_intpus', self.loss_with_gt_ins, collections=["summary_train"])

        self.loss_with_decoded_ins, self.decoded_seq = \
                loss_and_decoded(rnn_outputs_decoded_ins, False)

        tf.scalar_summary('loss_on_dev_data_with_decoded_inputs', self.loss_with_decoded_ins, collections=["summary_test"])
        tf.scalar_summary('loss_on_train_data_with_decoded_inputs', self.loss_with_decoded_ins, collections=["summary_train"])

        if scheduled_sampling:
            self.cost = self.loss_with_gt_ins
        else:
            #self.cost = self.loss_with_gt_ins
            self.cost = 0.5 * (self.loss_with_decoded_ins + self.loss_with_gt_ins)

        tf.scalar_summary('optimization_cost_on_dev_data', self.cost, collections=["summary_test"])
        tf.scalar_summary('optimization_cost_on_train_data', self.cost, collections=["summary_train"])

