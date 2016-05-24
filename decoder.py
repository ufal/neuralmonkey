import math
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn_cell

from utils import log
from decoding_function import attention_decoder
from learning_utils import log
from noisy_gru_cell import NoisyGRUCell

class Decoder:
    def __init__(self, encoders, vocabulary, data_id, rnn_size, embedding_size=128, use_attention=None,
                 max_out_len=20, scheduled_sampling=None,
                 dropout_keep_p=0.5, copy_net=None, reused_word_embeddings=None,
                 use_noisy_activations=False, depth=1, name="decoder"):

        """

        A class that collects the part of the computation graph that is needed
        for decoding.

        TensorBoard summaries are collected in this class into the following
        collections:

            * 'summary_train' - collects statistics from the train-time

            * 'sumarry_val' - collects statistics while being tested on the
                 development data

        Arguments:

            encoders: List of encoders. If no encoder is provided, the decoder
                can be used to train a LM.

            vocabulary: Vocabulary used for decoding

            data_id:

            rnn_size: Size of the RNN state.

            embedding_size (int): Dimensionality of the word
                embeddings used during decoding.

            use_attention (str): The type of attention to use or None. (Refer to
                cli_options script for allowed types of attention]

            max_out_len (int): Maximum length of the decoder output.

            use_peepholes (bool): Flag whether peephole connections should be
                used in the GRU decoder.

            scheduled_sampling: Parameter k for inverse sigmoid decay in
                scheduled sampling. If set to None, linear combination of the
                decoded and supervised loss is used as a cost function.

            dropoout_keep_p:

            copy_net: Tuple of (i) list of indices to the target vocabulary
                (most likely input placeholders of a different encoder) and (ii) he
                tensor over which the copying will be done, and (iii) mask telling
                which words part of the input

            reused_word_embeddings: The decoder can be given the matrix of word
                embeddings from outside (if the vocabulary indexing is the same).
                If it is None, the decoder creates its own matrix of word
                embeddings.

            use_noisy_activations: If set to True, the deocder will use the GRU
                units with noisy activation.


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

        log("Initializing decoder, name: \"{}\"".format(name))
        self.data_id = data_id
        self.dropout_keep_p = dropout_keep_p
        self.vocabulary = vocabulary
        self.rnn_size = rnn_size
        self.max_output_len = max_out_len
        self.dropout_placeholder = tf.placeholder(tf.float32, name="decoder_dropout_plc")
        self.is_training = tf.placeholder(tf.bool, name="decoder_is_training")

        if len(encoders) == 1 and rnn_size == encoders[0].encoded.get_shape()[1].value:
            encoded = encoders[0].encoded
            log("Using encoder output wihtout projection.")
        elif len(encoders) >= 1:
            with tf.variable_scope("encoders_projection"):
                encoded_concat = tf.concat(1, [e.encoded for e in encoders])
                concat_size = encoded_concat.get_shape()[1].value
                proj = tf.get_variable(name="project_encoders",
                                       shape=[concat_size, depth * rnn_size])
                encoded_concat_dropped = tf.nn.dropout(encoded_concat, self.dropout_placeholder)
                proj_bias = tf.Variable(tf.zeros([depth * rnn_size]))
                encoded = tf.matmul(encoded_concat_dropped, proj) + proj_bias
        elif len(encoders) == 0: # if we want to train just LM
            encoded = tf.zeros(rnn_size)
            log("No encoder - language model only.")
        self.encoded = encoded

        self.learning_step = tf.Variable(0, name="learning_step", trainable=False)
        self.gt_inputs = []

        with tf.variable_scope("decoder_inputs"):
            for i in range(max_out_len + 2):
                dec = tf.placeholder(tf.int64, [None],
                                     name='decoder{0}'.format(i))
                tf.add_to_collection('dec_encoder_ins', dec)
                self.gt_inputs.append(dec)

        self.targets = self.gt_inputs[1:]

        self.weights_ins = []
        with tf.variable_scope("input_weights"):
            for _ in range(len(self.targets)):
                self.weights_ins.append(tf.placeholder(tf.float32, [None]))

        with tf.variable_scope('decoder'):
            decoding_W = \
                tf.Variable(tf.random_uniform([rnn_size, len(vocabulary)], -0.5, 0.5),
                        name="state_to_word_W")
            decoding_B = \
                tf.Variable(tf.fill([len(vocabulary)], - math.log(len(vocabulary))),
                        name="state_to_word_b")

            if reused_word_embeddings is None:
                decoding_EM = \
                    tf.Variable(tf.random_uniform([len(vocabulary), embedding_size], -0.5, 0.5),
                                name="word_embeddings")
            else:
                decoding_EM = reused_word_embeddings

            embedded_gt_inputs = \
                    [tf.nn.embedding_lookup(decoding_EM, o) for o in self.gt_inputs[:-1]]

            embedded_gt_inputs = \
                [tf.nn.dropout(i, self.dropout_placeholder) for i in embedded_gt_inputs]

            def standard_logits(state):
                state = tf.nn.dropout(state, self.dropout_placeholder)
                return tf.matmul(state, decoding_W) + decoding_B, None

            logit_function = standard_logits


            if copy_net:
                # This is implementation of Copy-net (http://arxiv.org/pdf/1603.06393v2.pdf)
                encoder_input_indices, copy_states, copy_mask = copy_net
                copy_tensor_dropped = tf.nn.dropout(copy_states, self.dropout_placeholder)
                copy_tensors = [tf.squeeze(t, [1]) for t in tf.split(1, max_out_len + 2, copy_tensor_dropped)]
                copy_features_size = copy_states.get_shape()[2].value

                # first we do the learned projection of the ecnoder outputs
                copy_W = tf.get_variable(name="copy_W", shape=[copy_features_size, rnn_size])

                projected_inputs = \
                        tf.concat(1, [tf.expand_dims(tf.matmul(c, copy_W), 1) for c in copy_tensors])
                batch_size = tf.shape(encoder_input_indices[0])[0]

                # tensor of batch numbers for indexing in a sparse vector
                batch_range = tf.range(start=0, limit=batch_size)
                batch_time_vocabulary_shape = \
                        tf.concat(0, [tf.expand_dims(batch_size, 0), tf.constant(len(vocabulary), shape=[1])])
                ones = tf.ones(tf.expand_dims(batch_size, 0))

                vocabulary_shaped_list = []
                for slice_indices in encoder_input_indices:
                    complete_indices = \
                        tf.concat(1, [tf.expand_dims(batch_range, 1), tf.expand_dims(slice_indices, 1)])
                    vocabulary_shaped = \
                                tf.sparse_to_dense(complete_indices,
                                                   batch_time_vocabulary_shape,
                                                   ones)
                    vocabulary_shaped_list.append(vocabulary_shaped)
                vocabulary_shaped_indices = tf.concat(1, [tf.expand_dims(v, 1) for v in vocabulary_shaped_list])

                def copy_net_logit_function(state):
                    state = tf.nn.dropout(state, self.dropout_placeholder)
                    # the logits for generating the next word are computed in the standard way
                    generate_logits = tf.matmul(state, decoding_W) + decoding_B

                    # in addition to that logits for copying a word from the
                    # input are computed, here in a loop for each of the
                    # encoder words
                    all_vocabulary_logits = [generate_logits]

                    # Equation 8 in the paper ... in shape of source sentence (batch x time)
                    copy_logits_in_time = tf.reduce_sum(projected_inputs * tf.expand_dims(state, 1), [2])
                    # mask out the padding in exponential domain
                    copy_logits_in_time_exp_masked = tf.exp(tf.minimum([[80.0]], copy_logits_in_time)) * copy_mask

                    #  ... in shape of vocabulary (batch x time x vocabulary)
                    copy_logits_in_vocabulary = \
                        vocabulary_shaped_indices * tf.expand_dims(copy_logits_in_time_exp_masked, 2)

                    # Equation 6 without normalization
                    copy_logits_exp = tf.reduce_sum(copy_logits_in_vocabulary, [1])

                    logits_exp = copy_logits_exp + tf.exp(tf.minimum([[80.0]], generate_logits))

                    return tf.log(tf.maximum([[1e-40]], logits_exp)), copy_logits_in_time

                logit_function = copy_net_logit_function

            def loop(prev_state, _):
                # it takes the previous hidden state, finds the word and formats it
                # as input for the next time step ... used in the decoder in the "real decoding scenario"
                out_activation, _ = logit_function(prev_state)
                prev_word_index = tf.argmax(out_activation, 1)
                next_step_embedding = \
                        tf.nn.embedding_lookup(decoding_EM, prev_word_index)
                return tf.nn.dropout(next_step_embedding, self.dropout_placeholder)

            def sampling_loop(prev_state, i):
                """
                Loop function performing the scheduled sampling
                (http://arxiv.org/pdf/1506.03099v3.pdf) with the inverse
                sigmoid decay.
                """
                threshold = scheduled_sampling / \
                        (scheduled_sampling + tf.exp(tf.to_float(self.learning_step) / scheduled_sampling))
                condition = tf.less_equal(tf.random_uniform(tf.shape(embedded_gt_inputs[0])), threshold)
                return tf.select(condition, embedded_gt_inputs[i], loop(prev_state, i))

            def get_rnn_cell(input_size=rnn_size):
                if use_noisy_activations:
                    return NoisyGRUCell(rnn_size, training=self.is_training, input_size=input_size)
                else:
                    return rnn_cell.GRUCell(rnn_size, input_size=input_size)

            decoder_cells = [get_rnn_cell(input_size=embedding_size)]
            for _ in range(1, depth):
                decoder_cells[-1] = rnn_cell.DropoutWrapper(decoder_cells[-1], output_keep_prob=self.dropout_placeholder)
                decoder_cells.append(get_rnn_cell())

            decoder_cell = rnn_cell.MultiRNNCell(decoder_cells)

            gt_loop_function = sampling_loop if scheduled_sampling else None

            encoded = tf.nn.dropout(encoded, self.dropout_placeholder)

            if use_attention:
                attention_objects = [e.attention_object for e in encoders if e.attention_object]
            else:
                attention_objects = []


            rnn_outputs_gt_ins, _ = \
                attention_decoder(embedded_gt_inputs, encoded,
                                  attention_objects=attention_objects,
                                  cell=decoder_cell,
                                  loop_function=gt_loop_function)


            tf.get_variable_scope().reuse_variables()

            self.go_symbols = tf.placeholder(tf.int64, [None], name="go_symbols_placeholder")
            decoder_inputs = [tf.nn.embedding_lookup(decoding_EM, self.go_symbols)]
            decoder_inputs += [None for _ in range(self.max_output_len)]

            rnn_outputs_decoded_ins, _ = \
                attention_decoder(decoder_inputs, encoded,
                                  cell=decoder_cell,
                                  attention_objects=attention_objects,
                                  loop_function=loop)

            self.hidden_states = rnn_outputs_decoded_ins

        def get_decoded(rnn_outputs):
            logits = []
            decoded = []
            copynet_logits = []

            for o in rnn_outputs:
                out_activation, logits_in_time = logit_function(o)

                if copy_net:
                    copynet_logits.append(logits_in_time)

                logits.append(out_activation)
                decoded.append(tf.argmax(out_activation[:,1:], 1) + 1)

            return decoded, logits, copynet_logits


        _, self.gt_logits, _ = get_decoded(rnn_outputs_gt_ins)
        self.loss_with_gt_ins = seq2seq.sequence_loss(self.gt_logits, self.targets, self.weights_ins, len(vocabulary))

        tf.scalar_summary('val_loss_with_gt_input', self.loss_with_gt_ins, collections=["summary_val"])
        tf.scalar_summary('train_loss_with_gt_intpus', self.loss_with_gt_ins, collections=["summary_train"])


        self.decoded_seq, self.decoded_logits, self.copynet_logits = get_decoded(rnn_outputs_decoded_ins)
        self.loss_with_decoded_ins = seq2seq.sequence_loss(self.decoded_logits, self.targets, self.weights_ins, len(vocabulary))

        tf.scalar_summary('val_loss_with_decoded_inputs', self.loss_with_decoded_ins, collections=["summary_val"])
        tf.scalar_summary('train_loss_with_decoded_inputs', self.loss_with_decoded_ins, collections=["summary_train"])

        self.cost = self.loss_with_gt_ins

        tf.scalar_summary('val_optimization_cost', self.cost, collections=["summary_val"])
        tf.scalar_summary('train_optimization_cost', self.cost, collections=["summary_train"])

        log("Decoder initalized.")

    def feed_dict(self, dataset, batch_size, train=False, dicts=None):
        data_size = len(dataset)
        sentences = dataset.series.get(self.data_id)

        if dicts == None:
            dicts = [{} for _ in range(data_size / batch_size + int(data_size % batch_size > 0))]
        batched_sentences = []

        for fd, start in zip(dicts, range(0, data_size, batch_size)):
            batch_actual_size = min(start + batch_size, data_size) - start

            if sentences is not None:
                this_batch_sentences = sentences[start:start + batch_size]
                sentnces_tensors, weights_tensors = \
                    self.vocabulary.sentences_to_tensor(this_batch_sentences, self.max_output_len)

                for weight_plc, weight_tensor in zip(self.weights_ins, weights_tensors):
                    fd[weight_plc] = weight_tensor

                for words_plc, words_tensor in zip(self.gt_inputs, sentnces_tensors):
                    fd[words_plc] = words_tensor

                batched_sentences.append(this_batch_sentences)

            fd[self.go_symbols] = np.ones(batch_actual_size)

            if train:
                fd[self.dropout_placeholder] = self.dropout_keep_p
            else:
                fd[self.dropout_placeholder] = 1.0

        if sentences is not None:
            return dicts, batched_sentences
        else:
            return dicts, None

