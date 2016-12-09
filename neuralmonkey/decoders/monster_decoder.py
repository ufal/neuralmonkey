
"""
Old, deprecated version of the decoder. In time, this will vanish.
Now it is here for inspiration to young developers who do not wish
to end up like us.
Use this module on your own risk. It won't work anyway since the
API has changed.
"""

# tests: mypy, lint
# pylint: skip-file

import math
import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log
#from neuralmonkey.decoding_function import attention_decoder
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.checking import assert_type
from neuralmonkey.vocabulary import Vocabulary, START_TOKEN


class Decoder(object):

    def __init__(self, encoders, vocabulary, data_id, rnn_size, name,
                 embedding_size=128, use_attention=None, max_output_len=20,
                 scheduled_sampling=None, dropout_keep_prob=0.5, copy_net=None,
                 reused_word_embeddings=None, use_noisy_activations=False,
                 depth=1):
        """A class that collects the part of the computation graph that is
        needed for decoding.

        TensorBoard summaries are collected in this class into the following
        collections:

            * 'summary_train' - collects statistics from the train-time

            * 'sumarry_val' - collects OAstatistics while being tested on the
                 development data

        Arguments:

            encoders: List of encoders. If no encoder is provided, the decoder
                can be used to train a LM.

            vocabulary: Vocabulary used for decoding

            data_id:

            rnn_size: Size of the RNN state.

            embedding_size (int): Dimensionality of the word
                embeddings used during decoding.

            use_attention (str): The type of attention to use or None. (Refer
                to cli_options script for allowed types of attention]

            max_output_len (int): Maximum length of the decoder output.

            use_peepholes (bool): Flag whether peephole connections should be
                used in the GRU decoder.

            scheduled_sampling: Parameter k for inverse sigmoid decay in
                scheduled sampling. If set to None, linear combination of the
                decoded and supervised loss is used as a cost function.

            dropoout_keep_p:

            copy_net: Tuple of (i) list of indices to the target vocabulary
                (most likely input placeholders of a different encoder) and
                (ii) the tensor over which the copying will be done, and
                (iii) mask telling which words part of the input

            reused_word_embeddings: The decoder can be given the matrix of word
                embeddings from outside (if the vocabulary indexing is the
                same).  If it is None, the decoder creates its own matrix of
                word embeddings.

            use_noisy_activations: If set to True, the deocder will use the GRU
                units with noisy activation.


        Attributes:

            inputs: List of placeholders for the decoder inputs. The i-th
                element of the list contains a batch of i-th symbols in the
                sequence.

            weights_ins: List of placeholders of particular output symbols
                weights.  The i-th elements of the list contains a vector
                telling for each string of the batch how much the i-th word
                should.  contirbute to the loss cumputation. In practice it
                contains 1's for words which are parts of the decoded strings
                and 0's for the padding.

            loss_with_gt_ins: Operator computing the sequence loss when the
                decoder always gets the ground truth input.

            loss_with_decoded_ins: Operator computing the sequence loss when
                the decoder receives previously computed outputs on its input.

            decoded_seq: List of batches of decoded words. (When the
                decoder is fed with its own outputs.)

        """

        log("Initializing decoder, name: \"{}\"".format(name))
        self.encoders = encoders
        assert_type(self, 'vocabulary', vocabulary, Vocabulary)
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.use_attention = use_attention
        self.max_output_len = max_output_len
        self.scheduled_sampling = scheduled_sampling
        self.dropout_keep_prob = dropout_keep_prob
        self.copy_net = copy_net
        self.reused_word_embeddings = reused_word_embeddings
        self.use_noisy_activations = use_noisy_activations
        self.depth = depth
        self.name = name

        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  name="decoder_dropout_plc")

        self.is_training = tf.placeholder(tf.bool, name="decoder_is_training")

        self.learning_step = tf.Variable(0, name="learning_step",
                                         trainable=False)

        # tadyten nasledujici kus je rozhozeni podle poctu enkoderu
        # kdyz je jeden, tak berem rovnou jeho zakodovanej stav
        # kdyz je jich vic, tak je napred projektujem

        # lepsi by bylo dat nepovinnej atribut encoder_projection a tomu
        # dat jako hodnotu rnn_size. Ktera se odted bude inferovat automaticky
        # = bez projekce se konkatenujou vystupni stavy vsech enkoderu
        # a delka vyslednyho stavu bude rnn_size.

        if (len(encoders) == 1 and
                (rnn_size == encoders[0].encoded.get_shape()[1].value)):
            encoded = encoders[0].encoded
            log("Using encoder output without projection.")
        elif len(encoders) >= 1:
            with tf.variable_scope("encoders_projection"):
                encoded_concat = tf.concat(1, [e.encoded for e in encoders])
                concat_size = encoded_concat.get_shape()[1].value
                proj = tf.get_variable(name="project_encoders",
                                       shape=[concat_size, depth * rnn_size])
                encoded_concat_dropped = tf.nn.dropout(
                    encoded_concat, self.dropout_placeholder)
                proj_bias = tf.Variable(tf.zeros([depth * rnn_size]))
                encoded = tf.matmul(encoded_concat_dropped, proj) + proj_bias
        elif len(encoders) == 0:  # if we want to train just LM
            encoded = tf.zeros([rnn_size])
            log("No encoder - language model only.")

        # TODO OTAZKA je, jestli to je ve spravnym poradi
        self.encoded = encoded
        encoded = tf.nn.dropout(encoded, self.dropout_placeholder)

        # tenhle kus pode mnou je deklarovani placeholderu pro vstupy dekoderu
        # placeholdery se vrazi do kolekce dec_endoder_ins, ktera
        # se asi nikde nepouziva
        # self.targets je self.gt_inputs posunuty o jedno doleva

        self.gt_inputs = []
        with tf.variable_scope("decoder_inputs"):
            for i in range(max_output_len + 2):
                dec = tf.placeholder(tf.int64, [None],
                                     name='decoder{0}'.format(i))
                tf.add_to_collection('dec_encoder_ins', dec)
                self.gt_inputs.append(dec)

        self.targets = self.gt_inputs[1:]

        # tenhle kousek zadefinovava vahy na vstup. je jich tolik co
        # targetu, a nejspis obsahujou jen jednicky a nuly podle toho,
        # jestli uz jsme za koncem vstupni vety nebo ne.
        # tohle by se melo s prechodem na dynamic rnn uplne vyhodit

        self.weights_ins = []
        with tf.variable_scope("input_weights"):
            for _ in range(len(self.targets)):
                self.weights_ins.append(tf.placeholder(tf.float32, [None]))

        # nasleduje kod samotnyho decoderu ve vlastnim scopu
        # proc veci nade mnou jsou jinym vlastnim scopu, to nevim

        with tf.variable_scope('decoder'):

            # deklarovani promennych pro vahy a biasy pro prechod ze
            # stavu na vystupni vrstvu
            # proc tady neni get_variable? to pouziva uniform unit scaling
            # initializer, coz je prinejmensim vic cool nazev

            decoding_w = tf.Variable(
                tf.random_uniform([rnn_size, len(vocabulary)], -0.5, 0.5),
                name="state_to_word_W")

            decoding_b = tf.Variable(
                tf.fill([len(vocabulary)], - math.log(len(vocabulary))),
                name="state_to_word_b")

            # pokud nepouzivame sdileny embeddingy, vytvorime si vlastni
            # to slouzi jako mapovani ze slovniku na vektor, kterej se dava
            # na vstup dekoderu v kazdym time-stepu
            # pro sdileni embeddingu je zapotrebi, aby mely stejnou velikost

            if reused_word_embeddings is None:
                decoding_em = tf.Variable(
                    tf.random_uniform([len(vocabulary), embedding_size],
                                      -0.5, 0.5),
                    name="word_embeddings")
            else:
                decoding_em = reused_word_embeddings.embedding_matrix

            # vyrobime embeddovany ground-truth inputy a dropoutujem
            # pouzivaj se pri trenovani

            embedded_gt_inputs = [tf.nn.embedding_lookup(decoding_em, o)
                                  for o in self.gt_inputs[:-1]]

            embedded_gt_inputs = [tf.nn.dropout(i, self.dropout_placeholder)
                                  for i in embedded_gt_inputs]

            # zadefinujem funkci, ktera nam pro dany stav vrati logity
            # tohle se bude muset predelat, je tu i ten copynet
            # logity sou dropoutlej stav vynasobenej s vahovou matici
            # vystupni a pricteny biasy

            def standard_logits(state):
                state = tf.nn.dropout(state, self.dropout_placeholder)
                return tf.matmul(state, decoding_w) + decoding_b, None

            logit_function = standard_logits

            # COPY NET
            # tomuhle se ted nebudu venovat

            if copy_net:
                # This is implementation of Copy-net
                # (http://arxiv.org/pdf/1603.06393v2.pdf)
                encoder_input_indices, copy_states, copy_mask = copy_net
                copy_tensor_dropped = tf.nn.dropout(copy_states,
                                                    self.dropout_placeholder)
                copy_tensors = [tf.squeeze(t, [1])
                                for t in tf.split(1, max_output_len + 2,
                                                  copy_tensor_dropped)]

                copy_features_size = copy_states.get_shape()[2].value

                # first we do the learned projection of the ecnoder outputs
                copy_w = tf.get_variable(name="copy_W",
                                         shape=[copy_features_size, rnn_size])

                projected_inputs = tf.concat(
                    1, [tf.expand_dims(tf.matmul(c, copy_w), 1)
                        for c in copy_tensors])

                batch_size = tf.shape(encoder_input_indices[0])[0]

                # tensor of batch numbers for indexing in a sparse vector
                batch_range = tf.range(start=0, limit=batch_size)
                batch_time_vocabulary_shape = tf.concat(
                    0, [tf.expand_dims(batch_size, 0),
                        tf.constant(len(vocabulary), shape=[1])])

                ones = tf.ones(tf.expand_dims(batch_size, 0))

                vocabulary_shaped_list = []
                for slice_indices in encoder_input_indices:
                    complete_indices = tf.concat(
                        1, [tf.expand_dims(batch_range, 1),
                            tf.expand_dims(slice_indices, 1)])

                    vocabulary_shaped = tf.sparse_to_dense(
                        complete_indices, batch_time_vocabulary_shape, ones)

                    vocabulary_shaped_list.append(vocabulary_shaped)

                vocabulary_shaped_indices = tf.concat(
                    1, [tf.expand_dims(v, 1) for v in vocabulary_shaped_list])

                def copy_net_logit_function(state):
                    state = tf.nn.dropout(state, self.dropout_placeholder)

                    # the logits for generating the next word are computed in
                    # the standard way
                    generate_logits = tf.matmul(state, decoding_w) + decoding_b

                    # Equation 8 in the paper ... in shape of source sentence
                    # (batch x time)
                    copy_logits_in_time = tf.reduce_sum(
                        projected_inputs * tf.expand_dims(state, 1), [2])

                    # mask out the padding in exponential domain
                    copy_logits_in_time_exp_masked = tf.exp(
                        tf.minimum([[80.0]], copy_logits_in_time)) * copy_mask

                    #  ... in shape of vocabulary (batch x time x vocabulary)
                    copy_logits_in_vocabulary = tf.expand_dims(
                        copy_logits_in_time_exp_masked,
                        2) * vocabulary_shaped_indices

                    # Equation 6 without normalization
                    copy_logits_exp = tf.reduce_sum(copy_logits_in_vocabulary,
                                                    [1])

                    logits_exp = copy_logits_exp \
                        + tf.exp(tf.minimum([[80.0]], generate_logits))

                    return (tf.log(tf.maximum([[1e-40]], logits_exp)),
                            copy_logits_in_time)

                logit_function = copy_net_logit_function

            # KONEC COPY-NETU
            # Tohle pod nama jsou dve loop functions. Loop function je funkce
            # ktera se pouziva za run-timu. Bere stav a cislo kroku v case
            # a vraci vstup do dalsiho kroku po embeddovani a dropoutu

            def loop(prev_state, _):
                # it takes the previous hidden state, finds the word and
                # formats it as input for the next time step ... used in the
                # decoder in the "real decoding scenario"
                out_activation, _ = logit_function(prev_state)
                prev_word_index = tf.argmax(out_activation, 1)
                next_step_embedding = tf.nn.embedding_lookup(decoding_em,
                                                             prev_word_index)

                return tf.nn.dropout(next_step_embedding,
                                     self.dropout_placeholder)

            # tahle loop function je pro scheduled sampling
            # scheduled sampling trenuje napred na zlatejch datech a postupem
            # casu zvolna prepina na loop function. Tahle konkretne to dela
            # pro kazdou trenovaci instanci v batchi zvlast.

            def sampling_loop(prev_state, i):
                """
                Loop function performing the scheduled sampling
                (http://arxiv.org/pdf/1506.03099v3.pdf) with the inverse
                sigmoid decay.
                """
                threshold = scheduled_sampling / (scheduled_sampling + tf.exp(
                    tf.to_float(self.learning_step) / scheduled_sampling))

                condition = tf.less_equal(
                    tf.random_uniform(tf.shape(embedded_gt_inputs[0])),
                    threshold)

                return tf.select(condition, embedded_gt_inputs[i],
                                 loop(prev_state, i))

            gt_loop_function = sampling_loop if scheduled_sampling else None

            # Tahle funkce tu strasi kvuli tomu, abychom mohli vybrat
            # bunku, ktera se pouzije jako RNN cell. Jednak ty noisy
            # activations nepomahaly a jednak bych to stejne cely vyhodil
            # Dale tu je kod, kterej ty bunky vydropoutuje a udela z nich
            # multirnncell (v pripade ze bychom chteli hlubsi rekurentni cast)

            def get_rnn_cell():
                if use_noisy_activations:
                    return NoisyGRUCell(rnn_size, training=self.is_training)
                else:
                    return tf.nn.rnn_cell.GRUCell(rnn_size)

            decoder_cells = [get_rnn_cell()]

            for _ in range(1, depth):
                decoder_cells[-1] = tf.nn.rnn_cell.DropoutWrapper(
                    decoder_cells[-1],
                    output_keep_prob=self.dropout_placeholder)

                decoder_cells.append(get_rnn_cell())

            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells, state_is_tuple=False)

            # A ted prichazi na radu attention. To se jen kouknem na encodery,
            # jestli ho maji zadefinovanej nebo ne

            if use_attention:
                attention_objects = [e.attention_object
                                     for e in encoders if e.attention_object]
            else:
                attention_objects = []

            # A ted samotna dekodovaci procedura. Tahle prvni vraci vystupy
            # s pouzitim zlatych vstupu (pri trenovani)

            rnn_outputs_gt_ins, _ = attention_decoder(
                embedded_gt_inputs, encoded, attention_objects, embedding_size,
                cell=decoder_cell, loop_function=gt_loop_function)

            tf.get_variable_scope().reuse_variables()

            # Tady to dolejc je dekodovaci procedura pro run-time, takze
            # s pouzitim loop functioně
            # Proc je to placeholder? Proc to neni konstanta?

            self.go_symbols = tf.placeholder(tf.int32, shape=[None],
                                             name="decoder_go_symbols")

            decoder_inputs = [tf.nn.embedding_lookup(decoding_em,
                                                     self.go_symbols)]

            decoder_inputs += [None for _ in range(self.max_output_len)]

            rnn_outputs_decoded_ins, _ = attention_decoder(
                decoder_inputs, encoded, attention_objects, embedding_size,
                cell=decoder_cell, loop_function=loop)

            self.hidden_states = rnn_outputs_decoded_ins

            # KONEC decoder scope

        def get_decoded(rnn_outputs):
            logits = []
            decoded = []
            copynet_logits = []

            for out in rnn_outputs:
                out_activation, logits_in_time = logit_function(out)

                if copy_net:
                    copynet_logits.append(logits_in_time)

                logits.append(out_activation)
                decoded.append(tf.argmax(out_activation[:, 1:], 1) + 1)

            return decoded, logits, copynet_logits

        # decoding a loss s ground truth (behem trenovani)

        _, self.gt_logits, _ = get_decoded(rnn_outputs_gt_ins)

        self.loss_with_gt_ins = tf.nn.seq2seq.sequence_loss(
            self.gt_logits, self.targets, self.weights_ins,
            len(vocabulary))

        self.cost = self.loss_with_gt_ins

        # decoding a loss s loop function (runtime)

        self.decoded_seq, self.decoded_logits, self.copynet_logits = \
            get_decoded(rnn_outputs_decoded_ins)

        self.loss_with_decoded_ins = tf.nn.seq2seq.sequence_loss(
            self.decoded_logits, self.targets, self.weights_ins,
            len(vocabulary))

        # Tady pode mnou sou sumary. To je vsechno co se bude logovat do
        # tensorboardu.

        tf.scalar_summary('train_loss_with_gt_intpus',
                          self.loss_with_gt_ins,
                          collections=["summary_train"])

        tf.scalar_summary('train_loss_with_decoded_inputs',
                          self.loss_with_decoded_ins,
                          collections=["summary_train"])

        tf.scalar_summary('train_optimization_cost', self.cost,
                          collections=["summary_train"])

        log("Decoder initalized.")

    def feed_dict(self, dataset, train=False):
        sentences = dataset.get_series(self.data_id, allow_none=True)
        res = {}

        start_token_index = self.vocabulary.get_word_index(START_TOKEN)
        res[self.go_symbols] = np.repeat(start_token_index, len(dataset))

        if sentences is not None:
            sentnces_tensors, weights_tensors = \
                self.vocabulary.sentences_to_tensor(sentences,
                                                    self.max_output_len)

            for weight_plc, weight_tensor in zip(self.weights_ins,
                                                 weights_tensors):
                res[weight_plc] = weight_tensor

            for words_plc, words_tensor in zip(self.gt_inputs,
                                               sentnces_tensors):
                res[words_plc] = words_tensor

        if train:
            res[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            res[self.dropout_placeholder] = 1.0

        return res

    @property
    def train_loss(self):
        return self.loss_with_gt_ins

    @property
    def runtime_loss(self):
        return self.loss_with_decoded_ins

    @property
    def runtime_logprobs(self):
        return [tf.nn.log_softmax(l) for l in self.decoded_logits]

    @property
    def decoded(self):
        return self.decoded_seq



def attention_decoder(decoder_inputs, initial_state, attention_objects,
                      embedding_size, cell, output_size=None,
                      loop_function=None, dtype=tf.float32, scope=None):

    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(scope or "attention_decoder"):
        batch_size = tf.shape(decoder_inputs[0])[0]    # Needed for reshaping.

        # do manualy broadcasting of the initial state if we want it
        # to be the same for all inputs
        if len(initial_state.get_shape()) == 1:
            state_size = initial_state.get_shape()[0].value
            initial_state = tf.reshape(tf.tile(initial_state,
                                               tf.shape(decoder_inputs[0])[:1]),
                                       [-1, state_size])

        state = initial_state
        outputs = []
        prev = None

        def initialize(attention_obj, batch_size, dtype):
            batch_attn_size = tf.pack([batch_size, attention_obj.attn_size])
            initial = tf.zeros(batch_attn_size, dtype=dtype)
            # Ensure the second shape of attention vectors is set.
            initial.set_shape([None, attention_obj.attn_size])
            return initial

        attns = [initialize(a, batch_size, dtype) for a in attention_objects]

        states = []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right
            # size.
            x = tf.nn.seq2seq.linear([inp] + attns, embedding_size, True)
            # Run the RNN.

            cell_output, state = cell(x, state)
            states.append(state)
            # Run the attention mechanism.
            attns = [a.attention(state) for a in attention_objects]

            if attns:
                with tf.variable_scope("AttnOutputProjection"):
                    output = tf.nn.seq2seq.linear([cell_output] + attns, output_size,
                                             True)
            else:
                output = cell_output

            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, states
