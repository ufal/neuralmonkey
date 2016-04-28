import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from termcolor import colored
import regex as re
import os, time

def log(message, color='yellow'):
    print "{}: {}".format(colored(time.strftime("%Y-%m-%d %H:%M:%S"), color), message)


def print_header(title, args):
    """
    Prints the title of the experiment and the set of arguments it uses.
    """
    print colored("".join("=" for _ in range(80)), 'green')
    print colored(title.upper(), 'green')
    print colored("".join("=" for _ in range(80)), 'green')
    print "Launched at {}".format(time.strftime("%Y-%m-%d %H:%M:%S"))

    print ""
    for arg in vars(args):
        value = getattr(args, arg)
        if type(value) == file:
            value_str = value.name
        else:
            value_str = str(value)
        dots_count = 78 - len(arg) - len(value_str)
        print "{} {} {}".format(arg, "".join(['.' for _ in range(dots_count)]), value_str)
    print ""

    os.system("echo last commit: `git log -1 --format=%H`")
    os.system("git --no-pager diff --color=always")
    print ""


def load_tokenized(text_file, preprocess=None):
    """
    Loads a tokenized text file a list of list of tokens.

    Args:

        text_file: An opened file.

        preprocess: A function/callable that (linguistically) preprocesses the
            sentences

    """

    if not preprocess:
        preprocess = lambda x: x

    return [preprocess(re.split(ur"[ ]", l.rstrip())) for l in text_file]



def load_char_based(test_file):
    """
    Loads a tokenized text for character-based decoding.
    """
    pass


def tokenize_char_seq(chars):
    return word_tokenize("".join(chars))


def corpus_bleu_deduplicated_unigrams(batch_sentences, decoded_sentences,
                                      weights, smoothing_function):

    deduplicated_sentences = []

    for sentence in decoded_sentences:

        last_w = None
        dedup_snt = []

        for word in sentence:
            if word != last_w:
                dedup_snt.append(word)
                last_w = word

        deduplicated_sentences.append(dedup_snt)

    return corpus_bleu(batch_sentences, deduplicated_sentences, weights,
                       smoothing_function)


def feed_dropout_and_train(dicts, dropout_placeholder, dropout_value,
        training_placeholder, is_training):
    for d in dicts:
        d[training_placeholder] = is_training
        if is_training:
            d[dropout_placeholder] = dropout_value
        else:
            d[dropout_placeholder] = 1.0



def training_loop(sess, vocabulary, epochs, trainer,
                  decoder, train_feed_dicts, train_tgt_sentences,
                  val_feed_dicts, val_tgt_sentences,
                  postprocess, tensorboard_log,
                  evaluation_functions,
                  use_copynet,
                  batched_train_copy_sentences, batched_val_copy_sentences,
                  test_feed_dicts=None, batched_test_copy_sentences=None, test_output_file=None,
                  char_based=False,
                  use_beamsearch=False,
                  initial_variables=None):

    """

    Performs the training loop for given graph and data.

    Args:

        sess: TF Session.

        vocabulary: Vocabulary used on the decoder side.

        epochs: Number of epochs for which the algoritm will learn.

        trainer: The trainer object containg the TensorFlow code for computing
            the loss and optimization operation.

        decoder: The decoder object.

        train_feed_dicts: List of feed dictionaires for training batches.

        train_tgt_sentences: List of batches of target training sentences for
            BLEU computation. Each sentence must be clsoed in one additional list
            (potentially more reference sentences for BLEU computation). Even if
            the character based decoding is done, these must be tokenized
            sentences.

        val_feed_dict: List of feed dictionaries for validation data batches.

        val_tgt_sentences: List of batches (=lists) of validation target
            sentences for BLEU computation.  Lists of lists (there may be multiple
            references for a sentece) of list of words. Each sentence must be
            clsoed in one additional list (potentially more reference sentences for
            BLEU computation). Even if the character based decoding is done, these
            must be tokenized sentences.

        postprocess: Function that takes the output sentence as produced by the
            decoder and transforms into tokenized sentence.

        tensorboard_log: Directory where the TensordBoard log will be generated.
            If None, nothing will be done.

        evaluation_functions: List of evaluation functions. The last function
            is used as the main. Each function accepts list of decoded sequences
            and list of reference sequences and returns a float.

        use_copynet: Flag whether the copying mechanism is used.

        batched_train_copy_sentences: Batched training sentences from which we
            copy tokens if copy mechanism is used.

        batched_val_copy_sentences: Batched validation sentences from which we
            copy if copy mechanism is used.

        initial_variables: Either None or file where the variables are stored.
            Training then starts from the point the loaded values.

    """

    evaluation_labels = [f.__name__ for f in evaluation_functions]
    log("Starting training")
    step = 0
    seen_instances = 0

    saver = tf.train.Saver()

    if initial_variables:
        saver.restore(sess, initial_variables)

    tmp_save_file = 'variable-'+str(time.time())+'.tmp'
    saver.save(sess, tmp_save_file)

    if tensorboard_log:
        tb_writer = tf.train.SummaryWriter(tensorboard_log, sess.graph_def)

    max_score = 0.0
    max_score_epoch = 0
    max_score_batch_no = 0
    val_tgt_sentences_flatten = [s for batch in val_tgt_sentences for s in batch]

    def copynet_substitute(decoded_sentences, copy_sentences, computation):
        """
        Substitutes the <unk> tokens with the tokens from the source encoder we are
        copying from.
        """
        copy_logits = computation[-(2*decoder.max_output_len)-2 : -decoder.max_output_len - 1]
        assert len(copy_logits) == decoder.max_output_len + 1 ## kdyby nahodou
        #assert len(computation) -(2*decoder.max_output_len)-2 == 5 ## neplati pri validaci
        assert len(decoded_sentences) == len(copy_sentences)

        for i, (s, copy_s) in enumerate(zip(decoded_sentences, copy_sentences)):
            for j, w in enumerate(s):
                if w == '<unk>':
                    selected = np.argmax(copy_logits[j][i])

                    ## Copynet can generate <pad> tokens from outside the sentence
                    if selected < len(copy_s) and selected != 0:
                        decoded_sentences[i][j] = copy_s[selected-1]

        return decoded_sentences

    try:
        for i in range(epochs):
            print ""
            log("Epoch {} starts".format(i + 1), color='red')

            for batch_n, (batch_feed_dict, batch_sentences, batch_copy_sentences) in \
                    enumerate(zip(train_feed_dicts, train_tgt_sentences, batched_train_copy_sentences)):

                step += 1
                seen_instances += len(batch_sentences)
                if step % 20 == 1:

                    computation = trainer.run(sess, batch_feed_dict, batch_sentences, verbose=True)

                    decoded_sentences = vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])
                    if use_copynet:
                        decoded_sentences = copynet_substitute(decoded_sentences, batch_copy_sentences, computation)

                    decoded_sentences = [postprocess(s) for s in decoded_sentences]


                    if char_based:
                        decoded_sentences = \
                                [tokenize_char_seq(chars) for chars in decoded_sentences]

                    evaluation_result = \
                            [f(decoded_sentences, batch_sentences) for f in evaluation_functions]

                    eval_string = "    ".join(["{}: {:.2f}".format(name, value) for name, value \
                        in zip(evaluation_labels, evaluation_result)])

                    log("opt. loss: {:.4f}    dec. loss: {:.4f}    ".format(computation[2], computation[1]) + eval_string)

                    if tensorboard_log:
                        summary_str = computation[3]
                        tb_writer.add_summary(summary_str, seen_instances)
                        #histograms_str = computation[4]
                        #tb_writer.add_summary(histograms_str, seen_instances)
                        external_str = tf.Summary(value=[tf.Summary.Value(tag="train_"+name, simple_value=value) \
                                for name, value in zip(evaluation_labels, evaluation_result)])

                        tb_writer.add_summary(external_str, seen_instances)
                else:
                    trainer.run(sess, batch_feed_dict, batch_sentences, verbose=False)

                if step % 500 == 499:
                    decoded_val_sentences = []

                    for val_batch_n, (val_batch_feed_dict, val_batch_sentences, val_copy_sentences) in \
                        enumerate (zip(val_feed_dicts, val_tgt_sentences, batched_val_copy_sentences)):

                        def expand(feed_dict, state, hypotheses):
                            feed_dict[decoder.encoded] = state
                            lh = len(hypotheses[0][1])
                            nh = len(hypotheses)
                            if lh == 2:
                                for k in feed_dict:
                                    sh = k.get_shape()
                                    if not sh == tf.TensorShape(None):
                                        if len(sh) == 1:
                                            feed_dict[k] = np.repeat(feed_dict[k], nh)
                                        elif len(sh) == 2:
                                            feed_dict[k] = np.repeat(np.array(feed_dict[k]), nh, axis=0)
                                        else:
                                            log("ERROR in expanding beamsearch \
                                                hypothesis")

                            for i, n in zip(decoder.gt_inputs, range(lh)):
                                for k in range(nh):
                                    feed_dict[i][k] = hypotheses[k][1][n]
                            probs, prob_i = sess.run([decoder.top10_probs[lh - 1][0],
                                             decoder.top10_probs[lh - 1][1]],
                                             feed_dict=feed_dict)
                            beam = []
                            for i in range(nh):
                                for p, x in zip(probs[i], prob_i[i]):
                                    beam.append((hypotheses[i][0] + p, hypotheses[i][1] + [x]))
                            return beam

                        def beamsearch(fd):
                            beam = [(1.0, [1])]
                            state = sess.run(decoder.encoded, fd)
                            for _ in range(len(decoder.decoded_probs)):
                                 new_beam = expand(fd, state, beam)
                                 new_beam.sort(reverse=True)
                                 beam = new_beam[:10]
                            return beam[0][1]

                        if use_beamsearch and val_batch_n % 100 == 99:
                             decoded_val_sentences.append(beamsearch(val_batch_feed_dict))
                             log("Beamsearch: " + str(val_batch_n) + " sentences done.")
                        else:
                            computation = sess.run([decoder.loss_with_decoded_ins,
                                decoder.loss_with_gt_ins, trainer.summary_val] \
                                    + decoder.copynet_logits + decoder.decoded_seq, feed_dict=val_batch_feed_dict)
                            decoded_val_sentences_batch = vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])

                            if use_copynet: # TODO beamsearch nefunguje s copynetem
                                decoded_val_sentences_batch = \
                                        copynet_substitute(decoded_val_sentences_batch, val_copy_sentences, computation)

                        decoded_val_sentences += [postprocess(s) for s in decoded_val_sentences_batch]

                    if char_based:
                        decoded_val_sentences = \
                                [tokenize_char_seq(chars) for chars in decoded_val_sentences]


                    evaluation_result = \
                            [f(decoded_sentences, batch_sentences) for f in evaluation_functions]

                    eval_string = "    ".join(["{}: {:.2f}".format(name, value) for name, value \
                        in zip(evaluation_labels, evaluation_result)])


                    if evaluation_result[-1] > max_score:
                        max_score = evaluation_result[-1]
                        max_score_epoch = i
                        max_score_batch_no = batch_n
                        saver.save(sess, tmp_save_file)

                    print ""
                    log("Validation (epoch {}, batch number {}):".format(i, batch_n), color='cyan')
                    log("opt. loss: {:.4f}    dec. loss: {:.4f}    "\
                            .format(computation[1], computation[0]) + eval_string, color='cyan')
                    log("max {} on validation: {:.2f} (in epoch {}, after batch number {})".\
                            format(evaluation_labels[-1], max_score, max_score_epoch, max_score_batch_no), color='cyan')

                    print ""
                    print "Examples:"
                    for sent, ref_sent in zip(decoded_val_sentences[:15], val_tgt_sentences_flatten):
                        print "    {}".format(" ".join(sent))
                        print colored("      ref.: {}".format(" ".join(ref_sent[0])), color="magenta")
                    print ""

                    if tensorboard_log:
                        summary_str = computation[2]
                        tb_writer.add_summary(summary_str, seen_instances)
                        tb_writer.add_summary(external_str, seen_instances)
                        external_str = tf.Summary(value=[tf.Summary.Value(tag="val_"+name, simple_value=value) \
                                for name, value in zip(evaluation_labels, evaluation_result)])

                        tb_writer.add_summary(external_str, seen_instances)
    except KeyboardInterrupt:
        log("Training interrupted by user.")

    saver.restore(sess, tmp_save_file)
    log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}".format(evaluation_labels[-1], max_score, max_score_epoch))

    if test_feed_dicts and batched_test_copy_sentences and test_output_file:
        log("Translating test data and writing to {}".format(test_output_file))
        decoded_test_sentences = []

        for i, (test_feed_dict, test_copy_sentences) in enumerate(zip(test_feed_dicts, batched_test_copy_sentences)):
            computation = sess.run(decoder.copynet_logits + decoder.decoded_seq, feed_dict=test_feed_dict)
            decoded_test_sentences_batch = vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])

            if use_copynet: # TODO beamsearch (porad) nefunguje s copynetem
                decoded_test_sentences_batch = \
                    copynet_substitute(decoded_test_sentences_batch, test_copy_sentences, computation)

            decoded_test_sentences += [postprocess(s) for s in decoded_test_sentences_batch]

            with open(test_output_file.name, test_output_file.mode) as fout:

                for sent in decoded_test_sentences:
                    fout.write("{}\n".format(" ".join(sent)))
                fout.close()

    log("Finished.")


