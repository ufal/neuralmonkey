import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from termcolor import colored
import regex as re

from utils import log

try:
    #pylint: disable=unused-import,bare-except,invalid-name
    from typing import Dict, List, Union, Tuple
    from decoder import Decoder
    Hypothesis = Tuple[float, List[int]]
    Feed_dict = Dict[tf.Tensor, np.Array]
except:
    pass

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


def tokenize_char_seq(chars):
    return word_tokenize("".join(chars))


def feed_dicts(dataset, batch_size, coders, train=False):
    """

    This function ensures all encoder and decoder objects feed their the data
    they need from the dataset.
    """
    dicts = [{} for _ in range(len(dataset) / batch_size + int(len(dataset) % batch_size > 0))]

    for coder in coders:
        coder.feed_dict(dataset, batch_size, dicts=dicts, train=train)

    return dicts

# TODO postprocess, copynet, beamsearch will be hidden in runner


def training_loop(sess, epochs, trainer, all_coders, decoder, batch_size,
                  train_dataset, val_dataset,
                  postprocess, log_directory,
                  evaluation_functions,
                  use_copynet=False,
                  test_dataset=None,
                  use_beamsearch=False,
                  initial_variables=None,
                  test_run=False):

    """

    Performs the training loop for given graph and data.

    Args:

        sess: TF Session.

        epochs: Number of epochs for which the algoritm will learn.

        trainer: The trainer object containg the TensorFlow code for computing
            the loss and optimization operation.

        decoder: The decoder object.

        train_dataset:

        val_dataset:

        postprocess: Function that takes the output sentence as produced by the
            decoder and transforms into tokenized sentence.

        log_directory: Directory where the TensordBoard log will be generated.
            If None, nothing will be done.

        evaluation_functions: List of evaluation functions. The last function
            is used as the main. Each function accepts list of decoded sequences
            and list of reference sequences and returns a float.

        use_copynet: Flag whether the copying mechanism is used.

        use_beamsearch:

        initial_variables: Either None or file where the variables are stored.
            Training then starts from the point the loaded values.

    """

    evaluation_labels = [f.__name__ for f in evaluation_functions]
    if postprocess is None:
        postprocess = lambda s: s
    log("Starting training")
    step = 0
    seen_instances = 0

    saver = tf.train.Saver()

    if initial_variables:
        saver.restore(sess, initial_variables)

    variables_file = log_directory+'/variables.data'
    saver.save(sess, variables_file)

    if log_directory:
        tb_writer = tf.train.SummaryWriter(log_directory, sess.graph_def)

    max_score = 0.0
    max_score_epoch = 0
    max_score_batch_no = 0

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
            for j, wrd in enumerate(s):
                if wrd == '<unk>':
                    selected = np.argmax(copy_logits[j][i])

                    ## Copynet can generate <pad> tokens from outside the sentence
                    if selected < len(copy_s) and selected != 0:
                        decoded_sentences[i][j] = copy_s[selected-1]

        return decoded_sentences

    val_feed_dicts = feed_dicts(val_dataset, 1 if use_beamsearch else batch_size, all_coders, train=False)
    val_tgt_sentences = val_dataset.series[decoder.data_id]

    try:
        for i in range(epochs):
            print ""
            log("Epoch {} starts".format(i + 1), color='red')

            train_dataset.shuffle()
            train_feed_dicts = feed_dicts(train_dataset, batch_size, all_coders, train=True)
            batched_targets = train_dataset.batch_serie(decoder.data_id, batch_size)

            for batch_n, (batch_feed_dict, batch_sentences) in \
                    enumerate(zip(train_feed_dicts, batched_targets)):

                step += 1
                seen_instances += len(batch_sentences)
                if step % 20 == 1:

                    computation = trainer.run(sess, batch_feed_dict, batch_sentences, verbose=True)

                    decoded_sentences = \
                            decoder.vocabulary.vectors_to_sentences(\
                            computation[-decoder.max_output_len - 1:])

                    decoded_sentences = [postprocess(s) for s in decoded_sentences]

                    evaluation_result = \
                            [f(decoded_sentences, batch_sentences) for f in evaluation_functions]

                    eval_string = "    ".join(["{}: {:.2f}".format(name, value) for name, value \
                        in zip(evaluation_labels, evaluation_result)])

                    log("opt. loss: {:.4f}    dec. loss: {:.4f}    ".\
                            format(computation[2], computation[1]) + eval_string)

                    if log_directory:
                        summary_str = computation[3]
                        tb_writer.add_summary(summary_str, seen_instances)
                        #histograms_str = computation[4]
                        #tb_writer.add_summary(histograms_str, seen_instances)
                        external_str = \
                                tf.Summary(value=[tf.Summary.Value(tag="train_"+name,
                                                                   simple_value=value) \
                                for name, value in zip(evaluation_labels, evaluation_result)])

                        tb_writer.add_summary(external_str, seen_instances)
                else:
                    trainer.run(sess, batch_feed_dict, batch_sentences, verbose=False)

                if step % 500 == (61 if test_run else 499):
                    decoded_val_sentences = []

                    for val_batch_n, (val_batch_feed_dict) in enumerate(val_feed_dicts):
                        if use_beamsearch:
                             decoded_s = beamsearch(sess, decoder, val_batch_feed_dict)
                             decoded_val_sentences_batch = [[decoder.vocabulary.index_to_word[i] for i in decoded_s[1:]]]
                             if val_batch_n % 100 == 99:
                                 log("Beamsearch: " + str(val_batch_n + 1) + " sentences done.")
                        else:
                            computation = sess.run([decoder.loss_with_decoded_ins,
                                                    decoder.loss_with_gt_ins, trainer.summary_val] \
                                                     + decoder.copynet_logits + decoder.decoded_seq,
                                                   feed_dict=val_batch_feed_dict)
                            decoded_val_sentences_batch = \
                                    decoder.vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])

                            #if use_copynet: # TODO beamsearch nefunguje s copynetem
                            #    decoded_val_sentences_batch = \
                            #            copynet_substitute(decoded_val_sentences_batch, val_copy_sentences, computation)

                        decoded_val_sentences += decoded_val_sentences_batch

                    decoded_val_sentences = [postprocess(s) for s in decoded_val_sentences]
                    evaluation_result = \
                            [f(decoded_val_sentences, val_tgt_sentences) for f in evaluation_functions]

                    eval_string = "    ".join(["{}: {:.2f}".format(name, value) for name, value \
                        in zip(evaluation_labels, evaluation_result)])


                    if evaluation_result[-1] > max_score:
                        max_score = evaluation_result[-1]
                        max_score_epoch = i
                        max_score_batch_no = batch_n
                        saver.save(sess, variables_file)

                    print ""
                    log("Validation (epoch {}, batch number {}):".format(i + 1, batch_n), color='cyan')
                    if not use_beamsearch:
                        log("opt. loss: {:.4f}    dec. loss: {:.4f}    "\
                                .format(computation[1], computation[0]) + eval_string, color='cyan')
                    else:
                        log(eval_string, color='cyan')
                    log("max {} on validation: {:.2f} (in epoch {}, after batch number {})".\
                            format(evaluation_labels[-1], max_score,
                                   max_score_epoch, max_score_batch_no), color='cyan')

                    print ""
                    print "Examples:"
                    for sent, ref_sent in zip(decoded_val_sentences[:15], val_tgt_sentences):
                        print u"    {}".format(u" ".join(sent))
                        print colored(u"      ref.: {}".format(u" ".join(ref_sent)),
                                      color="magenta")
                    print ""

                    if log_directory:
                        if not use_beamsearch:
                            summary_str = computation[2]
                            tb_writer.add_summary(summary_str, seen_instances)
                            tb_writer.add_summary(external_str, seen_instances)
                        external_str = \
                            tf.Summary(value=[tf.Summary.Value(tag="val_"+name, simple_value=value)\
                                              for name, value in zip(evaluation_labels,
                                                                     evaluation_result)])

                        tb_writer.add_summary(external_str, seen_instances)
    except KeyboardInterrupt:
        log("Training interrupted by user.")

    saver.restore(sess, variables_file)
    log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}".format(evaluation_labels[-1], max_score, max_score_epoch))

#    if test_feed_dicts and batched_test_copy_sentences and test_output_file:
#        log("Translating test data and writing to {}".format(test_output_file))
#        decoded_test_sentences = []
#
#        for i, (test_feed_dict, test_copy_sentences) in enumerate(zip(test_feed_dicts, batched_test_copy_sentences)):
#            if use_beamsearch:
#                 decoded_test_sentence_indices = beamsearch(test_feed_dict)
#                 decoded_test_sentences_batch = [[decoder.decoder.vocabulary.index_to_word[i] \
#                         for i in decoded_test_sentence_indices][1:]]
#                 log(decoded_test_sentences_batch)
#            else:
#                computation = sess.run(decoder.copynet_logits + decoder.decoded_seq, feed_dict=test_feed_dict)
#                decoded_test_sentences_batch = decoder.vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])
#
#                #if use_copynet: # TODO beamsearch (porad) nefunguje s copynetem
#                #    decoded_test_sentences_batch = \
#                #        copynet_substitute(decoded_test_sentences_batch, test_copy_sentences, computation)
#
#            decoded_test_sentences += [postprocess(s) for s in decoded_test_sentences_batch]
#
#            with open(test_output_file.name, test_output_file.mode) as fout:
#
#                for sent in decoded_test_sentences:
#                    fout.write("{}\n".format(" ".join(sent)))
#                fout.close()

    log("Finished.")


