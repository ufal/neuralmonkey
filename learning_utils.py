import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from termcolor import colored
import regex as re
import codecs

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


def training_loop(sess, epochs, trainer, all_coders, decoder, batch_size,
                  train_dataset, val_dataset,
                  log_directory,
                  evaluation_functions,
                  runner,
                  test_datasets=[],
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

                    #decoded_sentences = [postprocess(s) for s in decoded_sentences]

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

                if step % 500 == 1:# (61 if test_run else 499):
                    decoded_val_sentences, opt_loss, dec_loss = \
                            runner(sess, val_dataset, all_coders)
                    evaluation_result = \
                            [f(decoded_val_sentences, val_tgt_sentences)
                             for f in evaluation_functions]

                    eval_string = "    ".join(["{}: {:.2f}".format(name, value) for name, value \
                        in zip(evaluation_labels, evaluation_result)])


                    if evaluation_result[-1] > max_score:
                        max_score = evaluation_result[-1]
                        max_score_epoch = i
                        max_score_batch_no = batch_n
                        saver.save(sess, variables_file)

                    print ""
                    log("Validation (epoch {}, batch number {}):"\
                            .format(i + 1, batch_n), color='cyan')
                    log("opt. loss: {:.4f}    dec. loss: {:.4f}    "\
                            .format(opt_loss, dec_loss) + eval_string, color='cyan')
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
                        # TODO include validation loss
                        external_str = \
                            tf.Summary(value=[tf.Summary.Value(tag="val_"+name, simple_value=value)\
                                              for name, value in zip(evaluation_labels,
                                                                     evaluation_result)])

                        tb_writer.add_summary(external_str, seen_instances)
    except KeyboardInterrupt:
        log("Training interrupted by user.")

    saver.restore(sess, variables_file)
    log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}"\
            .format(evaluation_labels[-1], max_score, max_score_epoch))

    for dataset in test_datasets:
        log("Applying model on dataset \"{}\"".format(dataset.name))
        result, _, _ = runner(sess, dataset, all_coders)
        if decoder.data_id in dataset.series_outputs:
            path = dataset.series_outputs[decoder.data_id]
            if isinstance(result, np.ndarray):
                np.save(path, result)
                log("Numpy array saved to \"{}\"".format(path))
            else:
                with codecs.open(path, 'w', 'utf-8') as f:
                    f.writelines([u" ".join(sent)+"\n" for sent in result])
        else:
            log("There is no output file for dataset: {}".format(dataset.name), color='red')

    log("Finished.")
