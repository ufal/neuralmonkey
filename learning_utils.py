import codecs
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from termcolor import colored
import regex as re

from utils import log, log_print

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


def get_eval_string(evaluation_functions, evaluation_res):
    """ Formats the external evaluation metric for the console output. """
    eval_string = \
        "    ".join(["{}: {:.2f}".format(f.__name__, evaluation_res[f])
                     for f in evaluation_functions[:-1]])
    if evaluation_functions:
        eval_string += colored(
            "    {}: {:.2f}".format(evaluation_functions[-1].__name__,
                                    evaluation_res[evaluation_functions[-1]]),
            attrs=['bold'])
    return eval_string


def initialize_tf(initial_variables, threads):
    """
    Initializes the TensorFlow session after the graph is built.

    Args:

        initial_variables: File with the saved TF variables.

    Returns:

        A tuple of the TF session and the the TF saver object.

    """
    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                            intra_op_parallelism_threads=threads))
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    if initial_variables:
        log("Loading variables from {}".format(initial_variables))
        saver.restore(sess, initial_variables)

    log("Session initialization done.")

    return sess, saver

def training_loop(sess, saver,
                  epochs, trainer, all_coders, decoder, batch_size,
                  train_dataset, val_dataset,
                  log_directory,
                  evaluation_functions,
                  runner,
                  test_datasets=[],
                  initial_variables=None,
                  logging_period=20,
                  validation_period=500,
                  postprocess=None):

    """
    Performs the training loop for given graph and data.

    Args:

        sess: TF Session.

        saver: TF saver object.

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

    if not postprocess:
        postprocess = lambda x, _: x

    evaluation_labels = [f.__name__ for f in evaluation_functions]
    step = 0
    seen_instances = 0

    saver = tf.train.Saver()

    if initial_variables:
        saver.restore(sess, initial_variables)

    variables_file = log_directory+'/variables.data'
    saver.save(sess, variables_file)

    if log_directory:
        log("Initializing TensorBoard summary writer.")
        tb_writer = tf.train.SummaryWriter(log_directory, sess.graph)
        log("TesorBoard writer initialized.")

    max_score = 0.0
    max_score_epoch = 0
    max_score_batch_no = 0

    val_tgt_sentences = \
        postprocess(val_dataset.series[decoder.data_id], val_dataset)

    log("Starting training")
    try:
        for i in range(epochs):
            log_print("")
            log("Epoch {} starts".format(i + 1), color='red')

            train_dataset.shuffle()
            log("Training dataset shuffled.")
            train_feed_dicts = feed_dicts(train_dataset, batch_size, all_coders, train=True)
            batched_targets = train_dataset.batch_serie(decoder.data_id, batch_size)

            for batch_n, (batch_feed_dict, batch_sentences) in \
                    enumerate(zip(train_feed_dicts, batched_targets)):

                step += 1
                seen_instances += len(batch_sentences)
                if step % logging_period == logging_period - 1:
                    computation = trainer.run(sess, batch_feed_dict, batch_sentences, verbose=True)

                    decoded_sentences = \
                            decoder.vocabulary.vectors_to_sentences(\
                            computation[-decoder.max_output_len - 1:])

                    decoded_sentences = postprocess(decoded_sentences, train_dataset)

                    evaluation_result = \
                            {f: f(decoded_sentences, batch_sentences) for f in evaluation_functions}

                    eval_string = get_eval_string(evaluation_functions, evaluation_result)

                    log("opt. loss: {:.4f}    dec. loss: {:.4f}    ".\
                            format(computation[2], computation[1]) + eval_string)

                    if log_directory:
                        summary_str = computation[3]
                        tb_writer.add_summary(summary_str, seen_instances)
                        #histograms_str = computation[4]
                        #tb_writer.add_summary(histograms_str, seen_instances)
                        external_str = \
                                tf.Summary(value=[tf.Summary.Value(tag="train_"+func.__name__,
                                                                   simple_value=value) \
                                for func, value in evaluation_result.iteritems()])

                        tb_writer.add_summary(external_str, seen_instances)
                else:
                    trainer.run(sess, batch_feed_dict, batch_sentences, verbose=False)

                if step % validation_period == validation_period - 1:
                    decoded_val_sentences, val_evaluation = \
                            run_on_dataset(sess, runner, all_coders, decoder, val_dataset,
                                           evaluation_functions, write_out=False)

                    eval_string = get_eval_string(evaluation_functions, val_evaluation)

                    this_score = val_evaluation[evaluation_functions[-1]]
                    if this_score > max_score:
                        max_score = this_score
                        max_score_epoch = i
                        max_score_batch_no = batch_n
                        saver.save(sess, variables_file)

                    log_print("")
                    log("Validation (epoch {}, batch number {}):"\
                            .format(i + 1, batch_n), color='blue')
                    log("opt. loss: {:.4f}    dec. loss: {:.4f}    "\
                            .format(val_evaluation['opt_loss'], val_evaluation['dec_loss']) \
                        + eval_string, color='blue')
                    log("max {} on validation: {:.2f} (in epoch {}, after batch number {})".\
                            format(evaluation_labels[-1], max_score,
                                   max_score_epoch, max_score_batch_no), color='blue')

                    log_print("")
                    log_print("Examples:")
                    for sent, ref_sent in zip(decoded_val_sentences[:15], val_tgt_sentences):
                        log_print(u"    {}".format(u" ".join(sent)))
                        log_print(colored(u"      ref.: {}".format(u" ".join(ref_sent)),
                                      color="magenta"))
                    log_print("")

                    if log_directory:
                        def format_name(name):
                            if hasattr(name, '__call__'):
                                return name.__name__
                            else:
                                return str(name)
                        external_str = \
                            tf.Summary(value=[tf.Summary.Value(tag="val_"+format_name(name),
                                                               simple_value=value)\
                                              for name, value in val_evaluation.iteritems()])

                        tb_writer.add_summary(external_str, seen_instances)
    except KeyboardInterrupt:
        log("Training interrupted by user.")

    saver.restore(sess, variables_file)
    log("Training finished. Maximum {} on validation data: {:.2f}, epoch {}"\
            .format(evaluation_labels[-1], max_score, max_score_epoch))

    for dataset in test_datasets:
        _, evaluation = run_on_dataset(sess, runner, all_coders, decoder,
                                       dataset, evaluation_functions, write_out=True)
        if evaluation:
            print_dataset_evaluation(dataset.name, evaluation)

    log("Finished.")


def run_on_dataset(sess, runner, all_coders, decoder, dataset,
                   evaluation_functions, write_out=False):
    """
    Applies the model on a dataset and eventualy writes outpus into a file.

    Args:

        session: TF session the model parameters are in.

        runner: A function that runs the code

        all_coders: List of all encoders and decoders in the model.

        decoder: The decoder used to generate outputs.

        dataset: The dataset on which the model will be executed.

        evaluation_functions: List of functions that are used for the model
            evaluation if the target data are provided.

        write_out: Flag whether the outputs should be printed to a file defined
            in the dataset object.

    Returns:

        Tuple of resulting sentences/numpy arrays, and evaluation results if
            they are available which are dictionary function -> value.

    """
    log("Applying model on dataset \"{}\"".format(dataset.name))
    result, opt_loss, dec_loss = runner(sess, dataset, all_coders)
    if write_out:
        if decoder.data_id in dataset.series_outputs:
            path = dataset.series_outputs[decoder.data_id]
            if isinstance(result, np.ndarray):
                np.save(path, result)
                log("Result saved as numpy array to \"{}\"".format(path))
            else:
                with codecs.open(path, 'w', 'utf-8') as f_out:
                    f_out.writelines([u" ".join(sent)+"\n" for sent in result])
                log("Result saved as plain text \"{}\"".format(path))
        else:
            log("There is no output file for dataset: {}"\
                    .format(dataset.name), color='red')

    evaluation = {}
    if decoder.data_id in dataset.series:
        test_targets = dataset.series[decoder.data_id]
        evaluation["opt_loss"] = opt_loss
        evaluation["dec_loss"] = dec_loss
        for func in evaluation_functions:
            evaluation[func] = func(result, test_targets)

    return result, evaluation


def print_dataset_evaluation(name, evaluation):
    line_len = 22
    log("Evaluating model on \"{}\"".format(name))

    log("    optimization loss:      {:.4f}".format(evaluation['opt_loss']))
    log("    runtime loss:           {:.4f}".format(evaluation['opt_loss']))

    for func in evaluation:
        if hasattr(func, '__call__'):
            name = func.__name__
            space = "".join([" " for _ in range(line_len - len(name))])
            log("    {}:{} {:.4f}".format(name, space, evaluation[func]))

    log_print("")

