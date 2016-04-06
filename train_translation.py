#!/usr/bin/env python

import argparse, time
import numpy as np
import tensorflow as tf
import regex as re

from sentence_encoder import SentenceEncoder
from decoder import Decoder
from vocabulary import Vocabulary
from learning_utils import log, training_loop, print_header, tokenize_char_seq, load_tokenized
from mixer import Mixer
from cross_entropy_trainer import CrossEntropyTrainer

def shape(string):
    res_shape = [int(s) for s in string.split("x")]
    return res_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains the translation.')
    parser.add_argument("--train-source-sentences", type=argparse.FileType('r'),
                        help="File with training source sentences", required=True)
    parser.add_argument("--val-source-sentences", type=argparse.FileType('r'),
                        help="File with validation source sentences.", required=True)
    parser.add_argument("--train-target-sentences", type=argparse.FileType('r'),
                        help="File with tokenized training target sentences.", required=True)
    parser.add_argument("--val-target-sentences", type=argparse.FileType('r'), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maximum-output", type=int, default=20)
    parser.add_argument("--use-attention", type=bool, default=False)
    parser.add_argument("--embeddings-size", type=int, default=256)
    parser.add_argument("--encoder-rnn-size", type=int, default=256)
    parser.add_argument("--decoder-rnn-size", type=int, default=512)
    parser.add_argument("--scheduled-sampling", type=float, default=None)
    parser.add_argument("--dropout-keep-prob", type=float, default=1.0)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--use-noisy-activations", type=bool, default=False)
    parser.add_argument("--character-based", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mixer", type=bool, default=False)
    args = parser.parse_args()

    print_header("TRANSLATION ONLY", args)

    if args.character_based:
        raise Exception("Not implemented")
        train_tgt_sentences = [list(re.sub(ur"[@#]", "", l.rstrip())) for l in args.tokenized_train_text]
        tokenized_train_tgt_sentences = [tokenize_char_seq(chars) for chars in train_tgt_sentences]
        log("Loaded {} training tgt_sentences.".format(len(train_tgt_sentences)))
        val_tgt_sentences = [list(re.sub(ur"[@#]", "", l.rstrip())) for l in args.tokenized_val_text]
        tokenized_val_tgt_sentences = [tokenize_char_seq(chars) for chars in val_tgt_sentences]
        log("Loaded {} validation tgt_sentences.".format(len(val_tgt_sentences)))
    else:
        train_tgt_sentences = load_tokenized(args.train_target_sentences)
        tokenized_train_tgt_sentences = train_tgt_sentences
        log("Loaded {} training tgt_sentences.".format(len(train_tgt_sentences)))
        val_tgt_sentences = load_tokenized(args.val_target_sentences)
        tokenized_val_tgt_sentences = val_tgt_sentences
        log("Loaded {} validation tgt_sentences.".format(len(val_tgt_sentences)))
        train_src_sentences = load_tokenized(args.train_source_sentences)
        log("Loaded {} training src_sentences.".format(len(train_src_sentences)))
        val_src_sentences = load_tokenized(args.val_source_sentences)
        log("Loaded {} validation src_sentences.".format(len(val_src_sentences)))

    listed_val_tgt_sentences = [[s] for s in tokenized_val_tgt_sentences]

    tgt_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_tgt_sentences for w in s])
    src_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_src_sentences for w in s])

    log("Training tgt_vocabulary has {} words".format(len(tgt_vocabulary)))

    log("Buiding the TensorFlow computation graph.")
    dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    training_placeholder = tf.placeholder(tf.bool, name="is_training")
    encoder = SentenceEncoder(args.maximum_output, src_vocabulary, args.embeddings_size,
                              args.encoder_rnn_size, dropout_placeholder, training_placeholder,
                              args.use_noisy_activations)
    decoder = Decoder([encoder], tgt_vocabulary, args.decoder_rnn_size, training_placeholder,
            embedding_size=args.embeddings_size,
            use_attention=args.use_attention, max_out_len=args.maximum_output, use_peepholes=True,
            scheduled_sampling=args.scheduled_sampling, dropout_placeholder=dropout_placeholder,
            use_noisy_activations=args.use_noisy_activations)

    def feed_dict(src_sentences, tgt_sentences, train=False):
        fd = {}

        fd[encoder.sentence_lengths] = np.array([min(args.maximum_output, len(s)) + 2 for s in src_sentences])
        src_vectors, _ = \
                src_vocabulary.sentences_to_tensor(src_sentences, args.maximum_output, train=train)
        for words_plc, words_tensor in zip(encoder.inputs, src_vectors):
            fd[words_plc] = words_tensor

        tgt_vectors, weights_tensors = \
            tgt_vocabulary.sentences_to_tensor(tgt_sentences, args.maximum_output, train=train)
        for weight_plc, weight_tensor in zip(decoder.weights_ins, weights_tensors):
            fd[weight_plc] = weight_tensor

        for words_plc, words_tensor in zip(decoder.gt_inputs, tgt_vectors):
            fd[words_plc] = words_tensor

        if train:
            fd[dropout_placeholder] = args.dropout_keep_prob
        else:
            fd[dropout_placeholder] = 1.0
        fd[training_placeholder] = train

        return fd

    val_feed_dict = feed_dict(val_src_sentences, val_tgt_sentences)

    if args.mixer:
        trainer = Mixer(decoder)
    else:
        trainer = CrossEntropyTrainer(decoder, args.l2_regularization)

    summary_train = tf.merge_summary(tf.get_collection("summary_train"))
    summary_test = tf.merge_summary(tf.get_collection("summary_test"))

    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())

    batched_train_tgt_sentences = \
            [train_tgt_sentences[start:start + args.batch_size] \
             for start in range(0, len(train_tgt_sentences), args.batch_size)]
    batched_listed_train_tgt_sentences = \
            [[[sent] for sent in batch] for batch in batched_train_tgt_sentences]

    batched_train_src_sentences = [train_src_sentences[start:start + args.batch_size]
             for start in range(0, len(train_src_sentences), args.batch_size)]
    train_feed_dicts = [feed_dict(src, tgt) \
            for src, tgt in zip(batched_train_src_sentences, batched_train_tgt_sentences)]

    training_loop(sess, tgt_vocabulary, args.epochs, trainer, decoder,
                  train_feed_dicts, batched_listed_train_tgt_sentences,
                  val_feed_dict, listed_val_tgt_sentences)
