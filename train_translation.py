#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf
import regex as re

from sentence_encoder import SentenceEncoder
from deep_sentence_encoder import DeepSentenceEncoder
from decoder import Decoder
from vocabulary import Vocabulary
from learning_utils import log, training_loop, print_header, tokenize_char_seq, load_tokenized, feed_dropout_and_train
from mixer import Mixer
from cross_entropy_trainer import CrossEntropyTrainer
import cli_options
from language_utils import untruecase, GermanPreprocessor, GermanPostprocessor, bleu_1, bleu_4_dedup, bleu_4

if __name__ == "__main__":
    parser = cli_options.get_translation_parser()
    args = parser.parse_args()

    print_header("TRANSLATION ONLY", args)

    postedit = untruecase
    preprocess = None
    if args.target_german:
        postedit = GermanPostprocessor()
        preprocess = GermanPreprocessor()

    train_tgt_sentences = load_tokenized(args.train_target_sentences, preprocess=preprocess)
    tokenized_train_tgt_sentences = train_tgt_sentences
    log("Loaded {} training tgt_sentences.".format(len(train_tgt_sentences)))
    val_tgt_sentences = load_tokenized(args.val_target_sentences, preprocess=preprocess)
    tokenized_val_tgt_sentences = val_tgt_sentences
    log("Loaded {} validation tgt_sentences.".format(len(val_tgt_sentences)))
    train_src_sentences = load_tokenized(args.train_source_sentences)
    log("Loaded {} training src_sentences.".format(len(train_src_sentences)))
    val_src_sentences = load_tokenized(args.val_source_sentences)
    log("Loaded {} validation src_sentences.".format(len(val_src_sentences)))

    tgt_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_tgt_sentences for w in s])
    src_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_src_sentences for w in s])

    log("Training tgt_vocabulary has {} words".format(len(tgt_vocabulary)))

    log("Buiding the TensorFlow computation graph.")
    dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    training_placeholder = tf.placeholder(tf.bool, name="is_training")

    if args.gru_bidi_depth is None:
        encoder = SentenceEncoder(args.maximum_output, src_vocabulary, args.embeddings_size,
                                  args.encoder_rnn_size, dropout_placeholder, training_placeholder,
                                  args.use_noisy_activations)
    else:
        encoder = DeepSentenceEncoder(args.maximum_output, src_vocabulary, args.embeddings_size,
                                      args.encoder_rnn_size, args.gru_bidi_depth,
                                      dropout_placeholder, training_placeholder,
                                      args.use_noisy_activations)

    decoder = Decoder([encoder], tgt_vocabulary, args.decoder_rnn_size, training_placeholder,
            embedding_size=args.embeddings_size,
            use_attention=args.use_attention, max_out_len=args.maximum_output, use_peepholes=True,
            scheduled_sampling=args.scheduled_sampling, dropout_placeholder=dropout_placeholder,
            use_noisy_activations=args.use_noisy_activations)

    def get_feed_dicts(src_sentences, tgt_sentences, batch_size, train=False):
        feed_dicts, batched_src_sentences = encoder.feed_dict(src_sentences, batch_size, train=train)
        _, batched_tgt_sentences = decoder.feed_dict(tgt_sentences, len(src_sentences), batch_size, feed_dicts)

        feed_dropout_and_train(feed_dicts, dropout_placeholder,
                args.dropout_keep_prob, training_placeholder, train)

        return feed_dicts, batched_src_sentences, batched_tgt_sentences

    trainer = CrossEntropyTrainer(decoder, args.l2_regularization)
    if args.mixer:
        xent_calls, moving_calls = args.mixer
        trainer = Mixer(decoder, trainer, xent_calls, moving_calls)

    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())


    val_feed_dicts, batched_val_src_sentences, batched_val_tgt_sentences = \
        get_feed_dicts(val_src_sentences, val_tgt_sentences,
                        1 if args.beamsearch else args.batch_size, train=False)
    train_feed_dicts, batched_train_src_sentences, batched_train_tgt_sentences = \
        get_feed_dicts(train_src_sentences, train_tgt_sentences, args.batch_size, train=True)

    training_loop(sess, tgt_vocabulary, args.epochs, trainer, decoder,
                  train_feed_dicts, batched_train_tgt_sentences,
                  val_feed_dicts, batched_val_tgt_sentences, postedit,
                  "logs-translation/"+str(int(time.time())),
                  [bleu_1, bleu_4_dedup, bleu_4],
                  False, batched_train_src_sentences, batched_val_src_sentences,
                  use_beamsearch=args.beamsearch,
                  initial_variables=args.initial_variables)
