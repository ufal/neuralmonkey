#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf
import regex as re

from image_encoder import ImageEncoder, VectorImageEncoder
from decoder import Decoder
from vocabulary import Vocabulary
from learning_utils import log, training_loop, print_header, tokenize_char_seq, feed_dropout_and_train
from cross_entropy_trainer import CrossEntropyTrainer
import cli_options
from language_utils import untruecase, bleu_1, bleu_4_dedup, bleu_4
from decoding_function import Attention, CoverageAttention

if __name__ == "__main__":
    parser = cli_options.get_captioning_parser()
    args = parser.parse_args()

    print_header("IMAGE CAPTIONING ONLY", args)

    postedit = untruecase

    if len(args.img_features_shape) == 1 and args.use_attention:
        log("Attention can be used only with 3D image features.")
        exit()

    log("The training script started")
    train_images = np.load(args.train_images)
    args.train_images.close()
    log("Loaded training images.")
    val_images = np.load(args.val_images)
    args.val_images.close()
    log("Loaded validation images.")

    train_sentences = [re.split(ur"[ @#-]", l.rstrip()) for l in args.tokenized_train_text][:len(train_images)]
    tokenized_train_senteces = train_sentences
    log("Loaded {} training sentences.".format(len(train_sentences)))
    val_sentences = [re.split(ur"[ @#-]", l.rstrip()) for l in args.tokenized_val_text][:len(val_images)]
    tokenized_val_sentences = val_sentences
    log("Loaded {} validation sentences.".format(len(val_sentences)))

    vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_sentences for w in s])

    log("Training vocabulary has {} words".format(len(vocabulary)))

    log("Buiding the TensorFlow computation graph.")
    dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    training_placeholder = tf.placeholder(tf.bool, name="is_training")
    if len(args.img_features_shape) == 1:
        encoder = VectorImageEncoder(args.img_features_shape[0], args.decoder_rnn_size, dropout_placeholder=dropout_placeholder)
    else:
        encoder = ImageEncoder(args.img_features_shape, args.decoder_rnn_size, dropout_placeholder=dropout_placeholder, eval(args.use_attention))
    decoder = Decoder([encoder], vocabulary, args.decoder_rnn_size, training_placeholder, embedding_size=args.embeddings_size,
            use_attention=args.use_attention, max_out_len=args.maximum_output, use_peepholes=True,
            scheduled_sampling=args.scheduled_sampling, dropout_placeholder=dropout_placeholder,
            use_noisy_activations=args.use_noisy_activations)

    def get_feed_dicts(images, sentences, batch_size, train=False):
        feed_dicts = encoder.feed_dict(images, batch_size)
        _, batched_sentences = decoder.feed_dict(sentences, len(sentences), batch_size, feed_dicts)
        feed_dropout_and_train(feed_dicts, dropout_placeholder,
                args.dropout_keep_prob, training_placeholder, train)

        postprocessed_tgt = [[postedit(s) for s in batch] for batch in batched_sentences]

        return feed_dicts, postprocessed_tgt

    trainer = CrossEntropyTrainer(decoder, args.l2_regularization)
    if args.mixer:
        xent_calls, moving_calls = args.mixer
        trainer = Mixer(decoder, trainer, xent_calls, moving_calls)

    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())

    train_feed_dicts, batched_train_sentences = \
            get_feed_dicts(train_images, train_sentences, args.batch_size, train=True)
    val_feed_dicts, batched_val_sentences = \
            get_feed_dicts(val_images, val_sentences, args.batch_size, train=False)

    training_loop(sess, vocabulary, args.epochs, trainer, decoder,
                  train_feed_dicts, batched_train_sentences,
                  val_feed_dicts, batched_val_sentences, postedit,
                  "logs-captioning/"+str(int(time.time())),
                  [bleu_1, bleu_4_dedup, bleu_4],
                  False,
                  [[] for _ in batched_train_sentences],
                  [[] for _ in batched_val_sentences],
                  use_beamsearch=args.beamsearch,
                  initial_variables=args.initial_variables)
