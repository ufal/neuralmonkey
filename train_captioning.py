#!/usr/bin/env python

import argparse, time
import numpy as np
import tensorflow as tf
import regex as re

from image_encoder import ImageEncoder, VectorImageEncoder
from decoder import Decoder
from vocabulary import Vocabulary
from learning_utils import log, training_loop, print_header, tokenize_char_seq, feed_dropout_and_train
from cross_entropy_trainer import CrossEntropyTrainer
from language_utils import untruecase

def shape(string):
    res_shape = [int(s) for s in string.split("x")]
    return res_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains the image captioning.')
    parser.add_argument("--train-images", type=argparse.FileType('rb'),
                        help="File with training images features", required=True)
    parser.add_argument("--val-images", type=argparse.FileType('rb'),
                        help="File with validation images features.", required=True)
    parser.add_argument("--tokenized-train-text", type=argparse.FileType('r'),
                        help="File with tokenized training target sentences.", required=True)
    parser.add_argument("--tokenized-val-text", type=argparse.FileType('r'), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maximum-output", type=int, default=20)
    parser.add_argument("--use-attention", type=bool, default=False)
    parser.add_argument("--embeddings-size", type=int, default=256)
    parser.add_argument("--scheduled-sampling", type=float, default=None)
    parser.add_argument("--decoder-rnn-size", type=int, default=256)
    parser.add_argument("--dropout-keep-prob", type=float, default=1.0)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--character-based", type=bool, default=False)
    parser.add_argument("--img-features-shape", type=shape, default='14x14x256', required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use-noisy-activations", type=bool, default=False)
    parser.add_argument("--initial-variables", type=str, default=None,
            help="File with saved variables for initialization.")
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

    if args.character_based:
        raise Exception("Not implemented.")
    else:
        train_sentences = [re.split(ur"[ @#-]", l.rstrip()) for l in args.tokenized_train_text][:len(train_images)]
        tokenized_train_senteces = train_sentences
        log("Loaded {} training sentences.".format(len(train_sentences)))
        val_sentences = [re.split(ur"[ @#-]", l.rstrip()) for l in args.tokenized_val_text][:len(val_images)]
        tokenized_val_sentences = val_sentences
        log("Loaded {} validation sentences.".format(len(val_sentences)))

    listed_val_sentences = [[postedit(s)] for s in tokenized_val_sentences]

    vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_sentences for w in s])

    log("Training vocabulary has {} words".format(len(vocabulary)))

    log("Buiding the TensorFlow computation graph.")
    dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    training_placeholder = tf.placeholder(tf.bool, name="is_training")
    if len(args.img_features_shape) == 1:
        encoder = VectorImageEncoder(args.img_features_shape[0], args.decoder_rnn_size, dropout_placeholder=dropout_placeholder)
    else:
        encoder = ImageEncoder(args.img_features_shape, args.decoder_rnn_size, dropout_placeholder=dropout_placeholder)
    decoder = Decoder([encoder], vocabulary, args.decoder_rnn_size, training_placeholder, embedding_size=args.embeddings_size,
            use_attention=args.use_attention, max_out_len=args.maximum_output, use_peepholes=True,
            scheduled_sampling=args.scheduled_sampling, dropout_placeholder=dropout_placeholder,
            use_noisy_activations=args.use_noisy_activations)

    def get_feed_dicts(images, sentences, batch_size, train=False):
        feed_dicts = encoder.feed_dict(images, batch_size)
        _, batched_sentences = decoder.feed_dict(sentences, batch_size, feed_dicts)
        feed_dropout_and_train(feed_dicts, dropout_placeholder,
                args.dropout_keep_prob, training_placeholder, train)

        return feed_dicts, batched_sentences

    trainer = CrossEntropyTrainer(decoder, args.l2_regularization)

    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())

    train_feed_dicts, batched_listed_train_sentences = \
            get_feed_dicts(train_images, train_sentences, args.batch_size, train=True)
    val_feed_dicts, batched_listed_val_sentences = \
            get_feed_dicts(val_images, val_sentences, args.batch_size, train=False)
    import ipdb; ipdb.set_trace()

    training_loop(sess, vocabulary, args.epochs, trainer, decoder,
                  train_feed_dicts, batched_listed_train_sentences,
                  val_feed_dicts, batched_listed_val_sentences, postedit,
                  "logs-captioning/"+str(int(time.time())),
                  False,
                  [[] for _ in batched_listed_train_sentences],
                  [[] for _ in batched_listed_val_sentences],
                  initial_variables=args.initial_variables)
