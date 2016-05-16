#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf
import regex as re

from sentence_encoder import SentenceEncoder
from image_encoder import ImageEncoder, VectorImageEncoder
from decoder import Decoder
from vocabulary import Vocabulary
from learning_utils import log, training_loop, print_header, tokenize_char_seq, load_tokenized, feed_dropout_and_train
from language_utils import GermanPreprocessor, GermanPostprocessor
from cross_entropy_trainer import CrossEntropyTrainer
from copy_net_trainer import CopyNetTrainer
import cli_options
from language_utils import untruecase, bleu_1, bleu_4_dedup, bleu_4
from decoding_function import Attention, CoverageAttention

def shape(string):
    res_shape = [int(s) for s in string.split("x")]
    return res_shape

if __name__ == "__main__":
    parser = cli_options.get_parser('Trains the multimodal translation')
    cli_options.add_captioning_arguments(parser)
    cli_options.add_translation_arguments(parser)
    args = parser.parse_args()

    print_header("MULTIMODAL TRANSLATION", args)

    postedit = untruecase
    preprocess = None
    if args.target_german:
        postedit = GermanPostprocessor()
        preprocess = GermanPreprocessor()

    train_tgt_sentences = load_tokenized(args.train_target_sentences, preprocess=preprocess)
    tokenized_train_tgt_sentences = train_tgt_sentences
    log("Loaded {} training tgt_sentences.".format(len(train_tgt_sentences)))
    val_tgt_sentences = load_tokenized(args.val_target_sentences, preprocess=preprocess)
    log("Loaded {} validation tgt_sentences.".format(len(val_tgt_sentences)))

    train_src_sentences = load_tokenized(args.train_source_sentences)
    log("Loaded {} training src_sentences.".format(len(train_src_sentences)))
    val_src_sentences = load_tokenized(args.val_source_sentences)
    log("Loaded {} validation src_sentences.".format(len(val_src_sentences)))

    train_images = np.load(args.train_images)
    args.train_images.close()
    log("Loaded training images.")
    val_images = np.load(args.val_images)
    args.val_images.close()
    log("Loaded validation images.")

    if args.test_output_file:
        if not args.test_source_sentences:
            raise Exception("must supply src when want to translate test set")

        test_src_sentences = load_tokenized(args.test_source_sentences)
        log("Loaded {} test src_sentences.".format(len(test_src_sentences)))
        test_images = np.load(args.test_images)
        args.test_images.close()
        log("Loaded test images.")


    tgt_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_tgt_sentences for w in s])
    src_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_src_sentences for w in s])

    log("Training tgt_vocabulary has {} words".format(len(tgt_vocabulary)))

    log("Buiding the TensorFlow computation graph.")
    dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    training_placeholder = tf.placeholder(tf.bool, name="is_training")

    encoder_src = SentenceEncoder(args.maximum_output, src_vocabulary,
                                  args.embeddings_size, args.encoder_rnn_size, dropout_placeholder,
                                  training_placeholder, use_noisy_activations=args.use_noisy_activations,
                                  attention_type=eval(args.use_attention), attention_fertility=3, name="source_encoder")
    if len(args.img_features_shape) == 1:
        encoder_img = VectorImageEncoder(args.img_features_shape[0],
                                         args.decoder_rnn_size,
                                         dropout_placeholder=dropout_placeholder)
    else:
        encoder_img = ImageEncoder(args.img_features_shape, args.decoder_rnn_size,
                                   dropout_placeholder=dropout_placeholder,
                                   attention_type=Attention)

    copy_net = None
    reused_word_embeddings = None

    decoder = Decoder([encoder_src, encoder_img], tgt_vocabulary, args.decoder_rnn_size,
                      training_placeholder,
                      embedding_size=args.embeddings_size, use_attention=args.use_attention,
                      max_out_len=args.maximum_output, use_peepholes=True,
                      scheduled_sampling=args.scheduled_sampling, dropout_placeholder=dropout_placeholder,
                      copy_net=copy_net, reused_word_embeddings=reused_word_embeddings,
                      use_noisy_activations=args.use_noisy_activations, depth=args.decoder_depth)

    trainer = CrossEntropyTrainer(decoder, args.l2_regularization)

    run_batch_size = 1 if args.beamsearch else args.batch_size

    def get_feed_dicts(src_sentences, tgt_sentences, images, batch_size, train=False):
        feed_dicts, _ = encoder_src.feed_dict(src_sentences, batch_size, train=train)
        feed_dicts = encoder_img.feed_dict(images, batch_size, dicts=feed_dicts)

        ## batched_tgt_sentences can be None, as well as tgt_sentences
        feed_dicts, batched_tgt_sentences = \
            decoder.feed_dict(tgt_sentences, len(src_sentences), batch_size, feed_dicts)

        feed_dropout_and_train(feed_dicts, dropout_placeholder,
                args.dropout_keep_prob, training_placeholder, train)

        postprocessed_tgt = None
        if batched_tgt_sentences is not None:
            postprocessed_tgt = [[postedit(s) for s in batch] for batch in batched_tgt_sentences]

        return feed_dicts, postprocessed_tgt


    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())

    val_feed_dicts, batched_val_tgt_sentences = \
            get_feed_dicts(val_src_sentences, val_tgt_sentences,
                    val_images,
                    run_batch_size, train=False)
    train_feed_dicts, batched_train_tgt_sentences = \
            get_feed_dicts(train_src_sentences, train_tgt_sentences,
                    train_images, args.batch_size, train=True)

    if args.test_output_file:
        test_feed_dicts, _ = \
                get_feed_dicts(test_src_sentences, None,
                    test_images, run_batch_size, train=False)
        training_loop(sess, tgt_vocabulary, args.epochs, trainer, decoder,
                      train_feed_dicts, batched_train_tgt_sentences,
                      val_feed_dicts, batched_val_tgt_sentences, postedit,
                      "logs-mmmt/"+str(int(time.time())),
                      [bleu_1, bleu_4_dedup, bleu_4],
                      False,
                      [[] for _ in batched_train_tgt_sentences],
                      [[] for _ in batched_val_tgt_sentences],
                      test_feed_dicts, [[] for _ in test_feed_dicts], args.test_output_file,
                      use_beamsearch=args.beamsearch,
                      initial_variables=args.initial_variables)
    else:
        training_loop(sess, tgt_vocabulary, args.epochs, trainer, decoder,
                      train_feed_dicts, batched_train_tgt_sentences,
                      val_feed_dicts, batched_val_tgt_sentences, postedit,
                      "logs-postedit/"+str(int(time.time())),
                      [bleu_1, bleu_4_dedup, bleu_4],
                      False,
                      [[] for _ in batched_train_tgt_sentences],
                      [[] for _ in batched_val_tgt_sentences],
                      use_beamsearch=args.beamsearch,
                      initial_variables=args.initial_variables)
