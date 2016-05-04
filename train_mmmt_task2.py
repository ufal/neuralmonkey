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
    parser = cli_options.get_mmmt_task2_parser()
    args = parser.parse_args()

    print_header("MULTIMODAL CAPTIONING", args)

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

    train_src_sentences_1 = load_tokenized(args.train_source_1)
    log("Loaded {} training src sentences #1.".format(len(train_src_sentences_1)))
    train_src_sentences_2 = load_tokenized(args.train_source_2)
    log("Loaded {} training src sentences #2.".format(len(train_src_sentences_2)))
    train_src_sentences_3 = load_tokenized(args.train_source_3)
    log("Loaded {} training src sentences #3.".format(len(train_src_sentences_3)))
    train_src_sentences_4 = load_tokenized(args.train_source_4)
    log("Loaded {} training src sentences #4.".format(len(train_src_sentences_4)))
    train_src_sentences_5 = load_tokenized(args.train_source_5)
    log("Loaded {} training src sentences #5.".format(len(train_src_sentences_5)))

    val_src_sentences_1 = load_tokenized(args.val_source_1)
    log("Loaded {} validation src sentences #1.".format(len(val_src_sentences_1)))
    val_src_sentences_2 = load_tokenized(args.val_source_2)
    log("Loaded {} validation src sentences #2.".format(len(val_src_sentences_2)))
    val_src_sentences_3 = load_tokenized(args.val_source_3)
    log("Loaded {} validation src sentences #3.".format(len(val_src_sentences_3)))
    val_src_sentences_4 = load_tokenized(args.val_source_4)
    log("Loaded {} validation src sentences #4.".format(len(val_src_sentences_4)))
    val_src_sentences_5 = load_tokenized(args.val_source_5)
    log("Loaded {} validation src sentences #5.".format(len(val_src_sentences_5)))

    train_images = np.load(args.train_images)
    args.train_images.close()
    log("Loaded training images.")
    val_images = np.load(args.val_images)
    args.val_images.close()
    log("Loaded validation images.")

    if args.test_output_file:
        if not args.test_source_sentences or not args.test_translated_sentences:
            raise Exception("must supply src and trans sentences when want to translate test set")

        test_src_sentences_1 = load_tokenized(args.test_source_1)
        log("Loaded {} testing src sentences #1.".format(len(test_src_sentences_1)))
        test_src_sentences_2 = load_tokenized(args.test_source_2)
        log("Loaded {} testing src sentences #2.".format(len(test_src_sentences_2)))
        test_src_sentences_3 = load_tokenized(args.test_source_3)
        log("Loaded {} testing src sentences #3.".format(len(test_src_sentences_3)))
        test_src_sentences_4 = load_tokenized(args.test_source_4)
        log("Loaded {} testing src sentences #4.".format(len(test_src_sentences_4)))
        test_src_sentences_5 = load_tokenized(args.test_source_5)
        log("Loaded {} testing src sentences #5.".format(len(test_src_sentences_5)))

        test_images = np.load(args.test_images)
        args.test_images.close()
        log("Loaded test images.")


    tgt_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_tgt_sentences for w in s])
    src_vocabulary = \
        Vocabulary(tokenized_text=[w for s in train_src_sentences_1 for w in s])
    src_vocabulary.add_tokenized_text([w for s in train_src_sentences_2 for w in s])
    src_vocabulary.add_tokenized_text([w for s in train_src_sentences_3 for w in s])
    src_vocabulary.add_tokenized_text([w for s in train_src_sentences_4 for w in s])
    src_vocabulary.add_tokenized_text([w for s in train_src_sentences_5 for w in s])

    log("Training tgt_vocabulary has {} words".format(len(tgt_vocabulary)))

    log("Buiding the TensorFlow computation graph.")
    dropout_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")
    training_placeholder = tf.placeholder(tf.bool, name="is_training")

    encoder_src_1 = SentenceEncoder(args.maximum_output, src_vocabulary,
                                    args.embeddings_size, args.encoder_rnn_size, dropout_placeholder,
                                    training_placeholder, use_noisy_activations=args.use_noisy_activations,
                                    attention_type=eval(args.use_attention), attention_fertility=3,
                                    name="source_encoder_1")
    encoder_src_2 = SentenceEncoder(args.maximum_output, src_vocabulary,
                                    args.embeddings_size, args.encoder_rnn_size, dropout_placeholder,
                                    training_placeholder, use_noisy_activations=args.use_noisy_activations,
                                    attention_type=eval(args.use_attention), attention_fertility=3,
                                    name="source_encoder_2", parent_encoder=encoder_src_1)
    encoder_src_3 = SentenceEncoder(args.maximum_output, src_vocabulary,
                                    args.embeddings_size, args.encoder_rnn_size, dropout_placeholder,
                                    training_placeholder, use_noisy_activations=args.use_noisy_activations,
                                    attention_type=eval(args.use_attention), attention_fertility=3,
                                    name="source_encoder_3", parent_encoder=encoder_src_1)
    encoder_src_4 = SentenceEncoder(args.maximum_output, src_vocabulary,
                                    args.embeddings_size, args.encoder_rnn_size, dropout_placeholder,
                                    training_placeholder, use_noisy_activations=args.use_noisy_activations,
                                    attention_type=eval(args.use_attention), attention_fertility=3,
                                    name="source_encoder_4", parent_encoder=encoder_src_1)
    encoder_src_5 = SentenceEncoder(args.maximum_output, src_vocabulary,
                                    args.embeddings_size, args.encoder_rnn_size, dropout_placeholder,
                                    training_placeholder, use_noisy_activations=args.use_noisy_activations,
                                    attention_type=eval(args.use_attention), attention_fertility=3,
                                    name="source_encoder_5", parent_encoder=encoder_src_1)

    if len(args.img_features_shape) == 1:
        encoder_img = VectorImageEncoder(args.img_features_shape[0],
                                         args.decoder_rnn_size,
                                         dropout_placeholder=dropout_placeholder)
    else:
        encoder_img = ImageEncoder(args.img_features_shape, args.decoder_rnn_size,
                                   dropout_placeholder=dropout_placeholder,
                                   attention_type=Attention)

    decoder = Decoder([encoder_src_1, encoder_src_2, encoder_src_3,
                       encoder_src_4, encoder_src_5, encoder_img], tgt_vocabulary, args.decoder_rnn_size,
                      training_placeholder,
                      embedding_size=args.embeddings_size, use_attention=args.use_attention,
                      max_out_len=args.maximum_output, use_peepholes=True,
                      scheduled_sampling=args.scheduled_sampling, dropout_placeholder=dropout_placeholder,
                      copy_net=None, reused_word_embeddings=None,
                      use_noisy_activations=args.use_noisy_activations, depth=args.decoder_depth)

    trainer = CrossEntropyTrainer(decoder, args.l2_regularization)

    if args.mixer:
        xent_calls, moving_calls = args.mixer
        trainer = Mixer(decoder, trainer, xent_calls, moving_calls)

    def get_feed_dicts(src_sentences_1, src_sentences_2, src_sentences_3, src_sentences_4, src_sentences_5,
                       tgt_sentences, images, batch_size, train=False):
        feed_dicts, _ = encoder_src_1.feed_dict(src_sentences_1, batch_size, train=train)
        _, _ = encoder_src_2.feed_dict(src_sentences_2, batch_size, train=train, dicts=feed_dicts)
        _, _ = encoder_src_3.feed_dict(src_sentences_3, batch_size, train=train, dicts=feed_dicts)
        _, _ = encoder_src_4.feed_dict(src_sentences_4, batch_size, train=train, dicts=feed_dicts)
        _, _ = encoder_src_5.feed_dict(src_sentences_5, batch_size, train=train, dicts=feed_dicts)
        _ = encoder_img.feed_dict(images, batch_size, dicts=feed_dicts)

        ## batched_tgt_sentences can be None, as well as tgt_sentences
        feed_dicts, batched_tgt_sentences = \
            decoder.feed_dict(tgt_sentences, len(tgt_sentences), batch_size, feed_dicts)

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
            get_feed_dicts(val_src_sentences_1, val_src_sentences_2, val_src_sentences_3,
                           val_src_sentences_4, val_src_sentences_5,
                           val_tgt_sentences, val_images,
                           args.batch_size, train=False)
    train_feed_dicts, batched_train_tgt_sentences = \
            get_feed_dicts(train_src_sentences_1, train_src_sentences_2, train_src_sentences_3,
                           train_src_sentences_4, train_src_sentences_5, train_tgt_sentences,
                           train_images, args.batch_size, train=True)

    if args.test_output_file:
        test_feed_dicts, _ = \
            get_feed_dicts(test_src_sentences_1, test_src_sentences_2, test_src_sentences_3,
                           test_src_sentences_4, test_src_sentences_5, test_tgt_sentences,
                           test_images, args.batch_size, train=True)
        training_loop(sess, tgt_vocabulary, args.epochs, trainer, decoder,
                      train_feed_dicts, batched_train_tgt_sentences,
                      val_feed_dicts, batched_val_tgt_sentences, postedit,
                      "logs-task2/"+str(int(time.time())),
                      [bleu_1, bleu_4_dedup, bleu_4],
                      False, # 3 lines of dummy copynet stuff
                      [[] for _ in batched_train_tgt_sentences],
                      [[] for _ in batched_val_tgt_sentences],
                      test_feed_dicts, [[] for _ in test_feed_dicts], args.test_output_file,
                      use_beamsearch=args.beamsearch,
                      initial_variables=args.initial_variables)
    else:
        training_loop(sess, tgt_vocabulary, args.epochs, trainer, decoder,
                      train_feed_dicts, batched_train_tgt_sentences,
                      val_feed_dicts, batched_val_tgt_sentences, postedit,
                      "logs-task2/"+str(int(time.time())),
                      [bleu_1, bleu_4_dedup, bleu_4],
                      False, # 3 lines of dummy copynet stuff
                      [[] for _ in batched_train_tgt_sentences],
                      [[] for _ in batched_val_tgt_sentences],
                      use_beamsearch=args.beamsearch,
                      initial_variables=args.initial_variables)
