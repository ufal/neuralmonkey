#!/usr/bin/python

import argparse
import numpy as np

from image_encoder import ImageEncoder
from decoder import Decoder
from vocabulary import Vocabulary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains the image captioning.')
    parser.add_argument("--train-images", type=lambda s: np.fromfile(s),
                        help="File with training images features", required=True)
    parser.add_argument("--valid-images", type=lambda s: np.fromfile(s),
                        help="File with validation images features.", required=True)
    parser.add_argument("--tokenized-train-text", type=argparse.FileType('r'),
                        help="File with tokenized training target sentences.", required=True)
    parser.add_argument("--tokenized-valid-text", type=argparse.FileType('r'), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    training_sentences = [l.rstrip().split(" ") for l in args.tokenized_train_text]
    validation_sentences = [l.rstrip().split(" ") for l in args.tokenized_train_text]

    vocabulary = \
        Vocabulary(tokenized_train_text=[w for s in training_sentences for w in s])


    encoder = ImageEncoder()
    # TODO parameters of the decoder to the command line
    decoder = Decoder(encoder, vocabulary, embedding_size=128,
            use_attention=True, max_out_len=20, use_peepholes=True, scheduled_sampling=10)

    valid_feed_dict = {
        encoder.image_features: args.valid_images,
        decoder.decoded_seq: vocabulary.sentences_to_tensor(validation_sentences)
    }

    optimze = tf.train.AdamOptimizer().optimze(decoder.cost)
    # gradients = optimizer.compute_gradients(cost)

    summary_train = tf.merge_summary(tf.get_collection("summary_train"))
    summary_test = tf.merge_summary(tf.get_collection("summary_test"))

    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())

    # TODO the training loop
    batch_count = len(training_sentences)
    for i in range(args.epochs):
        for start in range(0, len(training_sentences), args.batch_size):
            feed_dict = {
                encoder.image_features: args.train_images[start:start + args.batch_size],
                decoder.decoded_seq: training_sentences[start:start + args.batch_size]
            }
            sess.run([optimize], feed_dict=feed_dict)
        # TODO do validation after epoch and draw the tensorboard summaries


