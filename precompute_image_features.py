#!/usr/bin/env python

import argparse
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imresize, imread


def load_image(path):
    # load image
    img = imread(path)

    # crop the image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]

    # resize to 224, 224
    resized_img = imresize(crop_img, (224, 224)) / 255.0
    return resized_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes the image with a pretrained network.')
    parser.add_argument("--network-file", type=argparse.FileType('rb'), required=True,
                        help="File with the image processig network.")
    parser.add_argument("--layer", type=str, required=True,
                        help="Identifier of the layer.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory with images")
    parser.add_argument("--image-list", type=argparse.FileType('r'), required=True,
                        help="File with list of images located in provided directory.")
    parser.add_argument("--output", type=argparse.FileType('wb'), required=True,
                        help="Output file.")
    args = parser.parse_args()

    fileContent = args.network_file.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    images_placeholder = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={ "images": images_placeholder })
    print "Network graph loaded from disk."

    graph = tf.get_default_graph()

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    print "Variables initialized."

    images = np.array([load_image(os.path.join(args.image_dir, i.rstrip())) for i in args.image_list])
    print "Images loaded."
    feed_dict = {images_placeholder: images}

    tensor = graph.get_tensor_by_name(args.layer)

    data = sess.run(tensor, feed_dict=feed_dict)
    print "Images processed."

    np.save(args.output, data)
    print "Image tensors saved to: {}".format(args.output)
