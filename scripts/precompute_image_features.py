#!/usr/bin/env python3

import argparse
import os
import time
import numpy as np
import tensorflow as tf
from scipy.misc import imresize, imread

# tests: mypy

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


def main():
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

    file_content = args.network_file.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)

    images_placeholder = tf.placeholder("float", [None, 224, 224, 3])

    tf.import_graph_def(graph_def, input_map={"images": images_placeholder})
    print("Network graph loaded from disk.")

    graph = tf.get_default_graph()

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    print("Variables initialized.")

    image_paths = [os.path.join(args.image_dir, i.rstrip()) for i in args.image_list]

    image_batches = [np.array([load_image(p) for p in image_paths[i:i + 100]])
                     for i in range(0, len(image_paths), 100)]

    print("Images pre-loaded.")

    start = time.time()
    processed_batches = []
    for i, batch in enumerate(image_batches):
        it_start = time.time()
        feed_dict = {images_placeholder: batch}
        tensor = graph.get_tensor_by_name(args.layer)

        data = sess.run(tensor, feed_dict=feed_dict)
        processed_batches.append(data)
        it_time = time.time() - it_start
        print("Processed batch {} / {} in {:.4f}.".format(i + 1, len(image_batches), it_time))
    all_time = time.time() - start
    print("Done in {:.4f} seconds, i.e. {:.4f} per image.".format(
        all_time, all_time / len(image_paths)))

    np.save(args.output, np.concatenate(processed_batches))
    print("Image tensors saved to: {}".format(args.output))

if __name__ == "__main__":
    main()
