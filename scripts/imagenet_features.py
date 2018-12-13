#!/usr/bin/env python3
"""Extract imagenet features from given images.

The script reads a list of pahts to images (specified by path prefix and list
of relative paths), process the images using an imagenet network and extract a
given convolutional map from the image. The maps are saved as numpy tensors in
files with a different prefix and the same relative path from this prefix
ending with .npz.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from neuralmonkey.dataset import Dataset, BatchingScheme
from neuralmonkey.encoders.imagenet_encoder import ImageNet
from neuralmonkey.logging import log
from neuralmonkey.readers.image_reader import single_image_for_imagenet


SUPPORTED_NETWORKS = [
    "vgg_16", "vgg_19", "resnet_v2_50", "resnet_v2_101", "resnet_v2_152"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=str, choices=SUPPORTED_NETWORKS,
                        help="Type of imagenet network.")
    parser.add_argument("--input-prefix", type=str, default="",
                        help="Prefix of the image path.")
    parser.add_argument("--output-prefix", type=str, default="",
                        help="Prefix of the path to the output numpy files.")
    parser.add_argument("--slim-models", type=str, required=True,
                        help="Path to SLIM models in cloned tensorflow/models "
                        "repository")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to the ImageNet model checkpoint.")
    parser.add_argument("--conv-map", type=str, required=False, default=None,
                        help="Name of the convolutional map that is.")
    parser.add_argument("--vector", type=str, required=False, default=None,
                        help="Name of the feed-forward layer.")
    parser.add_argument("--images", type=str,
                        help="File with paths to images or stdin by default.")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.conv_map is None == args.vector is None:
        raise ValueError(
            "You must provide either convolutional map or feed-forward layer.")

    if not os.path.exists(args.input_prefix):
        raise ValueError("Directory {} does not exist.".format(
            args.input_prefix))
    if not os.path.exists(args.output_prefix):
        raise ValueError("Directory {} does not exist.".format(
            args.output_prefix))

    if args.net.startswith("vgg_"):
        img_size = 224
        vgg_normalization = True
        zero_one_normalization = False
    elif args.net.startswith("resnet_v2"):
        img_size = 229
        vgg_normalization = False
        zero_one_normalization = True
    else:
        raise ValueError("Unspported network: {}.".format(args._net))

    log("Creating graph for the ImageNet network.")
    imagenet = ImageNet(
        name="imagenet", data_id="images", network_type=args.net,
        slim_models_path=args.slim_models, load_checkpoint=args.model_checkpoint,
        spatial_layer=args.conv_map, encoded_layer=args.vector)

    log("Creating TensorFlow session.")
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    log("Loading ImageNet model variables.")
    imagenet.load(session)

    if args.images is None:
        log("No input file provided, reading paths from stdin.")
        source = sys.stdin
    else:
        source = open(args.images)

    images = []
    image_paths = []

    def process_images():
        dataset = Dataset("dataset", {"images": np.array(images)},
                          BatchingScheme(batch_size=1), {})
        feed_dict = imagenet.feed_dict(dataset)

        fetch = imagenet.encoded if args.vector else imagenet.spatial_states
        feature_maps = session.run(fetch, feed_dict=feed_dict)

        for features, rel_path in zip(feature_maps, image_paths):
            npz_path = os.path.join(args.output_prefix, rel_path + ".npz")
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez(npz_path, features)
            print(npz_path)


    for img in source:
        img_path = os.path.join(args.input_prefix, img.rstrip())
        images.append(single_image_for_imagenet(
            img_path, img_size, img_size, vgg_normalization,
            zero_one_normalization))
        image_paths.append(img.rstrip())

        if len(images) >= args.batch_size:
            process_images()
            images = []
            image_paths = []
    process_images()

    if args.images is not None:
        source.close()


if __name__ == "__main__":
    main()
