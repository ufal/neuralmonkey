#!/usr/bin/env python

"""

Does the Image feature extraction by calling directly Caffe.
Based on tutorial http://www.marekrei.com/blog/transforming-images-to-feature-vectors/

"""
import sys
sys.path.append("caffe/python")
import os, argparse, caffe
import numpy as np
from learning_utils import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image feature extraction")
    parser.add_argument("--model-prototxt", type=str, required=True)
    parser.add_argument("--model-parameters", type=str, required=True)
    parser.add_argument("--img-mean", type=str, required=True)
    parser.add_argument("--feature-layer", type=str, required=True)
    parser.add_argument("--image-directory", type=str, required=True)
    parser.add_argument("--image-list", type=argparse.FileType('r'), required=True)
    parser.add_argument("--output-file", type=argparse.FileType('wb'), required=True)
    args = parser.parse_args()

    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_cpu()

    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(args.model_prototxt, args.model_parameters,
                           mean=np.load(args.img_mean).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    log("Model loaded")

    # Processing one image at a time, printint predictions and writing the vector to a file
    data = []
    for i, image_path in enumerate(args.image_list):
        image_path = image_path.strip()
        input_image = caffe.io.load_image(os.path.join(args.image_directory, image_path))
        prediction = net.predict([input_image], oversample=False)
        f_output = net.blobs[args.feature_layer].data[0].transpose((1,2,0)).copy()
        data.append(np.expand_dims(f_output, axis=0))
        if i % 99 == 0:
            log("Processed {} images.".format(i + 1))

    log("All images processed.")
    np.save(args.output_file, np.concatenate(data))
    log("Featurs saved. Shape: {}".format(data[0].shape))
