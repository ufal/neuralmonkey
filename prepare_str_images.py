#!/usr/bin/env python

import argparse
import os
import numpy as np
from scipy.misc import imread, imresize

from utils import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the STR data.")
    parser.add_argument("--list", type=argparse.FileType('r'),
                        help="File with images.", required=True)
    parser.add_argument("--img-root", type=str, required=True,
                        help="Directory with images.")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--output-file", type=argparse.FileType('wb'), required=True)
    parser.add_argument("--output-log", type=argparse.FileType('w'), required=True)
    args = parser.parse_args()

    images = []
    paddings = []
    shrinkages = []
    for i, line in enumerate(args.list):
        img_path = os.path.join(args.img_root, line.rstrip())

        try:
            img = imread(img_path) / 255.0

            if img.shape[0] != args.height:
                ratio = float(args.height) / img.shape[0]
                width = int(ratio * img.shape[1])
                img = imresize(img, (args.height, width))

            if img.shape[1] >= args.max_width:
                images.append(img[:, :args.max_width])
                shrinkages.append(float(img.shape[1] - args.max_width))
            else:
                rest = args.max_width - img.shape[1]
                padding = np.zeros((args.height, rest, 3))
                img = np.concatenate((img, padding), axis=1)
                images.append(img)
                paddings.append(float(rest))

            args.output_log.write("{}\n".format(img_path))

            if i % 1000 == 999:
                log("Processed {} images".format(i + 1))
        except Exception as exc:
            log("Skipped {} (no. {}), expeption {}".format(img_path, i, exc), color='red')

    log("Done, saving {} images to {}".format(len(images), args.output_file))

    np.save(args.output_file, np.asarray(images))

    log("Padded {} times, on averaged {:.0f} pixels".\
            format(len(paddings), np.mean(paddings) if paddings else 0.0))
    log("Shrinked {} times, on averaged {:.0f} pixels".\
            format(len(shrinkages), np.mean(shrinkages) if shrinkages else 0.0))

