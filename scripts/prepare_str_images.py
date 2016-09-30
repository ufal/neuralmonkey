#!/usr/bin/env python3

import argparse
import os
import gzip
import pickle as pickle
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.image_utils import STRPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Prepares the STR data.")
    parser.add_argument("--list", type=argparse.FileType('r'),
                        help="File with images.", required=True)
    parser.add_argument("--img-root", type=str, required=True,
                        help="Directory with images.")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--output-log", type=argparse.FileType('w'), required=True)
    args = parser.parse_args()

    preprocessor = STRPreprocessor(args.height, args.max_width)

    f_out = gzip.open(args.output_file, mode='wb')
    processed = 0
    for i, line in enumerate(args.list):
        img_path = os.path.join(args.img_root, line.rstrip())

        try:
            img = preprocessor(img_path)
            pickle.dump(img, f_out)

            args.output_log.write("{}\n".format(img_path))
            processed += 1
            if i % 1000 == 999:
                log("Processed {} images".format(i + 1))
        except Exception as exc:
            log("Skipped {} (no. {}), expeption {}".format(img_path, i, exc), color='red')

    log("Done, saved {} images to {}".format(processed, args.output_file))

    f_out.close()

    log("Padded {} times, on averaged {:.0f} pixels".\
            format(len(preprocessor.paddings),
                   np.mean(preprocessor.paddings) if preprocessor.paddings else 0.0))
    log("Shrinked {} times, on averaged {:.0f} pixels".\
            format(len(preprocessor.shrinkages),
                   np.mean(preprocessor.shrinkages) if preprocessor.shrinkages else 0.0))

if __name__ == "__main__":
    main()
