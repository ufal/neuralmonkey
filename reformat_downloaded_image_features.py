#!/usr/bin/env python
# coding: utf-8

import h5py, argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab-file", type=str, required=True)
    parser.add_argument("--index-start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-file", type=argparse.FileType('wb'), required=True)
    args = parser.parse_args()

    f = h5py.File(args.matlab_file, 'r')
    dataset = f['feats']
    raw_features = dataset[args.index_start:args.index_start+args.count]
    concate_features = \
            np.concatenate([np.expand_dims(f, axis=0) for f in raw_features])
    features = concate_features.reshape([-1, 14, 14, 512])
    np.save(args.output_file, features)
