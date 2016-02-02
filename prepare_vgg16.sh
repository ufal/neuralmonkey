#!/bin/bash

# This script prepares the VGG-16 such in can be loaded from TensorFlow.

if [ ! -d caffe-env ]; then
    ./install_caffe.sh
fi

source caffe-env/bin/activate

if [ ! -d tensorflow-vgg16 ]; then
    git clone https://github.com/ry/tensorflow-vgg16
fi
cd tensorflow-vgg16
make
mv tensorflow-vgg16/vgg16.tfmodel ..
