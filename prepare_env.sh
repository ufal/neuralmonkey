#!/bin/bash

if [ ! -d tf ]; then
    virtualenv --system-site-packages tf
fi

source tf/bin/activate

pip install --upgrade pip
pip install ipdb
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

cd tf
git clone https://github.com/danielnaber/jwordsplitter
cd jwordsplitter
./build.sh
cd ../..

pip install --upgrade javabridge
