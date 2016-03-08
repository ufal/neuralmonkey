#!/bin/bash

if [ ! -d tf ]; then
    virtualenv --system-site-packages tf
fi

if [ ! -d tf-gpu ]; then
    virtualenv --system-site-packages tf-gpu
fi
cat tf-gpu/bin/activate - << EOF > tf-gpu/bin/activate-cuda
CUDA_DIR=/usr/local/cuda

if [ -d $CUDA_DIR ] ; then
    echo "Initializing CUDA directories"
    export CUDA_HOME=$CUDA_DIR
    export PATH=$PATH:$CUDA_DIR/bin
    export LD_LIBRARY_PATH=$CUDA_DIR/lib64:/home/helcl/cudnn/lib64
fi
EOF

source tf/bin/activate
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
deactivate

source tf-gpu/bin/activate
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
deactivate

for ENV in tf tf-gpu; do
    source $ENV/bin/activate
    pip install --upgrade pip
    pip install --upgrade ipdb
    pip install --upgrade javabridge
    pip install --upgrade termcolor
    deactivate
done

# Decompounder
cd tf
if [ ! -d jwordsplitter ]; then
    git clone https://github.com/danielnaber/jwordsplitter
    cd jwordsplitter
    ./build.sh
fi
cd ../..
ln -s tf/jwordsplitter tf-gpu/jwordsplitter
