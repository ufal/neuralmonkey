#!/bin/bash

echo "Do not try to run or source this file! :-)"
exit 1

set -ex

if [ ! -d tfpy3 ]; then
    virtualenv -p python3.4 --system-site-packages tfpy3
fi

if [ ! -d tfpy3-gpu ]; then
    virtualenv -p python3.4 --system-site-packages tfpy3-gpu
fi
cat tfpy3-gpu/bin/activate - << EOF > tfpy3-gpu/bin/activate-cuda
CUDA_DIR=/usr/local/cuda

if [ -d \$CUDA_DIR ] ; then
    echo "Initializing CUDA directories"
    export CUDA_HOME=\$CUDA_DIR
    export PATH=\$PATH:\$CUDA_DIR/bin
    export LD_LIBRARY_PATH=\$CUDA_DIR/lib64:/home/helcl/cudnn/lib64
fi
EOF

TF_VERSION=tensorflow-0.9.0-cp34-cp34m-linux_x86_64.whl

source tfpy3/bin/activate
pip install https://storage.googleapis.com/tensorflow/linux/cpu/$TF_VERSION
deactivate

rm $TF_VERSION || echo "Failed to remove"

source tfpy3-gpu/bin/activate
pip install https://storage.googleapis.com/tensorflow/linux/gpu/$TF_VERSION
deactivate

rm $TF_VERSION || echo "Failed to remove"

for ENV in tfpy3 tfpy3-gpu; do
    source $ENV/bin/activate

    # requirement to run the model
    pip install -r requirements.txt --upgrade

    # packages used during development
    pip install --upgrade ipdb
    pip install --upgrade pylint

    deactivate
done
