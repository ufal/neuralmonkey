#!/bin/bash

if [ ! -d tf ]; then
    virtualenv -p python3 --system-site-packages tf
fi

if [ ! -d tf-gpu ]; then
    virtualenv -p python3 --system-site-packages tf-gpu
fi
cat tf-gpu/bin/activate - << EOF > tf-gpu/bin/activate-cuda
CUDA_DIR=/usr/local/cuda

if [ -d \$CUDA_DIR ] ; then
    echo "Initializing CUDA directories"
    export CUDA_HOME=\$CUDA_DIR
    export PATH=\$PATH:\$CUDA_DIR/bin
    export LD_LIBRARY_PATH=\$CUDA_DIR/lib64:/home/helcl/cudnn/lib64
fi
EOF

TF_VERSION=tensorflow-0.8.0-cp27-none-linux_x86_64.whl

source tf/bin/activate
wget https://storage.googleapis.com/tensorflow/linux/cpu/$TF_VERSION
pip install $TF_VERSION
deactivate

rm $TF_VERSION

source tf-gpu/bin/activate
wget https://storage.googleapis.com/tensorflow/linux/gpu/$TF_VERSION
pip install $TF_VERSION
deactivate

rm $TF_VERSION

for ENV in tf tf-gpu; do
    source $ENV/bin/activate

    # requirement to run the model
    pip install -r requirements.txt --upgrade

    # packages used during development
    pip install --upgrade ipdb
    pip install --upgrade pylint

    deactivate
done
