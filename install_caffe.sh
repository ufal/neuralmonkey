#!/bin/bash

if [ ! -d caffe-env ]; then
    virtualenv caffe-env --system-site-packages
fi

source caffe-env/bin/activate

apt-get source protobuf-compiler
cd protobuf-*
./configure --prefix=$PWD/../caffe-env/
make
make install
cd ..
rm -r protobuf-* protobuf_*

apt-get source gflags
cd gflags-*
./configure --prefix=$PWD/../caffe-env
make
make install
cd ..
rm -r gflags-* gflags_*

apt-get source libgoogle-glog-dev
cd google-glog-*
./configure --prefix=$PWD/../caffe-env/
make
make install
cd ..
rm -r google-glog-* google-glog_*

apt-get source liblmdb-dev
cd lmdb*
cd libraries/liblmdb
sed -ie "s#^prefix\t= /usr#prefix\t= $PWD/../../../caffe-env#" Makefile
make
make install
cd ../../..
rm -rf lmdb-* lmdb_*

apt-get source libleveldb-dev
cd leveldb-*
sed -ie "s#PREFIX ?= /usr/local#PREFIX = $PWD/../caffe-env#" Makefile
make
make install
cd ..
rm -r leveldb-* leveldb_*

wget http://www.cmake.org/files/v3.3/cmake-3.3.0.tar.gz
tar zxvf cmake-3.3.0.tar.gz
rm cmake-3.3.0.tar.gz
cd cmake-3.3.0/
./configure --prefix=$PWD/../caffe-env/ --datadir=$PWD/../caffe-env/ --docdir=$PWD/../caffe-env/ --mandir=$PWD/../caffe-env/
make
make install
cd ..
rm -r cmake-3.3.0

git clone https://github.com/Itseez/opencv.git
cd opencv
mkdir -p build/3rdparty/ippicv
wget http://sourceforge.net/projects/opencvlibrary/files/3rdparty/ippicv/ippicv_linux_20141027.tgz/download -O build/3rdparty/ippicv/ippicv_linux_20141027.tgz
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$PWD/../../caffe-env/ -D WITH_IPP=OFF ..
make
#make install
cd ../..
rm -rf opencv

apt-get source lapack
cd lapack-*
mkdir build
cd build
cmake ..
make
cp bin/* ../../caffe-env/bin
cp lib/* ../../caffe-env/lib
cd ../..
rm -r lapack-* lapack_*

apt-get source openblas
cd openblas-*
make
cp libopenblas.a ../caffe-env/lib/
cd ..
rm -r openblas-* openblas_*

git clone https://github.com/BVLC/caffe
sed -e "s/# CPU_ONLY/CPU_ONLY/;s/BLAS := atlas/BLAS := open/;s|# BLAS_INCLUDE := /p.*|BLAS_INCLUDE := $PWD/../caffe-env/include|;s|# BLAS_LIB := /p.*|BLAS_LIB := $PWD/../caffe-env/lib|;s|^PYTHON_INCLUDE := |PYTHON_INCLUDE := $PWD/../caffe-env/include |;s|^PYTHON_LIB := |PYTHON_LIB := $PWD/../caffe-env/lib|;" Makefile.config.example > Makefile.config
sed -ie "s/opencv_imgproc\$/opencv_imgproc opencv_imgcodecs/" Makefile
make all
make pycaffe

pip install scikit-image
pip install cython
pip install pyaml

cd caffe/data/ilsvrc12
bash get_ilsvrc_aux.sh
cd ../../..

echo "" >> caffe-env/bin/activate
echo "export LD_LIBRARY_PATH=$PWD/caffe-env/lib" >> caffe-env/bin/activate
