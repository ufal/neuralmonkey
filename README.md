#Neural Monkey

[![Build Status](https://travis-ci.org/ufal/neuralmonkey.svg?branch=master)](https://travis-ci.org/ufal/neuralmonkey)

Neural Sequence Learning Using Tensorflow


### Usage

`python train.py <EXPERIMENT_INI>`

### Package Overview

The toolkit is organized into packages as follows:

- `encoders`: The encoders (get sequences of data and encode them)
- `decoders`: The decoders (get outputs of encoders and decode them)
- `trainers`: The trainers (train the whole thing)
- `runners`: The runners (run the trained thing)
- `readers`: The readers (read files)
- `processors`: The pre- and postprocessors (pre- and postprocess data)
- `config`: The configuration (loading, saving, parsing)

### Other files

- `caffe_image_features.py` extracts features from images using pre-trained network
- `decoding_function.py` todo
- `install_caffe.sh` installs caffe including prerequisites
- `learning_utils.py` logging, header printing, training loop
- `precompute_image_features.py` deprecated
- `prepare_env.sh` self-explaining
- `prepare_vgg16.sh` deprecated
- `README.md` this file
- `reformat_downloaded_image_features.py` deprecated
- `test_vocabulary.py` unit test for vocabulary
- `tokenize_data.py` tokenizes data
- `tokenize_persistable.sh` manages the tokenize_data script
- `vocabulary.py` python definition for vocabulary object

### Installation

- Install Tensorflow by following their installation docs
  [here](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup)

- Clone this repository and follow the installation procedure outlined in
`prepare_env.sh`. If the training crashes on an unknown dependency, just install
it with pip.