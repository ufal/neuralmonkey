# Neural Monkey

[![Build Status](https://travis-ci.org/ufal/neuralmonkey.svg?branch=master)](https://travis-ci.org/ufal/neuralmonkey)

__Neural Sequence Learning Using Tensorflow__

The _Neural Monkey_ package is supposed to provide a higher level abstraction
for sequential models, most prominently in Natural Language Processing (NLP)
built over [TensorFlow](http://tensorflow.org/). It should be used for fast
prototyping sequential models in NLP which can be e.g. for neural machine
translation or sentence classification. The higher-level API enables work with
standard building blocks (RNN encoder, RNN decoder, multi-layer percetpron) as
well as simple adding new building blocks implemented directly in TensorFlow.

### Usage

`neuralmonkey-train <EXPERIMENT_INI>`

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

## Related projects

- [tflearn](https://github.com/tflearn/tflearn) -- a more general and less
abstract deep learning toolkit built over TensorFlow

- [nlpnet](https://github.com/erickrf/nlpnet) -- deep learning tools for
tagging and parsing

- [NNBlocks](https://github.com/brmson/NNBlocks) -- a library build over Theano
containing NLP specific models

### License

The software is distributed under the [BSD
License](https://opensource.org/licenses/BSD-3-Clause)

