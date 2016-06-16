#Neural Monkey

[![Build Status](https://travis-ci.org/ufal/neuralmonkey.svg?branch=master)](https://travis-ci.org/ufal/neuralmonkey)

Multimodal machine translation and crosslingual image description generation

- `caffe_image_features.py` extracts features from images using pre-trained network
- `decoder.py` the decoder (aggregates everything a good decoder needs)
- `decoding_function.py` todo
- `image_encoder.py` transforms image features into tensorflow placeholder objects
- `install_caffe.sh` installs caffe including prerequisites
- `learning_utils.py` logging, header printing, training loop
- `precompute_image_features.py` deprecated
- `prepare_env.sh` self-explaining
- `prepare_vgg16.sh` deprecated
- `README.md` this file
- `reformat_downloaded_image_features.py` deprecated
- `tagging` see tagging/README.md
- `test_vocabulary.py` unit test for vocabulary
- `tokenize_data.py` tokenizes data
- `tokenize_persistable.sh` manages the tokenize_data script
- `train_captioning.py`	...
- `vocabulary.py` python definition for vocabulary object
