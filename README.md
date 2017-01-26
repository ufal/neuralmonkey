# Neural Monkey

Neural Sequence Learning Using TensorFlow

[![Build Status](https://travis-ci.org/ufal/neuralmonkey.svg?branch=master)](https://travis-ci.org/ufal/neuralmonkey)
[![Documentation Status](https://readthedocs.org/projects/neural-monkey/badge/?version=latest)](http://neural-monkey.readthedocs.io/en/latest/?badge=latest)

The _Neural Monkey_ package provides a higher level abstraction for sequential
neural network models, most prominently in Natural Language Processing (NLP).
It is built on [TensorFlow](http://tensorflow.org/). It can be used for fast
prototyping of sequential models in NLP which can be used e.g. for neural
machine translation or sentence classification.

The higher-level API brings together a collection of standard building blocks
(RNN encoder and decoder, multi-layer percetpron) and a simple way of adding new
building blocks implemented directly in TensorFlow.

## Usage

```
neuralmonkey-train <EXPERIMENT_INI>
neuralmonkey-run <EXPERIMENT_INI> <DATASETS_INI>
neuralmonkey-server <EXPERIMENT_INI> [OPTION] ...
neuralmonkey-logbook --logdir <EXPERIMENTS_DIR> [OPTION] ...
```

## Installation

- You need Python 3.5 to run _Neural Monkey_.

- For GPU version: Install dependencies by typing `pip install -r
  requirements-gpu.txt`.

- For CPU version, run `pip install -r requirements.txt` instead.

- If you are using the GPU version, make sure that the `LD_LIBRARY_PATH`
  environment variable points to `lib` and `lib64` directories of your CUDA and
  CuDNN installations. Similarly, your `PATH` variable should point to the
  `bin` subdirectory of the CUDA installation directory.

- If the training crashes on an unknown dependency, just install it with pip.

## Getting Started

There is a
[tutorial](http://neural-monkey.readthedocs.io/en/latest/tutorial.html) that
you can follow, which gives you the overwiev of how to design your experiments
with Neural Monkey.

## Package Overview

- `bin`: Directory with neuralmonkey executables

- `examples`: Example configuration files for ready-made experiments

- `lib`: Third party software

- `neuralmonkey`: Python package files

- `scripts`: Directory with tools that may come in handy. Note dependencies for
   these tools may not be listed in the project requirements.

- `tests`: Test files

The `neuralmonkey` package is organized into subpackages as follows:

- `encoders`: The encoders (get sequences of data and encode them)
- `decoders`: The decoders (get outputs of encoders and decode them)
- `nn`: The NN (as in Neural Network components)
- `trainers`: The trainers (train the whole thing)
- `runners`: The runners (run the trained thing)
- `readers`: The readers (read files)
- `processors`: The pre- and postprocessors (pre- and postprocess data)
- `evaluators`: The evaluators (used for measuring performance)
- `config`: The configuration (loading, saving, parsing)
- `logbook`: The Logbook (server for checking up on experiments)
- `tests`: The unit tests

## Additional Scripts

- `caffe_image_features.py` extracts features from images using pre-trained network
- `tokenize_data.py` tokenizes data
- `postedit_prepare_data.py` compute edit operations from two sets of sentences
- `postedit_reconstruct_data.py` applies edit operations to a set of sentences
- `postedit_rule_based_fixes.py` some rule-based fixes for post-editing task

## Documentation

You can find the API documentation of this package
[here](http://neural-monkey.readthedocs.io/en/latest). The documentation files
are generated from docstrings using
[autodoc](http://www.sphinx-doc.org/en/stable/ext/autodoc.html) and
[Napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/) extensions
to the Python documentation package
[Sphinx](http://www.sphinx-doc.org/en/stable/). The docstrings should follow
the recommendations in the [Google Python Style
Guide](http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments).
Additional details on the docstring formatting can be found in the Napoleon
documentation as well.

## Related projects

- [tflearn](https://github.com/tflearn/tflearn) – a more general and less
  abstract deep learning toolkit built over TensorFlow

- [nlpnet](https://github.com/erickrf/nlpnet) – deep learning tools for
  tagging and parsing

- [NNBlocks](https://github.com/brmson/NNBlocks) – a library build over Theano
  containing NLP specific models

- [Nematus](https://github.com/rsennrich/nematus) - A tool for training and
  running Neural Machine Translation models

![Ape is not a
monkey.](http://ufallab.ms.mff.cuni.cz/~helcl/gorila2.png)

## License

The software is distributed under the [BSD
License](https://opensource.org/licenses/BSD-3-Clause).
