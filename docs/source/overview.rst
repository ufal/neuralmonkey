.. _overview:

================
Package Overview
================

This overview should provide you with the basic insight on how Neural Monkey
conceptualizes the problem of sequence-to-sequence learning and how the data
flow during training and running models looks like.

-------------------------------
Loading and Processing Datasets
-------------------------------

We call a *dataset* a collection of named data *series*. By a series we mean a
list of data items of the same type representing one type of input or desired
output of a model. In the simple case of machine translation, there are two
series: a list of source-language sentences and a list of target-language
sentences.

The following scheme captures how a dataset is created from input
data.

.. image:: img/dataset_creation.svg

The dataset is created in the following steps:

1. An input file is read using a *reader*. Reader can e.g., load a file
   containing paths to JPEG images and load them as ``numpy`` arrays, or
   read a tokenized text as a list of lists (sentences) of string tokens.

2. Series created by the readers can be preprocessed by some *series-level
   preprocessors*. An example of such preprocessing is byte-pair encoding which
   loads a list of merges and segments the text accordingly.

3. The final step before creating a dataset is applying *dataset-level*
   preprocessors which can take more series and output a new series.

Currently there are two implementations of a dataset. An in-memory dataset
which stores all data in the memory and a lazy dataset which gradually reads
the input files step by step and only stores the batches necessary for the
computation in the memory.

----------------------------
Training and Running a Model
----------------------------

This section describes the training and running workflow. The main concepts and
their interconnection can be seen in the following scheme.

.. image:: img/model_workflow.svg

The dataset series can be used to create a *vocabulary*. A vocabulary
represents an indexed set of tokens and provides functionality for converting
lists of tokenized sentences into matrices of token indices and vice
versa. Vocabularies are used by encoders and decoders for feeding the provided
series into the neural network.

The model itself is defined by *encoders* and *decoders*. Most of the
TensorFlow code is in the encoders and decoders. Encoders are parts of the
model which take some input and compute a representation of it. Decoders are
model parts that produce some outputs. Our definition of encoders and decoders
is more general than in the classical sequence-to-sequence learning. An encoder
can be for example a convolutional network processing an image. The RNN decoder
is for
us only a special type of decoder, it can be also a sequence labeler or a
simple multilayer-perceptron classifier.

Decoders are executed using so-called *runners*. Different runners
represent different ways of running the model. We might want to get a single
best estimation, get an ``n``-best list or a sample from the model. We might
want to use an RNN decoder to get the decoded sequences or we might be
interested in the word alignment obtained by its attention model. This is all
done by employing different runners over the decoders. The outputs of the
runners can be subject of further *post-processing*.

Additionally to runners, each training experiment has to have its *trainer*.  A
*trainer* is a special case of a runner that actually modifies the parameters of
the model. It collects the objective functions and uses them in an optimizer.

Neural Monkey manages TensorFlow sessions using an object called *TensorFlow
manager*. Its basic capability is to execute runners on provided datasets.
