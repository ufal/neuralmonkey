Features
========

Byte Pair Encoding
------------------

Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. 
Byte pair encoding (BPE) enables NMT model translation on open-vocabulary by encoding rare and unknown words as sequences of subword units. 
This is based on the intuition that various word classes are translatable via smaller units than words. More information in the paper https://arxiv.org/abs/1508.07909
BPE creates a list of merges that are used for splitting out-of-vocabulary words. Example of such splitting::

  basketball => basket@@ ball

Postprocessing can be manually done by::

  sed "s/@@ //g"

BPE generation
**************

In order to use BPE, you must first generate merge_file, over all data. This file is generated on both source and target dataset.
You can generate it by running following script::

  neuralmonkey/lib/subword_nmt/learn_bpe.py -s 50000 < train.txt > merge_file.bpe

You can change number of merges, this number is equivalent to the size of the vocabulary. Do not forget that as an input is file containing both source and target sides.

Use of BPE
**********

Now that you have merge_file you can implement the BPE into your model. First you have to create preprocessor and postprocessor::

  [bpe_preprocess]
  class=processors.bpe.BPEPreprocessor
  merge_file=merge_file.bpe

  [bpe_postprocess]
  class=processors.bpe.BPEPostprocessor

Second you need to redefine the vocabulary sections. The vocabulary is shared for the BPE and therefore you only need to define one vocabulary for both encoder and decoder as in the following way::

  [shared_vocabulary]
  class=vocabulary.from_bpe
  path=merge_file.bpe

To each of the datasets, you need to add a preprocessor::

  [dataset]
  preprocessor=<bpe_preprocess>
  ...

You must add the postprocessing into the [main] section::

  [main]
  postprocess=<bpe_postprocess>
  ...


Dropout
-------

Neural networks with a large number of parameters have a serious problem with an overfitting. 
Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural
network during training. This prevents units from co-adapting too much. But during the test time, the dropout is turned off. More information in https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

If you want to enable dropout on an encoder or on the decoder, you can simply add dropout_keep_prob to the particular section::
  
  [encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  dropout_keep_prob=0.8
  ...

or::
 
  [decoder]
  class=decoders.decoder.Decoder
  dropout_keep_prob=0.8
  ...

Pervasive dropout
*****************

Detailed information in https://arxiv.org/abs/1512.05287

If you want allow dropout on the recurrent layer of your encoder, you can add use_pervasive_dropout parameter into it and then the dropout probability will be used::

  [encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  dropout_keep_prob=0.8
  use_pervasive_dropout=True
  ...

