Byte Pair Encoding
==================

Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. 
Byte pair encoding (BPE) enables NMT model translation on open-vocabulary by encoding rare and unknown words as sequences of subword units. 
This is based on the intuition that various word classes are translatable via smaller units than words. More information in the paper https://arxiv.org/abs/1508.07909
BPE creates a list of merges that are used for splitting out-of-vocabulary words. Example of such splitting:

  basketball => basket@@ ball

Postprocessing can be manually done by:

  sed "s/@@ //g"

BPE generation
--------------

In order to use BPE, you must first generate merge_file, over all data. This file is generated on both source and target dataset.
You can generate it by running following script:

  neuralmonkey/lib/subword_nmt/learn_bpe.py -s 50000 < train.txt > merge_file.bpe

You can change number of merges, this number is equivalent to the size of the vocabulary. Do not forget that as an input is file containing both source and target sides.

Use of BPE
----------

Now that you have merge_file you can implement the BPE into your model. First you have to create preprocessor and postprocessor.

  [bpe_preprocess]
  class=processors.bpe.BPEPreprocessor
  merge_file=merge_file.bpe

  [bpe_postprocess]
  class=processors.bpe.BPEPostprocessor

Second you need to redefine the vocabulary sections in the following way:

  [source_vocabulary]
  class=config.utils.vocabulary_from_bpe
  path=merge_file.bpe

  [target_vocabulary]
  class=config.utils.vocabulary_from_bpe
  path=merge_file.bpe

To each of the datasets, you need to add a preprocessor:

  [dataset]
  preprocessor=<bpe_preprocess>
  ...

You must add the postprocessing into the [main] section:

  [main]
  postprocess=<bpe_postprocess>
  ...


