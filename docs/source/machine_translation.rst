.. _machine-translation:

=================================
Machine Translation Task Tutorial
=================================

This tutorial will guide you through designing Machnie Translation Task
experiments in Neural Monkey. It assumes that you already read the
Post-Editing tutorial.

The goal of this task is to translate sentences from one language into
another. For this tutorial we will use data from IT translation task of WMT
16 on English to Czech direction.


Part I. - The Data
--------------------

We are going to use the data for WMT 16 shared IT Machine Translation task. You
can get them at the `WMT 16 IT Translation Task
<http://www.statmt.org/wmt16/it-translation-task.html>`_ and there download
Batch1 and Batch2 answers and Batch3 as a testing set. Or directly `here
<http://ufallab.ms.mff.cuni.cz/~popel/batch1and2.zip>`_ and
`testset <http://ufallab.ms.mff.cuni.cz/~popel/batch3.zip>`_.

Note: Since we are using only small data for example which are not enough
 for real-life machine translation training.

When we open the archive we find several files for different languages, we
will use only following files as our training, validation and test set::

    1. ``Batch1a_cs.txt and Batch1a_en.txt`` as our Training set
    2. ``Batch2a_cs.txt and Batch2a_en.txt`` as a Validation set
    3. ``Batch3`` as a Test set

Now - before we start, let's make our experiment directory, in which we will
place all our work. We will call it ``exp-nm-mt``.

First extract all the downloaded files, then make gzip files from individual
files and put them into following directory structure::

  exp-nm-mt
  |
  \== data
      |
      |== train
      |   |
      |   |== Batch1a_en.txt.gz
      |   \== Batch1a_cs.txt.gz
      |
      |== dev
      |   |
      |   |== Batch2a_en.txt.gz
      |   \== Batch2a_cs.txt.gz
      |
      \== test
          |
          \== batch3.gz

The gzipping is not a necessary, if you put the dataset there in plaintext it
 will work the same way. Neural Monkey always try to recognize gzip by the
extension and opens it.

TODO The dataset is not tokenized and need to be preprocessed.

Byte Pair Encoding
******************

Neural machine translation (NMT) models typically operate with a fixed
vocabulary, but translation is an open-vocabulary problem.
Byte pair encoding (BPE) enables NMT model translation on open-vocabulary by
encoding rare and unknown words as sequences of subword units.
This is based on the intuition that various word classes are translatable via
smaller units than words. More information in the paper
https://arxiv.org/abs/1508.07909 BPE creates a list of merges that are used
for splitting out-of-vocabulary words. Example of such splitting::

  basketball => basket@@ ball

Postprocessing can be manually done by::

  sed "s/@@ //g"

BPE generation
**************

In order to use BPE, you must first generate merge_file, over all data. This
file is generated on both source and target dataset.
You can generate it by running following script::

  neuralmonkey/lib/subword_nmt/learn_bpe.py -s 50000 < DATA > merge_file.bpe

With the data from this tutorial it would be the following command::

  paste Batch1a_en.txt Batch1a_cs.txt |
  neuralmonkey/lib/subword_nmt/learn_bpe.py -s 8000 >
  exp-nm-mt/data/merge_file.bpe

You can change number of merges, this number is equivalent to the size of the
vocabulary. Do not forget that as an input is the file containing both source
and target sides.




Part II. - The Model Configuration
----------------------------------

In this section, we will go through setting the configuration file
``translation.ini`` needed for
the training of the machine translation task. We will mention only
differences from the main Post_editing tutorial

1 - Datasets
************

For training, we prepare two datasets. Since we will be using BPE, we need to
 define the preprocessor. The configuration of the datasets looks like this::

  [train_data]
  class=config.utils.dataset_from_files
  s_source=exp-nm-mt/data/train/Batch1a_en.txt.gz
  s_target=exp-nm-mt/data/train/Batch1a_cs.txt.gz
  preprocessor=<bpe_preprocess>

  [val_data]
  class=config.utils.dataset_from_files
  s_source=exp-nm-mt/data/dev/Batch2a_en.txt.gz
  s_target=exp-nm-mt/data/dev/Batch2a_cs.txt.gz
  preprocessor=<bpe_preprocess>

2 - Preprocessor and postprocessor
**********************************

Wee need to tell the Neural Monkey how it should handle preprocessing and
postprocessing due to the BPR::

  [bpe_preprocess]
  class=processors.bpe.BPEPreprocessor
  merge_file=exp-nm-mt/data/merge_file.bpe

  [bpe_postprocess]
  class=processors.bpe.BPEPostprocessor


3 - Vocabularies
****************

For both encoder and decoder we will use shared vocabulary created from BPE
merges::

  [shared_vocabulary]
  class=vocabulary.from_bpe
  path=exp-nm-mt/data/merge_file.bpe

4 - Encoder and Decoder
************

The encoder and decored is similar to Post-Editing ones::

  [encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  name=sentence_encoder
  rnn_size=300
  max_input_len=50
  embedding_size=300
  dropout_keep_prob=0.8
  attention_type=decoding_function.Attention
  data_id=source
  vocabulary=<shared_vocabulary>

  [decoder]
  class=decoders.decoder.Decoder
  name=decoder
  encoders=[<encoder>]
  rnn_size=256
  embedding_size=300
  use_attention=True
  dropout_keep_prob=0.8
  data_id=target
  vocabulary=<shared_vocabulary>
  max_output_len=50


5 - Training sections
**********************

Following sections are described more in detail in Post-editing task::

  [trainer]
  class=trainers.cross_entropy_trainer.CrossEntropyTrainer
  decoders=[<decoder>]
  l2_weight=1.0e-8

  [runner]
  class=runners.runner.GreedyRunner
  decoder=<decoder>
  output_series=series_named_greedy

  [bleu]
  class=evaluators.bleu.BLEUEvaluator
  name=BLEU-4

  [ter]
  class=evaluators.edit_distance.EditDistance
  name=TER

  [tf_manager]
  class=tf_manager.TensorFlowManager
  num_threads=4
  num_sessions=1


As for the main configuration section do not forget to add BPE postprocessing::

  [main]
  name=machine translation
  output=exp-nm-mt/out-example-translation
  runners=[<runner>]
  tf_manager=<tf_manager>
  trainer=<trainer>
  train_dataset=<train_data>
  val_dataset=<val_data>
  evaluation=[(series_named_greedy,target,<bleu>), (series_named_greedy,target,
  <ter>)]
  minimize=True
  batch_size=128
  runners_batch_size=256
  epochs=10
  validation_period=1000
  logging_period=20
  save_n_best=3
  postprocess=<bpe_postprocess>

Part III. - Running and Evaluation of the Experiment
----------------------------------------------------

The training can be run as simply as::

  bin/neuralmonkey-train exp-nm-mt/translation.ini

As for the evaluation, you need to create ``test_datasets.ini``::

  [main]
  test_datasets=[<eval_data>]

  [eval_data]
  class=config.utils.dataset_from_files
  s_source=exp-nm-mt/data/test/batch3.gz
  preprocessor=<bpe_preprocess>

and run::

 bin/neuralmonkey-run exp-nm-mt/translation.ini exp-nm-mt/test_datasets.ini

Now, you have a translation produced from your own translation model and you
can start training various models.