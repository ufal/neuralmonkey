.. _machine-translation:

============================
Machine Translation Tutorial
============================

This tutorial will guide you through designing Machnine Translation
experiments in Neural Monkey. We assumes that you already read
:ref:`the post-editing tutorial <post-editing>`.

The goal of the translation task is to translate sentences from one language
into
another. For this tutorial we use data from the WMT 16 IT-domain
translation shared task on English-to-Czech direction.

`WMT <http://www.statmt.org/wmt16/>`_
is an annual machine translation conference where academic
groups compete in translating different datasets over various language pairs.


Part I. - The Data
--------------------

We are going to use the data for the WMT 16 IT-domain translation shared task.
You can get them at the `WMT IT Translation Shared Task webpage
<http://www.statmt.org/wmt16/it-translation-task.html>`_ and there download
Batch1 and Batch2 answers and Batch3 as a testing set. Or directly `here
<http://ufallab.ms.mff.cuni.cz/~popel/batch1and2.zip>`_ and
`testset <http://ufallab.ms.mff.cuni.cz/~popel/batch3.zip>`_.

Note: In this tutorial we are using only small dataset as an example, which is
not big enough for real-life machine translation training.

We find several files for different languages in the downloaded archive.
From which we use only the following files as our training, validation and
test set::

    1. ``Batch1a_cs.txt and Batch1a_en.txt`` as our Training set
    2. ``Batch2a_cs.txt and Batch2a_en.txt`` as a Validation set
    3. ``Batch3a_en.txt`` as a Test set

Now - before we start, let's make our experiment directory, in which we place
all our work. Let's call it ``exp-nm-mt``.

First extract all the downloaded files, then make gzip files from individual
files and put arrange them into the following directory structure::

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
          \== Batch3a_en.txt.gz

The gzipping is not necessary, if you put the dataset there in plaintext, it
 will work the same way. Neural Monkey recognizes gzipped files by their MIME
type and chooses the correct way to open them.

TODO The dataset is not tokenized and need to be preprocessed.

Byte Pair Encoding
******************

Neural machine translation (NMT) models typically operate with a fixed
vocabulary, but translation is an open-vocabulary problem.
Byte pair encoding (BPE) enables NMT model translation on open-vocabulary by
encoding rare and unknown words as sequences of subword units.
This is based on an intuition that various word classes are translatable via
smaller units than words. More information in the paper
https://arxiv.org/abs/1508.07909 BPE creates a list of merges that are used
for splitting out-of-vocabulary words. Example of such splitting::

  basketball => basket@@ ball

Postprocessing can be manually done by:

.. code-block:: bash

  sed "s/@@ //g"

but Neural Monkey manages it for you.

BPE Generation
**************

In order to use BPE, you must first generate `merge_file`, over all data. This
file is generated on both source and target dataset.
You can generate it by running following script:

.. code-block:: bash

  neuralmonkey/lib/subword_nmt/learn_bpe.py -s 50000 < DATA > merge_file.bpe

With the data from this tutorial it would be the following command:

.. code-block:: bash

  paste Batch1a_en.txt Batch1a_cs.txt \
  | neuralmonkey/lib/subword_nmt/learn_bpe.py -s 8000 \
  > exp-nm-mt/data/merge_file.bpe

You can change number of merges, this number is equivalent to the size of the
vocabulary. Do not forget that as an input is the file containing both source
and target sides.




Part II. - The Model Configuration
----------------------------------

In this section, we create the configuration file
``translation.ini`` needed for the machine translation training.
We mention only the differences from the main post-editing tutorial.

1 - Datasets
************

For training, we prepare two datasets. Since we are using BPE, we need to
 define the preprocessor. The configuration of the datasets looks like this:

.. code-block:: ini

  [train_data]
  class=dataset.load_dataset_from_files
  s_source="exp-nm-mt/data/train/Batch1a_en.txt.gz"
  s_target="exp-nm-mt/data/train/Batch1a_cs.txt.gz"
  preprocessors=[("source", "source_bpe", <bpe_preprocess>), ("target", "target_bpe", <bpe_preprocess>)]

  [val_data]
  class=dataset.load_dataset_from_files
  s_source="exp-nm-mt/data/dev/Batch2a_en.txt.gz"
  s_target="exp-nm-mt/data/dev/Batch2a_cs.txt.gz"
  preprocessors=[("source", "source_bpe", <bpe_preprocess>), ("target", "target_bpe", <bpe_preprocess>)]
.. TUTCHECK exp-nm-mt/translation.ini


2 - Preprocessor and Postprocessor
**********************************

We need to tell the Neural Monkey how it should handle preprocessing and
postprocessing due to the BPE:

.. code-block:: ini

  [bpe_preprocess]
  class=processors.bpe.BPEPreprocessor
  merge_file="exp-nm-mt/data/merge_file.bpe"

  [bpe_postprocess]
  class=processors.bpe.BPEPostprocessor
.. TUTCHECK exp-nm-mt/translation.ini

3 - Vocabularies
****************

For both encoder and decoder we use shared vocabulary created from BPE
merges:

.. code-block:: ini

  [shared_vocabulary]
  class=vocabulary.from_bpe
  path="exp-nm-mt/data/merge_file.bpe"
.. TUTCHECK exp-nm-mt/translation.ini


4 - Encoder and Decoder
***********************

The encoder and decored are similar to those from
:ref:`the post-editing tutorial <post-editing>`:

.. code-block:: ini

  [encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  name="sentence_encoder"
  rnn_size=300
  max_input_len=50
  embedding_size=300
  dropout_keep_prob=0.8
  attention_type=decoding_function.Attention
  data_id="source_bpe"
  vocabulary=<shared_vocabulary>

  [decoder]
  class=decoders.decoder.Decoder
  name="decoder"
  encoders=[<encoder>]
  rnn_size=256
  embedding_size=300
  dropout_keep_prob=0.8
  use_attention=True
  data_id="target_bpe"
  vocabulary=<shared_vocabulary>
  max_output_len=50
.. TUTCHECK exp-nm-mt/translation.ini

You can notice that both encoder and decoder uses as input data id the data
preprocessed by `<bpe_preprocess>`.

5 - Training Sections
*********************

The following sections are described in more detail in
:ref:`the post-editing tutorial <post-editing>`:

.. code-block:: ini

  [trainer]
  class=trainers.cross_entropy_trainer.CrossEntropyTrainer
  decoders=[<decoder>]
  l2_weight=1.0e-8

  [runner]
  class=runners.runner.GreedyRunner
  decoder=<decoder>
  output_series="series_named_greedy"
  postprocess=<bpe_postprocess>

  [bleu]
  class=evaluators.bleu.BLEUEvaluator
  name="BLEU-4"

  [tf_manager]
  class=tf_manager.TensorFlowManager
  num_threads=4
  num_sessions=1
  minimize_metric=False
  save_n_best=3
.. TUTCHECK exp-nm-mt/translation.ini

As for the main configuration section do not forget to add BPE postprocessing:

.. code-block:: ini

  [main]
  name="machine translation"
  output="exp-nm-mt/out-example-translation"
  runners=[<runner>]
  tf_manager=<tf_manager>
  trainer=<trainer>
  train_dataset=<train_data>
  val_dataset=<val_data>
  evaluation=[("series_named_greedy", "target", <bleu>), ("series_named_greedy", "target", evaluators.ter.TER)]
  batch_size=80
  runners_batch_size=256
  epochs=10
  validation_period=5000
  logging_period=80
.. TUTCHECK exp-nm-mt/translation.ini

Part III. - Running and Evaluation of the Experiment
----------------------------------------------------

The training can be run as simply as:

.. code-block:: bash

  bin/neuralmonkey-train exp-nm-mt/translation.ini

As for the evaluation, you need to create ``translation_run.ini``:

.. code-block:: ini

  [main]
  test_datasets=[<eval_data>]

  [bpe_preprocess]
  class=processors.bpe.BPEPreprocessor
  merge_file="exp-nm-mt/data/merge_file.bpe"

  [eval_data]
  class=dataset.load_dataset_from_files
  s_source="exp-nm-mt/data/test/Batch3a_en.txt.gz"
  preprocessors=[("source", "source_bpe", <bpe_preprocess>)]
.. TUTCHECK exp-nm-mt/translation_run.ini

and run:

.. code-block:: bash

 bin/neuralmonkey-run exp-nm-mt/translation.ini exp-nm-mt/translation_run.ini

You are ready to experiment with your own models.
