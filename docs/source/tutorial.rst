.. _post-editing:

==========================
Post-Editing Task Tutorial
==========================

This tutorial will guide you through designing your first experiment in Neural
Monkey.

Before we get started with the tutorial, please check that you have the Neural
Monkey package properly
:ref:`installed and working <instalation>`.


Part I. - The Task
------------------

This section gives an overall description of the task we will try to solve in
this tutorial. To make things more interesting than plain machine translation,
let's try automatic post-editing task (APE, rhyming well with Neural Monkey).

In short, automatic post-editing is a task, in which we have a source language
sentence (let's call it ``f``, as grown-ups do), a machine-translated sentence
of ``f`` (I actually don't know what grown-ups call this, so let's call this
``e'``), and we are expected to generate another sentence in the same language
as ``e'`` but cleaned of all the errors that the machine translation system have
made (let's call this cleaned sentence ``e``). Consider this small example:

Source sentence ``f``:
  Bärbel hat eine Katze.

Machine-translated sentence ``e'``:
  Bärbel has a dog.

Corrected translation ``e``:
  Bärbel has a cat.

In the example, the machine translation system wrongly translated the German
word "Katze" as the English word "dog". It is up to the post-editing system to
fix this error.

In theory (and in practice), we regard the machine translation task as searching
for a target sentence ``e*`` that has the highest probability of being the
translation given the source sentence ``f``. You can put it to a formula::

  e* = argmax_e p(e|f)

In the post-editing task, the formula is slightly different::

  e* = argmax_e p(e|f, e')

If you think about this a little, there are two ways one can look at this
task. One is that we are translating the machine-translated sentence from a kind
of *synthetic* language into a proper one, with additional knowledge what the
source sentence was. The second view regards this as an ordinary machine
translation task, with a little help from another MT system.

In our tutorial, we will assume the MT system used to produce the sentence
``e'`` was good enough. We thus generally trust it and expect only to make
small edits to the
translated sentence in order to make it fully correct. This means that we don't need
to train a whole new MT system that would translate the source sentences from
scratch. Instead, we will build a system that will tell us how to edit the
machine translated sentence ``e'``.


Part II. - The Edit Operations
------------------------------

How can an automatic system tell us how to edit a sentence? Here's one way to do
it: We will design a set of edit operations and train the system to generate a
sequence of these operations. If we consider a sequence of edit operations a
function ``R`` (as in *rewrite*), which transforms one sequence to another, we
can adapt the formulas above to suit our needs more::

  R* = argmax_R p(R(e')|f, e')
  e* = R*(e')

So we are searching for the best edit function ``R*`` that, once applied
to ``e'``, will give us the corrected output ``e*``.
Another question is what the class of all possible edit functions
should look like, for now we simply limit them to functions that can be
defined as sequences of edit operations.

The edit function ``R`` processes the input sequence token-by-token in left-to-right
direction. It has a pointer to the input sequence, which starts by pointing to
the first word of the sequence.

We design three types of edit operations as follows:

1. KEEP - this operation copies the current word to the output and moves the
   pointer to the next token of the input,
2. DELETE - this operation does not emit anything to the output and moves the pointer
   to the next token of the input,
3. INSERT - this operation puts a word on the output, leaving the pointer to the
   input intact.

The edit function applies all its operations to the input sentence. We handle malformed
edit sequences simply: if the pointer reaches the end of the input seqence, operations KEEP
and DELETE do nothing. If the sequence of edits ends before the end of the input
sentence is reached, we apply as many additional KEEP operations as needed to reach
the end of the input sequence.

Let's see another example::

  Bärbel  has   a     dog          .
  KEEP    KEEP  KEEP  DELETE  cat  KEEP

The word "cat" on the second line is an INSERT operation parameterized by the
word "cat". If we apply all the edit operations to the input (i.e. keep the
words "Bärbel", "has", "a", and ".", delete the word "dog" and put the word
"cat" in its place), we get the corrected target sentence.


Part III. - The Data
--------------------

We are going to use the data for WMT 16 shared APE task. You can get them at the
`WMT 16 website <http://www.statmt.org/wmt16/ape-task.html>`_ or directly at the
`Lindat repository <http://hdl.handle.net/11372/LRT-1632>`_. There are three
files in the repository:

1. TrainDev.zip - contains training and development data set
2. Test.zip - contains source and translated test data
3. test_pe.zip - contains the post-edited test data

Now - before we start, let's create our experiment directory, in which we will
place all our work. We shall call it for example ``exp-nm-ape`` (feel free to
choose another weird string).

Extract all the files into the ``exp-nm-ape/data`` directory. Rename the files and
directories so you get this directory structure::

  exp-nm-ape
  |
  \== data
      |
      |== train
      |   |
      |   |== train.src
      |   |== train.mt
      |   \== train.pe
      |
      |== dev
      |   |
      |   |== dev.src
      |   |== dev.mt
      |   \== dev.pe
      |
      \== test
          |
          |== test.src
          |== test.mt
          \== test.pe

The data is already tokenized so we don't need to run any preprocessing
tools. The format of the data is plain text with one sentence per line.  There
are 12k training triplets of sentences, 1k development triplets and 2k of
evaluation triplets.

Preprocessing of the Data
*************************

The next phase is to prepare the post editing sequences that we should learn
during training. We apply the Levenshtein algorithm to find the shortest edit
path from the translated sentence to the post-edited sentence. As a little
coding excercise, you can implement your own script that does the job, or you
may use our preprocessing script from the Neural Monkey package. For this, in the
neuralmonkey root directory, run:

.. code-block:: bash

  scripts/postedit_prepare_data.py \
    --translated-sentences=exp-nm-ape/data/train/train.mt \
    --target-sentences=exp-nm-ape/data/train/train.pe \
        > exp-nm-ape/data/train/train.edits

And the same for the development data.

NOTE: You may have to change the path to the exp-nm-ape directory if it is not
located inside the repository root directory.

NOTE 2: There is a hidden option of the preparation script
(``--target-german=True``) which turns on some steps
tailored for better processing of German text. In this tutorial, we are not
going to use it.

If you look at the preprocessed files, you will see that the KEEP and DELETE
operations are represented with special tokens while the INSERT operations are
represented simply with the word they insert.

Congratulations! Now, you should have train.edits, dev.edits and test.edits
files all in their respective data directories. We can now move to work with
Neural Monkey configurations!


Part IV. - The Model Configuration
----------------------------------

In Neural Monkey, all information about a model and its training is stored in
configuration files. The syntax of these files is a plain INI syntax (more
specifically, the one which gets processed by Python's ConfigParser). The
configuration file is structured into a set of sections, each describing a part
of the training. In this section, we will go through all of them and write our
configuration file needed for the training of the post-editing task.

First of all, create a file called ``post-edit.ini`` and put it inside the
``exp-nm-ape`` directory. Put all the snippets that we will describe in the
following paragraphs into the file.


1 - Datasets
************

For training, we prepare two datasets. The first dataset will serve for the
training, the second one for validation. In Neural Monkey, each dataset contains
a number of so called `data series`. In our case, we will call the data series
`source`, `translated`, and `edits`. Each of those series will contain the
respective set of sentences.

It is assumed that all series within a given dataset have the same number of
elements (i.e. sentences in our case).

The configuration of the datasets looks like this:

.. code-block:: ini

  [train_dataset]
  class=dataset.load_dataset_from_files
  s_source="exp-nm-ape/data/train/train.src"
  s_translated="exp-nm-ape/data/train/train.mt"
  s_edits="exp-nm-ape/data/train/train.edits"

  [val_dataset]
  class=dataset.load_dataset_from_files
  s_source="exp-nm-ape/data/dev/dev.src"
  s_translated="exp-nm-ape/data/dev/dev.mt"
  s_edits="exp-nm-ape/data/dev/dev.edits"
.. TUTCHECK exp-nm-ape/post-edit.ini

Note that series names (`source`, `translated`, and `edits`) are arbitrary and
defined by their first mention. The ``s_`` prefix stands for "series" and
is used only here in the dataset sections, not later when the series are referred to.

These two INI sections represent two calls to function
``neuralmonkey.config.dataset_from_files``, with the series file paths as keyword
arguments. The function serves as a constructor and builds an object for every call.
So at the end, we will have two objects representing the two datasets.


2 - Vocabularies
****************

Each encoder and decoder which deals with language data operates with some kind
of vocabulary. In our case, the vocabulary is just a list of all unique words in
the training data. Note that apart the special ``<keep>`` and ``<delete>``
tokens, the vocabularies for the `translated` and `edits` series are from the
same language. We can save some memory and perhaps improve quality of the target
language embeddings by share vocabularies for these datasets. Therefore, we need
to create only two vocabulary objects:

.. code-block:: ini

  [source_vocabulary]
  class=vocabulary.from_dataset
  datasets=[<train_dataset>]
  series_ids=["source"]
  max_size=50000

  [target_vocabulary]
  class=vocabulary.from_dataset
  datasets=[<train_dataset>]
  series_ids=["edits", "translated"]
  max_size=50000
.. TUTCHECK exp-nm-ape/post-edit.ini

The first vocabulary object (called ``source_vocabulary``) represents the
(English) vocabulary used for this task. The 50,000 is the maximum size of the
vocabulary. If the actual vocabulary of the data was bigger, the rare words
would be replaced by the ``<unk>`` token (hardcoded in Neural Monkey, not part
of the 50,000 items), which stands for unknown words.  In
our case, however, the vocabularies of the datasets are much smaller so we won't
lose any words.

Both vocabularies are created out of the training dataset, as specified by the
line ``datasets=[<train_dataset>]`` (more datasets could be given in the list). This
means that if there are any unseen words in the development or test data, our
model will treat them as unknown words.

We know that the languages in the ``translated`` series and ``edits`` are
the same (except for the KEEP and DELETE tokens in the edits), so we create a
unified vocabulary for them. This is achieved by specifying
``series_ids=[edits, translated]``. The one-hot encodings (or more precisely,
indices to the vocabulary) will be identical for words in ``translated`` and
``edits``.


3 - Encoders
************

Our network will have two inputs. Therefore, we must design two separate
encoders. The first encoder will process source sentences, and the second will
process translated sentences, i.e. the candidate translations that we are
expected to post-edit. This is the configuration of the encoder for
the source sentences:

.. code-block:: ini

  [src_encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  rnn_size=300
  max_input_len=50
  embedding_size=300
  dropout_keep_prob=0.8
  attention_type=decoding_function.Attention
  data_id="source"
  name="src_encoder"
  vocabulary=<source_vocabulary>
.. TUTCHECK exp-nm-ape/post-edit.ini

This configuration initializes a new instance of sentence encoder with the
hidden state size set to 300 and the maximum input length set to 50. (Longer
sentences are trimmed.) The sentence encoder looks up the words in a word
embedding matrix. The size of the embedding vector used for each word from the
source vocabulary is set to 300. The source data series is fed to this
encoder. 20% of the weights is dropped out during training from the word
embeddings and from the attention vectors computed over the hidden states of
this encoder. Note the ``name`` attribute must be set in each encoder and
decoder in order to prevent collisions of the names of Tensorflow graph nodes.

The configuration of the second encoder follows:

.. code-block:: ini

  [trans_encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  rnn_size=300
  max_input_len=50
  embedding_size=300
  dropout_keep_prob=0.8
  attention_type=decoding_function.Attention
  data_id="translated"
  name="trans_encoder"
  vocabulary=<target_vocabulary>
.. TUTCHECK exp-nm-ape/post-edit.ini

This config creates a second encoder for the ``translated`` data series. The
setting is the same as for the first encoder, except for the different
vocabulary and name.


4 - Decoder
***********

Now, we configure perhaps the most important object of the training - the
decoder. Without further ado, here it goes:

.. code-block:: ini

  [decoder]
  class=decoders.decoder.Decoder
  name="decoder"
  encoders=[<trans_encoder>, <src_encoder>]
  rnn_size=300
  max_output_len=50
  embeddings_encoder=<trans_encoder>
  dropout_keep_prob=0.8
  use_attention=True
  data_id="edits"
  vocabulary=<target_vocabulary>
.. TUTCHECK exp-nm-ape/post-edit.ini

As in the case of encoders, the decoder needs its RNN and embedding size
settings, maximum output length, dropout parameter, and vocabulary settings.

The outputs of the individual encoders are by default simply concatenated
and projected to the decoder hidden state (of ``rnn_size``). Internally,
the code is ready to support arbitrary mappings by adding one more parameter
here: ``encoder_projection``.

Note that you may set ``rnn_size`` to ``None``. Neural Monkey will then directly
use the concatenation of encoder states without any mapping. This is particularly
useful when you have just one encoder as in MT.

The line ``embeddings_encoder=<trans_encoder>`` means that the embeddings (including
embedding size) are shared with ``trans_encoder``.


The loss of the decoder is computed
against the ``edits`` data series of whatever dataset the decoder will be
applied to.


5 - Runner and Trainer
**********************

As their names suggest, runners and trainers are used for running and training
models. The ``trainer`` object provides the optimization operation to the graph. In
the case of the cross entropy trainer (used in our tutorial), the default optimizer
is Adam and it is run against the decoder's loss, with added L2
regularization (controlled by the ``l2_weight`` parameter of the
trainer). The runner is used to process a dataset by the model and return the
decoded sentences, and (if possible) decoder losses.

We define these two objects like this:

.. code-block:: ini

  [trainer]
  class=trainers.cross_entropy_trainer.CrossEntropyTrainer
  decoders=[<decoder>]
  l2_weight=1.0e-8

  [runner]
  class=runners.runner.GreedyRunner
  decoder=<decoder>
  output_series="greedy_edits"
.. TUTCHECK exp-nm-ape/post-edit.ini


Note that a runner can only have one decoder, but during training you can train
several decoders, all contributing to the loss function.

The purpose of the trainer is to optimize the model, so we are not interested in
the actual outputs it produces, only the loss compared to the reference outputs
(and the loss is calculated by the given decoder).

The purpose of the runner is to get the actual outputs and for further use, they
are collected to a new series called ``greedy_edits`` (see the line
``output_series=``) of whatever dataset the runner will be applied to.

6 - Evaluation Metrics
**********************

During validation, the whole validation dataset gets processed by the models and
the decoded sentences are evaluated against a reference to provide the user with
the state of the training. For this, we need to specify evaluator objects which
will be used to score the outputted sentences. In our case, we will use BLEU and
TER:

.. code-block:: ini

  [bleu]
  class=evaluators.bleu.BLEUEvaluator
  name="BLEU-4"
.. TUTCHECK exp-nm-ape/post-edit.ini


7 - TensorFlow Manager
******************

In order to handle global variables such as how many CPU cores
TensorFlow should use, you need to specify a "TensorFlow manager":

.. code-block:: ini

  [tf_manager]
  class=tf_manager.TensorFlowManager
  num_threads=4
  num_sessions=1
  minimize_metric=True
  save_n_best=3
.. TUTCHECK exp-nm-ape/post-edit.ini


8 - Main Configuration Section
******************************

Almost there! The last part of the configuration puts all the pieces
together. It is called ``main`` and specifies the rest of the training
parameters:

.. code-block:: ini

  [main]
  name="post editing"
  output="exp-nm-ape/training"
  runners=[<runner>]
  tf_manager=<tf_manager>
  trainer=<trainer>
  train_dataset=<train_dataset>
  val_dataset=<val_dataset>
  evaluation=[("greedy_edits", "edits", <bleu>), ("greedy_edits", "edits", evaluators.ter.TER)]
  batch_size=128
  runners_batch_size=256
  epochs=100
  validation_period=1000
  logging_period=20
.. TUTCHECK exp-nm-ape/post-edit.ini

The ``output`` parameter specifies the directory, in which all the files generated by
the training (used for replicability of the experiment, logging, and saving best
models variables) are stored.  It is also worth noting, that if the output
directory exists, the training is not run, unless the line
``overwrite_output_dir=True`` is also included here.

The ``runners``, ``tf_manager``, ``trainer``, ``train_dataset`` and ``val_dataset`` options are self-explanatory.

The parameter ``evaluation`` takes list of tuples, where each tuple contains:
- the name of output series (as produced by some runner), ``greedy_edits`` here,
- the name of the reference series of the dataset, ``edits`` here,
- the reference to the evaluation algorithm, ``<bleu>`` and ``evaluators.ter.TER`` in the two tuples here.

The ``batch_size`` parameter controls how many sentences will be in one training
mini-batch. When the model does not fit into GPU memory, it might be a good idea to
start reducing this number before anything else. The larger the batch size, however, the
sooner the training should converge to the optimum.

Runners are less memory-demanding, so ``runners_batch_size`` can be set higher than ``batch_size``.

The ``epochs`` parameter specifies
the number of passes through the training data that the training loop should
do. There is no early stopping mechanism in Neural Monkey yet, the training can be resumed after the
end, however. The training can be safely ctrl+C'ed in any time: Neural Monkey preserves the
last ``save_n_best`` best model variables saved on the disk.

The validation and logging periods specify how often to measure the model's
performance on the training batch (``logging_period``) or on validation data
(``validation_period``). Note that both logging and validation involve running the runners
over the current batch or the validation data, resp. If this happens too often,
the time needed to train the model can significantly grow.

At each validation (and logging), the output
is scored using the specified evaluation metrics. The last of the evaluation
metrics (TER in our case) is used to keep track of the model performance over
time. Whenever the score on validation data is better than any of the ``save_n_best``
(3 in our case) previously saved models, the model is saved, discaring
unneccessary lower scoring models.


Part V. - Running an Experiment
-------------------------------

Now that we have prepared the data and the experiment INI file, we can run the
training. If your Neural Monkey installation is OK, you can just run this
command from the root directory of the Neural Monkey repository:

.. code-block:: bash

  bin/neuralmonkey-train exp-nm-ape/post-edit.ini

You should see the training program reporting the parsing of the configuration
file, initializing the model, and eventually the training process. If everything
goes well, the training should run for 100 epochs. You should see a new line
with the status of the model's performance on the current batch every few
seconds, and there should be a validation report printed every few minutes.

As given in the ``main.output`` config line, the Neural Monkey creates the directory
``experiments/training`` with these files:

- ``git_commit`` - the Git hash of the current Neural Monkey revision.
- ``git_diff`` - the diff between the clean checkout and the working copy.
- ``experiment.ini`` - the INI file used for running the training (a simple copy of the file NM was started with).
- ``experiment.log`` - the output log of the training script.
- ``checkpoint`` - file created by Tensorflow, keeps track of saved variables.
- ``events.out.tfevents.<TIME>.<HOST>`` - file created by Tensorflow, keeps the
  summaries for TensorBoard visualisation
- ``variables.data[.<N>]`` - a set of files with N best saved models.
- ``variables.data.best`` - a symbolic link that points to the variable file
  with the best model.


Part VI. - Evaluation of the Trained Model
------------------------------------------

If you have reached this point, you have nearly everything this tutorial
offers. The last step of this tutorial is to take the trained model and to
apply it to a previously unseen dataset. For this you will need two additional
configuration files. But fear not - it's not going to be that difficult. The
first configuration file is the specification of the model. We have this from
Part III and a small optional change is needed. The second
configuration file tells the run script which datasets to process.

The optional change of the model INI file prevents the training dataset from
loading. This is a flaw in the present design and it is planned to change. The
procedure is simple:

1. Copy the file ``post-edit.ini`` into e.g. ``post-edit.test.ini``
2. Open the ``post-edit.test.ini`` file and remove the ``train_dataset`` and
   ``val_dataset`` sections, as well as the ``train_dataset`` and
   ``val_dataset`` configuration from the ``[main]`` section.

Now we have to make another file specifying the testing dataset
configuration. We will call this file ``post-edit_run.ini``:

.. code-block:: ini

  [main]
  test_datasets=[<eval_data>]

  [eval_data]
  class=dataset.load_dataset_from_files
  s_source="exp-nm-ape/data/test/test.src"
  s_translated="exp-nm-ape/data/test/test.mt"
  s_greedy_edits_out="exp-nm-ape/test_output.edits"
.. TUTCHECK exp-nm-ape/post-edit_run.ini


The dataset specifies the two input series ``s_source`` and ``s_translated`` (the
candidate MT output output to be post-edited) as in the training. The series
``s_edits`` (containing reference edits) is **not** present in the evaluation
dataset, because we do not want to use the reference edits to
compute loss at this point. Usually, we don't even *know* the correct output at runtime.

Instead, we introduce the output series ``s_greedy_edits_out`` (the prefix ``s_`` and
the suffix ``_out`` are hardcoded in Neural Monkey and the series name in between
has to match the name of the series produced by the runner).

The line ``s_greedy_edits_out=`` specifies the file where the output should be saved.
(You may want to alter the path to the ``exp-nm-ape`` directory if it is not located inside
the Neural Monkey package root dir.)

We have all that we need to run the trained model on the evaluation
dataset. From the root directory of the Neural Monkey repository, run:

.. code-block:: bash

  bin/neuralmonkey-run exp-nm-ape/post-edit.test.ini exp-nm-ape/post-edit_run.ini

At the end, you should see a new file ``exp-nm-ape/test_output.edits``.
As you notice, the contents of this file are the
sequences of edit operations, which if applied to the machine translated
sentences, generate the output that we want. The final step is to call the
provided post-processing script. Again, feel free to write your own as a simple
exercise:

.. code-block:: bash

  scripts/postedit_reconstruct_data.py \
    --edits=exp-nm-ape/test_output.edits \
    --translated-sentences=exp-nm-ape/data/test/test.mt \
      > test_output.pe

Now, you can run the official tools (like mteval or the tercom software
available on the `WMT 16 website <http://www.statmt.org/wmt16/ape-task.html>`_)
to measure the score of ``test_output.pe`` on the ``data/test/test.pe``
reference evaluation dataset.


Part VII. - Conclusions
-----------------------

This tutorial gave you the basic overview of how to design your experiments using
Neural Monkey. The sample experiment was the task of automatic
post-editing. We got the data from the WMT 16 APE shared task and pre-processed
them to fit our needs. We have written the configuration file and run the
training. At the end, we evaluated the model on the test dataset.

If you want to learn more, the next step is perhaps to browse the ``examples``
directory in Neural Monkey repository and see some further possible setups. If you are
planning to just design an experiment using existing modules, you can start by
editing one of those examples as well.

If you want to dig in the code, you can browse the `repository
<https://github.com/ufal/neuralmonkey>`_ Please feel free to fork the repository
and to send us pull requests. The `API documentation
<http://neural-monkey.readthedocs.io/>`_ is currently under construction, but it
already contains a little information about Neural Monkey objects and their
configuraiton options.

Have fun!
