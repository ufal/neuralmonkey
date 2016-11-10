
Neural Monkey Tutorial
======================

This tutorial will guide you through designing your first experiment in Neural
Monkey.

Before we get started with the tutorial, please check that you have the Neural
Monkey package properly installed and working (TODO add link to installation
docs here).


Part I. - The Task
------------------

This section gives an overall description of the task we will try to solve in
this tutorial. For I cannot think of anything better now and because I think
making this tutorial about plain-old machine translation task would not be fun,
our experiments will aim at the automatic post-editing task.

In short, automatic post-editing is a task, in which we have a source language
sentence (let's call it ``f``, as grown-ups do), a machine-translated sentence
of ``f`` (I actually don't know what grown-ups call this, so let's call this
``e'``), and we are required to generate another sentence in the same language
as ``e'`` but cleaned of all the errors that the machine translation system have
made (this clean sentence, we will call ``e``). Consider this small example:

Source sentence ``f``:
  Bärbel hat eine Katze.

Machine-translated sentence ``e'``:
  Bärbel has a dog.

Correct translation ``e``:
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
``e'`` was good enough for us to trust it suffice to make small edits to the
translated sentence in order to make them correct. This means that we don't need
to train a whole new MT system that would translate the source sentences from
scratch. Instead we will build a system that will tell us how to edit the
machine translated sentence ``e'``.


Part II. - The Edit Operations
------------------------------

How can an automatic system tell us how to edit a sentence? Here's one way to do
it: We will design a set of edit operations and train the system to generate a
sequence of this operations. If we consider a sequence of edit operations a
function ``R`` (as in *rewrite*), which transforms one sequence to another, we
can adapt the formulas above to suit our needs more::

  R* = argmax_R p(R(e')|f, e')
  e* = R*(e')

Another question that arises is what the class of all possible edit functions
should look like.

The edit function processes the input sequence token-by-token in left-to-right
direction. It has a pointer to the input sequence, which starts by pointing to
the first word of the sequence.

We design three types of edit operations as follows:

1. KEEP - this operation copies the current word to the output and moves the
   pointer to the next token on the input
2. DELETE - this operation does nothing and moves the pointer to the next token
   on the input
3. INSERT - this operation puts a word on the output, leaving the pointer to the
   input intact.

The edit function applies all its operations to the input sentence. For
simplicity, if the pointer reaches the end of the input seqence, operations KEEP
and DELETE do nothing. If the editation is ended before the end of the input
sentence, the edit function applies additional KEEP operations, until it points
to the end of the input sequence.

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

Now - before we start, let's make our experiment directory, in which we will
place all our work. We shall call it for example ``exp-nm-ape`` (feel free to
choose another weird string).

Extract all the files in the ``exp-nm-ape/data`` directory. Rename the files and
direcotries so you get this directory structure::

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
is 12k training triplets of sentences, 1k development triplets and 2k of
evaluation triplets.

Preprocessing of the data
*************************

The next phase is to prepare the post editing sequences that we should learn
during training. We apply the Levenshtein algorithm to find the shortest edit
path from the translated sentence to the post-edited sentence. As a little
coding excercise, you can implement your own script that does the job, or you
may use our preprocessing script from the neuralmonkey package. For this, in the
neuralmonkey root directory, run::

  scripts/postedit_prepare_data.py \
    --translated-sentences=exp-nm-ape/data/train/train.mt \
    --target-sentences=exp-nm-ape/data/train.train.pe \
        > exp-nm-ape/data/train/train.edits

NOTE: You may have to change the path to the exp-nm-ape directory if it is not
located inside the repository root directory.

NOTE 2: There is a hidden option of the preparation script
(``--target-german=True``), which if used, it performs some preprocessing steps
tailored for better processing of German text. In this tutorial, we are not
going to use it.

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
training, the other one for validation. In Neural Monkey, each dataset contains
a number of so called `data series`. In our case, we will call the data series
`source`, `translated`, and `edits`. Each of those series will contain the
respective set of sentences. The configuration of the datasets looks like this::


  [train_dataset]
  class=config.utils.dataset_from_files
  s_source=exp-nm-ape/data/train/train.src
  s_translated=exp-nm-ape/data/train/train.mt
  s_edits=exp-nm-ape/data/train/train.edits

  [val_dataset]
  class=config.utils.dataset_from_files
  s_source=exp-nm-ape/data/dev/dev.src
  s_translated=exp-nm-ape/data/dev/dev.mt
  s_edits=exp-nm-ape/data/dev/dev.edits


These two INI sections represent two calls to function
``neuralmonkey.config.dataset_from_files``, with the series paths as keyword
arguments. At the end, we will have two objects representing the two datasets.


2 - Vocabularies
****************

Each encoder and decoder which deals with language data operates with some kind
of vocabulary. In our case, the vocabulary is just a list of all unique words in
the training data. Note that apart the special ``<keep>`` and ``<delete>``
tokens, the vocabularies for the `translated` and `edits` series are from the
same language. We can save some memory and perhaps improve quality of the target
language embeddings by share vocabularies for these datasets. Therefore, we need
to create only two vocabulary objects::

  [source_vocabulary]
  class=vocabulary.from_dataset
  datasets=[<train_dataset>]
  series_ids=[source]
  max_size=50000

  [target_vocabulary]
  class=vocabulary.from_dataset
  datasets=[<train_dataset>]
  series_ids=[edits, translated]
  max_size=50000

The first vocabulary object (called ``source_vocabulary``) represents the
(English) vocabulary used for this task. The 50,000 is the maximum size of the
vocabulary. If the actual vocabulary of the data was bigger, the rare words
would be replaced by the ``<unk>`` token, which stands for unknown words.  In
our case, however, the vocabularies of the datasets are much smaller so we won't
lose any words. Both vocabularies are created out of the training dataset. This
means that if there are any unseen words in the development or test data, our
model will treat them as unknown words.

The ``target_vocabulary`` is created from both ``edits`` and ``translated``
series from the data. This doesn't mean anything else than the mappings from
words to their one-hot encodings (or more precisely, indices to the vocabulary)
will be identical.


3 - Encoders
************

Our network will have two inputs. Therefore, we must design two separate
encoders. First encoder will process the source sentences, and the second will
process the translated sentences. This is the configuration of the encoder for
the source sentences::

  [src_encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  rnn_size=300
  max_input_len=50
  embedding_size=300
  dropout_keep_p=0.8
  attention_type=decoding_function.Attention
  data_id=source
  name=src_encoder
  vocabulary=<source_vocabulary>

This configuration initializes a new instance of sentence encoder with the
hidden state size set to 300 and the maximum input length set to 50. (Longer
sentences are trimmed.) The sentence encoder looks up the words in a word
embedding matrix. The size of the embedding vector used for each word from the
source vocabulary is set to 300. The source data series is fed to this
encoder. 20% of the weights is dropped out during training from the word
embeddings and from the attention vectors computed over the hidden states of
this encoder. Note the ``name`` attribute must be set in each encoder and
decoder in order to prevent collisions of the names of Tensorflow graph nodes.

The configuration of the second encoder follows::

  [trans_encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  rnn_size=300
  max_input_len=50
  embedding_size=300
  dropout_keep_p=0.8
  attention_type=decoding_function.Attention
  data_id=translated
  name=trans_encoder
  vocabulary=<target_vocabulary>

This config creates a second encoder for the ``translated`` data series. The
setting is the same as in the first encoder case. (Except for the different
vocabulary).


4 - Decoder
***********

Now, we configure perhaps the most important object of the training - the
decoder. Without furhter ado, here it goes::

  [decoder]
  class=decoders.decoder.Decoder
  name=decoder
  encoders=[<trans_encoder>, <src_encoder>]
  rnn_size=300
  max_output_len=50
  reuse_word_embeddings=True
  dropout_keep_p=0.8
  use_attention=True
  data_id=edits
  vocabulary=<target_vocabulary>

As in the case of encoders, the decoder needs its RNN and embedding size
settings, maximum output length, dropout parameter, and vocabulary settings.  In
this case, the embedding size parameter is inferred by the embedding size of the
first encoder (``trans_encoder``), and the embeddings themselves are shared
between that encoder and the decoder. The loss of the decoder is computed
against the ``edits`` data series.


5 - Runner and trainer
**********************

As their names suggest, runners and trainers are used for running and training
models. The trainer object provides the optimization operation to the graph. In
case of the cross entropy trainer (used in our tutorial as well), the optimizer
used is Adam and it's run against the decoder's loss, with added L2
regularization (controlled by the ``l2_regularization`` parameter of the
trainer). The runner is used to process a dataset by the model and return the
decoded sentences, and (if possible) decoder losses.

We define these two objects like this::

  [trainer]
  class=trainers.cross_entropy_trainer.CrossEntropyTrainer
  decoder=<decoder>
  l2_regularization=1.0e-8

  [runner]
  class=runners.runner.GreedyRunner
  decoder=<decoder>
  batch_size=256


6 - Evaluation metrics
**********************

During validation, the whole validation dataset gets processed by the models and
the decoded sentences are evaluated against reference to provide the user with
the state of the training. For this, we need to specify evaluator objects which
will be used to score the outputted sentences. In our case, we will use BLEU and
TER::

  [bleu]
  class=evaluators.bleu.BLEUEvaluator
  name=BLEU-4

  [ter]
  class=evaluators.edit_distance.EditDistance
  name=TER

TODO check if the TER evaluator works as expected


7 - Main configuration section
******************************

Almost there! The last part of the configuration puts all the pieces
together. It is called ``main`` and specifies the rest of the training
parameters::

  [main]
  name=post editing
  output=exp-nm-ape/training
  encoders=[<trans_encoder>, <src_encoder>]
  decoder=<decoder>
  runner=<runner>
  trainer=<trainer>
  train_dataset=<train_dataset>
  val_dataset=<val_dataset>
  evaluation=[<bleu>, <ter>]
  minimize=True
  batch_size=128
  epochs=100
  validation_period=1000
  logging_period=20
  save_n_best=3

The output parameter specify the directory, in which all the files generated by
the training (used for replicability of the experiment, logging, and saving best
models variables) are stored.  It is also worth noting, that if the output
directory exists, the training is not run, unless the ``overwrite_output_dir``
flag is set to ``True``.

The ``encoders`` and ``decoder`` parameters specify the model, the ``runner``,
``trainer``, ``train_dataset`` and ``val_dataset`` options are self-explanatory
as well.

The ``batch_size`` parameter controls how many sentences will be in one training
mini-batch. When model does not fit into GPU memory, it might be a good idea to
start reducing this number before anything else. The larger it is, however, the
sooner the training should converge to the optimum. The ``epochs`` parameter is
the number of passes through the training data that the training loop should
do. There is no early stopping mechanism, the training can be resumed after the
end, however. The training can be safely ctrl+c'ed in any time (preserving the
last ``save_n_best`` best model variables saved (judged by the score on
validation dataset) on the disk).

The validation and logging periods specify how often to measure the model's
performance on training batch or on validation data. If too often, these can
increase the time to train the model. Each validation (and logging), the model
is scored using the specified evaluation metrics. The last of the evaluation
metrics (TER in our case) is used to keep track of the model performance over
time. Whenever the score on validation is better than any of the ``save_n_best``
(3 in our case) previously saved models, the model is saved. The worse scoring
model files are discarded.


Part V. - Running an Experiment
-------------------------------

Now that we have prepred the data and the experiment INI file, we can run the
training. If your Neural Monkey installation is OK, you can just run this
command from the root directory of the Neural Monkey repository::

  bin/neuralmonkey-train exp-nm-ape/post-edit.ini

Again, you may want to adapt the path to the experiment directory.

You should see the training program logging the parsing of the configuration
file, initializing the model, and eventually the training process. If everything
goes well, the training should run for 100 epochs. You should see a new line
with the status of the model's performance on the current batch every few
seconds, and there should be validation report printed every few minutes.

The training script creates a subdirectory called ``training`` in our experiment
directory. The contents of the directory are:

- ``git_commit`` - the Git hash of the current Neural Monkey revision.
- ``git_diff`` - the diff between the clean checkout and the working copy.
- ``experiment.ini`` - the INI file used for running the training (copied).
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
evaluate it on previously unseen dataset. For this you will need additional two
configuration files. But fear not - it's not going to be that difficult. The
first configuration file is the specification of the model. We have this from
the Part III. It will only require a small alteration (optional). The second
configuration file tells the run script which datasets to process.

The optional alteration of the model INI file prevents the training dataset from
loading. This is a flaw in the present design and it's subject to change. The
procedure is simple:

1. Copy the file ``post-edit.ini`` into e.g. ``post-edit.test.ini``
2. Open the ``post-edit.test.ini`` file and remove the ``train_dataset`` and
   ``val_dataset`` sections, as well as the ``train_dataset`` and
   ``val_dataset`` configuration from the ``[main]`` section.

Now we have to make another file specifying the testing dataset
configuration. We will call this file ``test_datasets.ini``::

  [main]
  test_datasets=[<eval_data>]

  [eval_data]
  class=config.utils.dataset_from_files
  s_source=exp-nm-ape/data/test/test.src
  s_translated=exp-nm-ape/data/test/test.mt
  s_edits_out=exp-nm-ape/test_output.edits

Please note the ``s_edits`` data series is **not** present in the evaluation
dataset. That is simply because we do not want to use the reference edits to
compute loss at this point. Usually, we don't even *know* the correct output.
Instead, we will provide the output series ``s_edits_out``, which points to a
file to which the output of the model gets stored. Also note that you may want
to alter the path to the ``exp-nm-ape`` directory if it is not located inside
the Neural Monkey package root dir.

We have all that we need to run the trained model on the evaluation
dataset. From the root directory of the Neural Monkey repository, run::

 bin/neuralmonkey-run exp-nm-ape/post-edit.test.ini exp-nm-ape/test_datasets.ini

At the end, you should see a new file in ``exp-nm-ape``, called
``test_output.edits``. As you notice, the contents of this file are the
sequences of edit operations, which if applied to the machine translated
sentences, generate the output that we want. So the final step is to call the
provided postprocessing script. Again, feel free to write your own as a simple
excercise::

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

This tutorial gave you the basic notion of how to design your experiments using
Neural Monkey. We designed the experiment on the task of automatic
post-editing. We got the data from the WMT 16 APE shared task and preprocessed
them to fit our needs. We have written the configuration file and run the
training. At the end, we evaluated the model on the test dataset.

If you want to learn more, the next step is perhaps to browse the ``examples``
directory in the repository and try to see what's going on there. If you are
planning to just design an experiment using existing modules, you can start by
editing one of those examples as well.

If you want to dig in the code, you can browse the `repository
<https://github.com/ufal/neuralmonkey>`_ Please feel free to fork the repository
and to send us pull requests. The `API
manual <http://neural-monkey.readthedocs.io/>`_ is currently under construction,
but it should contain something very soon.

Have fun!
