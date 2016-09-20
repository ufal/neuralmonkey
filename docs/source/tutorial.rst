
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
`WMT 16 website <http://www.statmt.org/wmt16/ape-task.html>` or directly at the
`Lindat repository <http://hdl.handle.net/11372/LRT-1632>`. There are three
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
path from the translated sentence to the post-edited sentence. You can implement
your own script that does the job, or you may want to use our preprocessing
script from the neuralmonkey pakage. For this, in the neuralmonkey root
directory, run::

  bin/postedit_prepare_data.py \
    --translated-sentences=exp-nm-ape/data/train/train.mt \
    --target-sentences=exp-nm-ape/data/train.train.pe \
        > exp-nm-ape/data/train/train.edits

TODO check if this still works

NOTE: You may have to change the path to the exp-nm-ape directory if it is not
located inside the repository root directory.

NOTE 2: There is a hidden option of the preparation script
(``--target-german=True``), which if used, it performs some preprocessing steps
tailored for better processing of German text. In this tutorial, we are not
going to use it.

Congratulations! Now, you should have train.edits, dev.edits and test.edits files all in their
respective data directories. We can now move to work with Neural Monkey configurations!

Part IV. - The Model Configuration
----------------------------------

Part V. - Running an Experiment
-------------------------------

Part VI. - Evaluation of the Trained Model
------------------------------------------

Part VII. - Conclusions
-----------------------
