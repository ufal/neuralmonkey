
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
translation given the source sentence ``f``. You can put it to a formula:

``e* = argmax_e p(e|f)``

In the post-editing task, the formula is slightly different:

``e* = argmax_e p(e|f, e')``

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
