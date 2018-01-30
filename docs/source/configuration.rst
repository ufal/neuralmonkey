Configuration
=============

Experiments with NeuralMonkey are configured using configuration files
which specifies the architecture of the model, meta-parameters of the
learning, the data, the way the data are processed and the way the model
is run.

Syntax
------

The configuration files are based on the syntax of INI files, see
e.g., the corresponding `Wikipedia
page <https://en.wikipedia.org/wiki/INI_file>`__..

Neural Monkey INI files contain
*key-value pairs*, delimited by an equal sign (``=``) with no spaces
around. The key-value pairs are grouped into
*sections* (Neural Monkey requires all pairs to belong to a section.)

Every section starts with its header which consists of the section
name in square brackets. Everything below the header is considered a
part of the section.

Comments can appear on their own (otherwise empty) line, prefixed either with a
hash sign (``#``) or a semicolon (``;``) and possibly indented.

The configuration introduces several additional constructs for the
values. There are both atomic values, and compound values.

Supported atomic values are:

-  booleans: literals ``True`` and ``False``

-  integers: strings that could be interpreted as integers by Python
   (e.g., ``1``, ``002``)

-  floats: strings that could be interpreted as floats by Python (e.g.,
   ``1.0``, ``.123``, ``2.``, ``2.34e-12``)

-  strings: string literals in quotes (e.g., ``"walrus"``, ``"5"``)

-  section references: string literals in angle brackets (e.g.,
   ``<encoder>``), sections are later interpreted as Python objects

-  Python names: strings without quotes which are neither booleans, integers
   and floats, nor section references (e.g.,
   ``neuralmonkey.encoders.SentenceEncoder``)

On top of that, there are two compound types syntax from Python:

-  lists: comma-separated in squared brackets (e.g., ``[1, 2, 3]``)

-  tuples: comma-separated in round brackets (e.g.,
   ``("target", <ter>)``)


Variables
---------

The configuration file can contain a ``[vars]`` section, defining
variables which can be used in the rest of the config file. Their values
can have any of the types listed above. To reference a variable, use its
name preceded by a dollar sign (e.g. ``$variable``). Variable values can
also be included inside strings using the :py:func:`str.format` notation.
For example:

.. code-block:: ini

  [vars]
  parent_dir="experiments"
  drop_keep_p=0.5
  output_dir="{parent_dir}/test_drop{drop_keep_p:.2f}"
  prefix="my"

  [main]
  output=$output_dir

  ...

  [encoder]
  name="{prefix}_encoder"
  dropout_keep_prob=$drop_keep_p
  ...

Interpretation
--------------

Each configuration file contains a ``[main]`` section which is
interpreted as a dictionary having keys specified in the section and
values which are results of interpretation of the right hand sides.

Both the atomic and compound types taken from Python (i.e., everything
except the section references) are interpreted as their Python
counterparts. (So if you write 42, Neural Monkey actually sees 42.)

Section references are interpreted as references to
objects constructed when interpreting the referenced section. (So if
you write ``<session_manager>`` in a right-hand side and a section
``[session_manager]`` later in the file, Neural Monkey will construct
a Python object based on the key-value pairs in the section
``[session_manager]``.)

Every section except the ``[main]`` and ``[vars]`` sections needs
to contain the key ``class`` with
a value of Python name which is a callable (e.g., a class constructor or a
function). The other keys are used as named arguments of the callable.

Session Manager
---------------

This and following sections describes TensorFlow Manager from the users' perspective: what
can be configured in Neural Monkey with respect to TensorFlow.  The
configuration of the TensorFlow manager is specified within the INI file in
section with class :py:class:`neuralmonkey.tf_manager.TensorFlowManager`:

.. code-block:: ini

  [session_manager]
  class=tf_manager.TensorFlowManager
  ...

The ``session_manager`` configuration object is then referenced from the main
section of the configuration:

.. code-block:: ini

  [main]
  tf_manager=<session_manager>
  ...


Training on GPU
---------------

You can easily switch between CPU and GPU version by running your experiments
in virtual environment containing either CPU or GPU version of TensorFlow
without any changes to config files.

Similarly, standard techniques like setting the environment variable
``CUDA_VISIBLE_DEVICES`` can be used to control which GPUs are accessible for
Neural Monkey.

By default, Neural Monkey prefers to allocate GPU memory stepwise only as
needed. This can create problems with memory
fragmentation. If you know that you can allocate the whole memory at once
add the following parameter the ``session_manager`` section:

.. code-block:: ini

  gpu_allow_growth=False

You can also restrict TensorFlow to use only a fixed proportion of GPU memory:

.. code-block:: ini

  per_process_gpu_memory_fraction=0.65

This parameter tells TensorFlow to use only 65% of GPU memory.

Training on CPUs
----------------

TensorFlow Manager settings also affect training on CPUs.

The line:

.. code-block:: ini

  num_threads=4

indicates that 4 CPUs should be used for TensorFlow computations.
