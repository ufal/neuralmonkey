Configuration
=============

Experiments with NeuralMonkey are configured using configuration files
which specifies the architecture of the model, meta-parameters of the
learning, the data, the way the data are processed and the way the model
is run.

Syntax
------

The configuration files extends syntax of INI files. INI files contain
*key-value pairs*, delimited by an equal sign (``=``) with no spaces
around. Traditionally, everything on the right side of the equal sign is
interpreted as a string. The key-value pairs can be grouped into
*sections*. Section starts with its header which consists of the section
name in square brackets. Everything below the header is considered a
part of the section. The files can also contain comments starting with
semi-colons (``;``). You can find more details on INI syntax, e.g., on
the corresponding `Wikipedia
page <https://en.wikipedia.org/wiki/INI_file>`__.

The configuration introduces several additional constructs for the
values. There are both atomic values, and compound values.

The atomic values are:

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
   ``neuraLmonkey.encoders.SentenceEncoder``)

On top of that there are two compound types syntax from Python:

-  lists: comma-separated in squared brackets (e.g., ``[1, 2, 3]``)

-  tuples: comma-separated in round brackets (e.g.,
   ``("target", <ter>)``)

We also require to have all key-value pairs in sections.

Interpretation
--------------

Each configuration file contains a ``[main]`` section which is
interpreted as a dictionary having keys specified in the section and
values which are results of interpretation of the INI file values.

Both the atomic and compound types taken from Python (i.e., everything
except the section references) are interpreted as their Python
counterparts. The section references are interpreted as references to
objects which are results of section interpretation.

Every section except the ``[main]`` section needs to contain key ``class`` with
a value of Python name which is a callable (e.g., a class constructor or a
function). The other keys are used as named arguments of the callable.
