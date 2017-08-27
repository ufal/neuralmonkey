

=============
Running tests
=============

Every time a commit is pushed to the Github `repository
<https://github.com/ufal/neuralmonkey>`_, the tests are run on `Travis CI
<https://travis-ci.org/ufal/neuralmonkey>`_.

If you want to run the tests locally, install the required tools:


.. code-block:: bash

   (nm)$ pip install --upgrade -r <(cat tests/*_requirements.txt)


Test scripts
------------

Test scripts located in the `tests` directory:

- `tests_run.sh` runs training with small dataset and `small.ini` configuration
- `unit-tests_run.sh` runs unit tests
- `lint_run.sh` runs pylint
- `mypy_run.sh` runs mypy

All the scripts should be run from the main directory of the repository. There
is also `run_tests.sh` in the main directory, that runs all the tests above.
