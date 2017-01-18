.. _developers-guidlines:

======================
Developers' guidelines
======================

This is a brief document about the Neural Monkey development workflow. Its
primary aim is to describe the environment around the Github repository
(e.g. continuous integration tests, documentation), pull requests, code-review,
etc.

This document is written chronologically, from the point of view of a
contributor.


Creating an issue
-----------------

Everytime there is a need to change the codebase, the contributor should create
a corresponing issue on Github.

The name of the issue should be comprehensive, and should summarize the issue in
less than 10 words.  In the issue description, all the relevant information
should be mentioned, and, if applicable, a sketch of the solution should be
given so the fashion and method of the solution can be subject to further
discussion.

Labels
******

There is a number of label tags to use to provide an easier way to orient among
the issues. Here is an explanation of some of them, so they are not used
incorrectly (notably, there is a slight difference between "enhancement" and
"feature").

- bug: Use when there is something wrong in the current codebase that needs to
  be fixed. For example, "Random seeds are not working"
- documentation: Use when the main topic of the issue or pull request is to
  contribute to the documentation (be it a rst document or a request for more
  docstrings)
- tests: Similarly to documentation, use if the main topic of the issue is to
  write a test or to do changes to the testing process itself.
- feature: A request for implementing a feature regarding the training of the
  models or the models themselves, e.g. "Minimum risk training" or
  "Implementation of conditional GRU".
- enhancement: A request for implementing a feature to Neural Monkey aimed at
  improving the user experience with the package, e.g. "GPU profiling" or
  "Logging of config building".
- help wanted: Used as an additional label, which specify that solving the issue
  is suitable either for new contributors or for researchers who want to try out
  a feature, which would be otherwise implemented after a longer time.
- refactor: Refactor issues are requests for cleaning the codebase, using better
  ways to achieve the same results, conforming to a future API, etc. For
  example, "Rewrite decoder using decorators"

.. todo::
   Replace text with label pictures from Github


Selecting an issue to work on and assigning people
--------------------------------------------------

.. note:: If you want to start working on something and don't have a preference,
   check out the issues labeled "Help wanted"

When you decide to work on an issue, assign yourself to it and describe your
plans on how you will proceed (in case there is no solution sketch provided in
the issue description). This way, others may comment on your plans prior to the
work, which can save a lot of time.

Please make sure that you put all additional information as a comment to the
issue in case the issue has been discussed elsewhere.


Creating a branch
-----------------

Prior to writing code (or at least before the first commit), you should create a
branch for solution of the issue. This command creates a new branch called
``new_branch_name`` and switches your working copy to that branch::

.. code-block:: bash

$ git checkout -b your_branch_name


Writing code
------------

On the new branch, you can make changes and commit, until your solution is done.

It is worth noting that we are trying to keep our code clean by enforcing some
code writing rules and guidelines. These are automatically check by Travis CI on
each push to the Github repository. Here is a list of tools used to check the
quality of the code:

* `pylint <https://www.pylint.org>`_
* `pycodestyle <http://pypi.python.org/pypi/pycodestyle>`_
* `mypy <http://mypy-lang.org>`_
* `markdownlint <https://github.com/mivok/markdownlint>`_

.. todo:: provide short description to the tools, check that markdownlint has
          correct URL

You can run the tests on your local machine by using scripts (and requirements)
from the ``tests/`` directory of this package,

This is a usual mantra that you can use for committing and pushing to the remote
branch in the repository:

.. code-block:: bash
$ git add .
$ git commit -m 'your commit message'
$ git push origin your_branch_name

.. note:: If you are working on a branch with someone else, it is always a good
          idea to do a ``git pull --rebase`` before pushing. This command
          updates your branch with remote changes and apply your new commits on
          top of them.

.. warning:: If your commit message contains the string ``[ci skip]`` the
	     continuous integration tests are not run. However, try not to use
	     this feature unless you know what you're doing.


Creating a pull request
-----------------------

Whenever you want to add a feature or push a bugfix, you should make a new pull
request, which can be reviewed and merged by someone else. The typical workflow
should be as follows:

1. Create a new branch, make your changes and push them to the repository.

2. You should now see the new branch on the Github project page. When you open
   the branch page, click on "Create Pull request" button.

3. When the pull request is created, the continuous integration tests are run on
   Travis. You can see the status of the test run on the pull request
   page. There is also a link to Travis so you can inspect the results of the
   test run, and make additional changes in order to make the tests successful,
   if needed. Additionally to the code quality checking tools, unit and
   regression tests are run as well.

When you create a pull request, assign one or two people to do the review.


Code review and merging
-----------------------

Your pull requests should always be subject to code review. After you create the
pull request, select one or two contributors and assign them to make a review.

This phase consists of discussion about the introduced changes, suggestions, and
another requirements made by the reviewers. Anyone who wants to do a review can
contribute, the reviewer roles are not considered exclusive.

After all of the reviewers' comments have been addressed and the reviewers
approved the pull request, the pull request can be merged. It is usually a good
idea to rebase the code to the recent version of master. Assuming your working
copy is switched to the **master** branch, do::

.. code-block:: bash
$ git pull --rebase
$ git checkout your_branch_name
$ git rebase master

These commands first update your local copy of master from the remote
repository, then switch your working copy to the ``your_branch_name`` branch,
and then rebases the branch on the updated master.

Rebasing is a process in which commits from a branch (``your_branch_name``) are
applied on a second branch (master), and the new HEAD is marked as the first
branch.

.. warning:: Rebasing is a process which overwrites history. Therefore be
             absolutely sure that you know what are you doing. Usually if you
             work on a branch alone, rebasing is a safe procedure.

When the branch is rebased, you have to force-push it to the repository::

.. code-block:: bash
$ git push -f origin your_branch_name

This command overwrites the your branch in the remote repository with your local
branch (which is now rebased on master, and therefore, up-to-date)

.. note:: You can use rebasing also for updating your branch to work with newer
          versions of master instead of merging the master in the branch. Bear
          in mind though, that you should force-push these updates, so no-one
          works on the outdated version of the branch.

Finally, one more round of tests is run and if everything is OK, you can click
the "Merge pull request" button, which executes the merge. You can also click
another button to delete the ``your_branch_name`` branch from the repository
after the merge.


Documentation
-------------

Documentation related to GitHub is written in `Markdown
<https://daringfireball.net/projects/markdown/>` files, Python documentation
using `reStructuredText
<http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html>`. This
concerns both the standalone documents (in `/docs/`) and the docstrings in
source code.

Style of the Markdown files is automatically checked using `Markdownlint
<https://github.com/mivok/markdownlint>`.
