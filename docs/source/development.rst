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
given so the fasion and method of the solution can be subject to further
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
  ways to achieve the same results, conforming to a future API, etc. For example,
  "Rewrite decoder using decorators"



Selecting an issue to work on and assigning people
--------------------------------------------------




Commiting code
--------------

Use pull requests to introduce new changes. Whenever you want to add a
feature or push a bugfix, you should make a new pull request, which can be
reviewed and merged by someone else. The typical workflow should be as follows:

1. Create a new branch for the changes. Use ``git checkout -b branch_name`` to
   create a new branch and switch the working copy to that branch.

2. Make your changes to the code, commit them to the new branch.

3. Push the new branch to the repository by typing ``git push origin
   branch_name``, or just ``git push``.

4. You should now see the new branch on the Github project page. When you open
   the branch page, click on "Create Pull request" button.

5. When the pull request is created, the tests are run on Travis. You can see
   the status of the test run on the pull request page. There is also a link to
   Travis so you can inspect the results of the test run, and make additional
   changes in order to make the tests successful, if needed.

6. Simultaneously to the test runs, anyone now can look at the code changes by
   clicking on the "Files changed" tab and comment on individual lines in the
   diff, and approve or reject the proposed pull request.

7. When all the tests are passing and the pull request is approved, it can be
   merged.

.. note:: When you are pushing a commit that does not change the functionality
          of the code (e.g. changes in comments, documentation, etc.), you can
          push them to master directly (TODO: subject to discussion). Every
          commit that does not affect the program behavior should be marked with
          ``[ci skip]`` inside the commit message.

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

Other
-----

.. todo:: describe other stuff too
