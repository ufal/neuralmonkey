
Developers' guidelines
======================

.. warning:: This document is far from finished. For now, it should serve as a
             collection of hints for new developers.


This is a brief document describing the Neural Monkey development workflow.


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
