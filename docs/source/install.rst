.. _instalation:

============
Installation
============


Before you start, make sure that you already have installed Python 3.5, pip 
and git.

Then clone Neural Monkey from GitHub::


	git clone https://github.com/ufal/neuralmonkey

Change your directory to ``neuralmonkey`` folder::


	cd neuralmonkey

And now run pip to install all requirements. If you don't want to upgrade your
packages, for example if you installed Tensorflow from sources and do not want
to replace it with a downloaded package, consult the requirements file and
install the dependencies by hand.
For CPU version install dependencies by this command::


	pip3 install --upgrade -r --requirements.txt

For GPU version install dependencies ty this command::


	pip3 install --upgrade -r --requirements-gpu.txt

If you are using the GPU version, make sure that the ``LD_LIBRARY_PATH``
environment variable points to ``lib`` and ``lib64`` directories of your CUDA
and CuDNN installations. Similarly, your ``PATH`` variable should point to the
``bin`` subdirectory of the CUDA installation directory.

You made it! Neural Monkey is now installed!
