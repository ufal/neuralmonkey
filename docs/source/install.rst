.. _instalation:

============
Installation
============

Before you start, make sure that you already have installed Python 3.5, pip
and git.

Create and activate a virtual environment to install the package into:

.. code-block:: bash

   $ python3 -m venv nm
   $ source nm/bin/activate
   # after this, your prompt should change

Then clone Neural Monkey from GitHub and switch to its root directory:

.. code-block:: bash

   (nm)$ git clone https://github.com/ufal/neuralmonkey
   (nm)$ cd neuralmonkey

Run pip to install all requirements. For the CPU version install
dependencies by this command:

.. code-block:: bash

   (nm)$ pip install --upgrade -r requirements.txt

For the GPU version install dependencies try this command:

.. code-block:: bash

   (nm)$ pip install --upgrade -r requirements-gpu.txt

If you are using the GPU version, make sure that the ``LD_LIBRARY_PATH``
environment variable points to ``lib`` and ``lib64`` directories of your CUDA
and CuDNN installations. Similarly, your ``PATH`` variable should point to the
``bin`` subdirectory of the CUDA installation directory.

You made it! Neural Monkey is now installed!

Note for Ubuntu 14.04 users
***************************

If you get Segmentation fault errors at the very end of the training process,
you can either ignore it, or follow the steps outlined in `this
document <ubuntu1404_fix.html>`_.
