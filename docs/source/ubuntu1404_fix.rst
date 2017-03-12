Fixing segmentation fault on exit on Ubuntu 14.04
=================================================

* On Ufal machines, segfault can be prevented by doing this:

.. code-block:: bash

  export LD_PRELOAD=/home/helcl/lib/libtcmalloc_minimal.so.4
  bin/neuralmonkey-train tests/vocab.ini

* On machines with ``sudo``, one can do this:

.. code-block:: bash

  sudo apt-get install libtcmalloc-minimal4
  export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"

* On machines with neither ``sudo`` nor
  ``~helcl/lib/libtcmalloc_minimal.so.4``, this is the way to fix segfaulting:

.. code-block:: bash

  wget http://archive.ubuntu.com/ubuntu/pool/main/g/google-perftools/google-perftools_2.1.orig.tar.gz
  tar xpzvf google-perftools_2.1.orig.tar.gz
  cd gperftools-2.1/
  ./configure --prefix=$HOME
  make
  make install

if the compilation crashes on the need of the ``libunwind`` library (as did for
me), do this:

.. code-block:: bash

  wget http://download.savannah.gnu.org/releases/libunwind/libunwind-0.99-beta.tar.gz
  tar xpzvf libunwind-0.99-beta.tar.gz
  cd libunwind-0.99-beta/
  ./configure --prefix=$HOME
  make
  make install

if, by any chance, compilation of this crashes on something like: ``error:
'longjmp' aliased to undefined symbol '_longjmp'``, replace the ``make`` call
with ``make CFLAGS+=-U_FORTIFY_SOURCE`` command.

Then, in ``$HOME/share`` directory, create file ``config.site`` like this:

.. code-block:: bash

  cat << EOF > $HOME/share/config.site
  CPPFLAGS=-I$HOME/include
  LDFLAGS=-L$HOME/lib
  EOF

and then redo the configure-make-make install mantra from gperftools. Finally,
set the ``LD_PRELOAD`` environment variable to point to
``$HOME/lib/libtcmalloc_minimal.4.so``.
