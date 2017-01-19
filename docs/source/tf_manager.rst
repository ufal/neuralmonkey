.. _tensorflow-manager:

==================
TensorFlow Manager
==================

TensorFlow manager is a helper object in Neural Monkey which manages TensorFlow
sessions, execution of the computation graph, and saving and restoring of model
variables.

This document describes
TensorFlow Manager from the users' perspective: what can be configured in Neural Monkey with respect to TensorFlow.
The configuration of the TensorFlow manager is specified
within the INI file in section with class ``tf_manager.TensorFlowManager``::

  [session_manager]
  class=tf_manager.TensorFlowManager
  ...

The ``session_manager`` configuration object is then referenced from the main
section of the configuration::

  [main]
  tf_manager=<session_manager>
  ...

Training on GPU
---------------

You can easily switch between CPU and GPU version by running your experiments in
virtual environment containing either CPU or GPU version of TensorFlow
without any changes to config files.

Similarly, standard techniques like setting the environment variable
``CUDA_VISIBLE_DEVICES`` can be used to control which GPUs are accessible for
Neural Monkey.

By default, Neural Monkey prefers to allocate GPU memory stepwise only as
needed. This can create problems with memory
fragmentation. If you know that you can allocate the whole memory at once
add the following parameter the ``session_manager`` section::

  gpu_allow_growth=False

You can also restrict TensorFlow to use only a fixed proportion of GPU memory::

  per_process_gpu_memory_fraction=0.65

This parameter tells TensorFlow to use only 65% of GPU memory.

With the default ``gpu_allow_growth=True``, it makes sense to monitor memory
consumption. Neural Monkey can include a short summary total GPU memory used
in the periodic log line. Just set::

  report_gpu_memory_consumption=True

The log line will then contain the information like:
``MiB:0:7971/8113,1:4283/8113``. This particular message means that there are
two GPU cards and the one indexed 1 has 4283 out of the total 8113 MiB
occupied. Note that the information reports all GPUs on the machine, regardless
``CUDA_VISIBLE_DEVICES``.


Training on CPUs
----------------

TensorFlow Manager settings also affect training on CPUs.

The line::

  num_threads=4

indicates that 4 CPUs should be used for TensorFlow computations.

