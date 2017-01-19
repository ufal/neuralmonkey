.. _tensorflow-manager:

==================
TensorFlow manager
==================

TensorFlow manager is a helper object in Neural Monkey which manages TensorFlow
sessions, execution of the computation graph, and saving and restoring of model
variables.

This document describes a bunch of features that can be configured in
configuration files. The configuration of the TensorFlow manager is specified
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

By default, Neural Monkey will attempts to allocate only as much GPU memory
based on runtime allocations. This can create few problems with memory
fragmentation. If you know, that you can allocate the whole memory at once
add following parameter into your tensorflow manager::

  gpu_allow_growth=False

You can also restrict TensorFlow to use only part of GPU memory::

  per_process_gpu_memory_fraction=0.65

this parameter tells TensorFlow to use only 65% of GPU memory.
