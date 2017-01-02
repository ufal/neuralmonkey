.. _tensorflow-manager:

==================
TensorFlow manager
==================

Training on GPU
---------------

You can easily switch between CPU and GPU version by running your experiments in
virtual environment containing either CPU or GPU version of TensorFlow
without any changes to config files.

By default, Neural Monkey will attempts to allocate only as much GPU memory
based on runtime allocations. This can create few problems with memory
fragmentation. If you know, that you can allocate the whole memory at once
add following parameter into your tensorflow manager::

  [tf_manager]
  gpu_allow_growth=False
  ...

You can also restrict TensorFlow to use only part of GPU memory::

  [tf_manager]
  per_process_gpu_memory_fraction=0.65
  ...

this parameter tells TensorFlow to use only 65% of GPU memory.