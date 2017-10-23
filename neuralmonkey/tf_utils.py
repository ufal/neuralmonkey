# pylint: skip-file
# Implementation of this feature is outdated

"""Small helper functions for TensorFlow."""

import os

from subprocess import check_output
from tensorflow.python.client import device_lib as _device_lib
import tensorflow as tf


__HAS_GPU_RESULT = None


def has_gpu() -> bool:
    """Check if TensorFlow can access GPU.

    The test is based on
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
    ...but we are interested only in CUDA GPU devices.

    Returns:
        True, if TF can access the GPU
    """
    # pylint: disable=global-statement
    global __HAS_GPU_RESULT
    # pylint: enable=global-statement
    if __HAS_GPU_RESULT is None:
        __HAS_GPU_RESULT = any((x.device_type == 'GPU')
                               for x in _device_lib.list_local_devices())
    return __HAS_GPU_RESULT


def gpu_memusage() -> str:
    """Return '' or a string showing current GPU memory usage.

    nvidia-smi result parsing based on https://github.com/wookayin/gpustat
    """
    if not has_gpu():
        return ''

    # gpu_query_columns = ('index', 'uuid', 'name', 'temperature.gpu',
    #                      'utilization.gpu', 'memory.used', 'memory.total')
    gpu_query_columns = ('index', 'memory.used', 'memory.total')
    gpu_list = []

    command = ['nvidia-smi',
               '--query-gpu=' + ','.join(gpu_query_columns),
               '--format=csv,noheader,nounits']

    visible_gpus = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_gpus:
        command.append('--id=' + visible_gpus)

    smi_output = check_output(command).decode().strip()

    for line in smi_output.split('\n'):
        if not line:
            continue
        if line == "fake modprobe":
            continue
        query_results = line.split(',')
        gpu_res = {col_name: col_value.strip()
                   for (col_name, col_value)
                   in zip(gpu_query_columns, query_results)}
        gpu_list.append(gpu_res)

    if len(gpu_list) == 1:
        stats = gpu_list[0]
        info = ['{}/{}'.format(stats['memory.used'], stats['memory.total'])]
    else:
        info = ['{}:{}/{}'.format(e['index'],
                                  e['memory.used'], e['memory.total'])
                for e in gpu_list]

    return 'MiB:' + ",".join(info)

class Saver(tf.train.Saver):
    """TODO: docstring."""

    def __init__(max_to_keep):
        """TODO: docstring."""
        self._saver = tf.train.Saver(max_to_keep)
        self._save_file = None

    def save(self,
             sess,
             save_path,
             global_step=None,
             latest_filename=None,
             meta_graph_suffix='meta',
             write_meta_graph=True,
             write_state=True):
        if self.save_file:
            self.saver.save(sess=sess,
                            save_path=self.save_file,
                            global_step=global_step,
                            latest_filename=latest_filename,
                            meta_graph_suffix=meta_graph_suffix,
                            write_meta_graph=write_meta_graph,
                            write_state=write_state)
            self.save_file = None

    def set_save_file(self, file_path):
        pass

    def restore(self):
        pass
