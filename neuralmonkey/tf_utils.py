""" Small helper functions for TensorFlow
"""

from tensorflow.python.client import device_lib as _device_lib

from subprocess import check_output


__has_gpu_result = None


def has_gpu():
    """ Check if TensorFlow can access GPU
    The test is based on
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
    ...but we are interested only in CUDA GPU devices.

    Returns:
        True, if TF can access the GPU
    """
    global __has_gpu_result
    if __has_gpu_result is None:
        __has_gpu_result = any((x.device_type == 'GPU')
                               for x in _device_lib.list_local_devices())
    return __has_gpu_result


def gpu_memusage():
    """ Return '' or a string showing current GPU memory usage
    nvidia-smi result parsing based on https://github.com/wookayin/gpustat
    """
    if not has_gpu():
        return ''

    # gpu_query_columns = ('index', 'uuid', 'name', 'temperature.gpu',
    #                      'utilization.gpu', 'memory.used', 'memory.total')
    gpu_query_columns = ('index', 'memory.used', 'memory.total')
    gpu_list = []

    smi_output = check_output(
        r'nvidia-smi --query-gpu={query_cols} --format=csv,noheader,nounits'
        .format(query_cols=','.join(gpu_query_columns)),
        shell=True).decode().strip()

    for line in smi_output.split('\n'):
        if not line: continue
        query_results = line.split(',')
        g = {col_name: col_value.strip()
             for (col_name, col_value)
             in zip(gpu_query_columns, query_results)}
        gpu_list.append(g)

    if len(gpu_list) == 1:
        e = gpu_list[0]
        info = ['{}/{}'.format(e['memory.used'], e['memory.total'])]
    else:
        info = ['{}:{}/{}'.format(e['index'],
                                  e['memory.used'], e['memory.total'])
                for e in gpu_list]

    return 'MiB:' + ",".join(info)
