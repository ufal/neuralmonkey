""" This module contains function that are generate configuration INI file for
running model, if it is provided a coniguration to train a model. """

import codecs
import vocabulary
from inspect import isfunction

from utils import log

def final_dict_to_ini(path, dictionary):
    f_ini = codecs.open(path, 'w', 'utf-8')
    for name, values in dictionary.iteritems():
        f_ini.write("[{}]\n".format(name))
        for key, val in values.iteritems():
            f_ini.write("{}={}\n".format(key, val))
        f_ini.write("\n")
    f_ini.close()


def object_to_dict(obj, final_dict, depth):
    print obj
    if obj is None:
        return 'None'
    elif isinstance(obj, basestring):
        return obj
    elif isinstance(obj, int) or isinstance(obj, float):
        return str(obj)
    elif isinstance(obj, type):
        return obj.__module__ + "." + obj.__name__
    elif isinstance(obj, list):
        return [object_to_dict(item, final_dict, depth + 1) for item in obj]
    elif isfunction(obj):
        return "{}.{}".format(obj.__module__, obj.__name__)
    elif obj in final_dict:
        return final_dict[obj]
    elif isinstance(obj, vocabulary.Vocabulary):
        # TODO pickle to file and save path
        return "unimplemented_vocabulary"
    else:
        clazz = obj.__class__
        init_f = clazz.__init__.func_code
        argument_count = init_f.co_argcount
        # from all variables take only arguments without self
        argument_names = init_f.co_varnames[1:argument_count]

        base_name = clazz.__name__
        if hasattr(obj, 'name'):
            base_name = obj.name
        name = base_name
        distinct_id = 0
        while name in final_dict:
            name = "{}_{}".format(base_name, distinct_id)
            distinct_id += 1

        obj_dict = {'class' : clazz.__module__ + "." + clazz.__name__}
        for arg in argument_names:
            try:
                obj_dict[arg] = object_to_dict(obj.__dict__[arg], final_dict, depth + 1)
            except KeyError:
                log('Class "{}" in module "{}" is missing attribute "{}"'.\
                        format(clazz.__name__, clazz.__module__, arg), 'red')
                exit(1)
        return "<{}>".format(name)


def save_configuration(configuration):
    final_dict = {}
    main_dict = {}
    for key, value in configuration.iteritems():
        main_dict[key] = object_to_dict(value, final_dict, 0)

    final_dict['main'] = main_dict
    return final_dict
