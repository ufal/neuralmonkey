""" This module contains function that generate configuration INI file for
running model, if it is provided a coniguration to train a model. """

# TODO: ^ this does not make any sense! TM

import codecs
import os
from inspect import isfunction

import vocabulary
from utils import log

def final_dict_to_ini(path, dictionary):
    f_ini = codecs.open(path, 'w', 'utf-8')
    for name, values in dictionary.iteritems():
        f_ini.write("[{}]\n".format(name))
        for key, val in values.iteritems():
            f_ini.write("{}={}\n".format(key, val))
        f_ini.write("\n")
    f_ini.close()


def object_to_dict(obj, final_dict, name_dict, depth, out_dir):
    if obj is None:
        return 'None'
    elif isinstance(obj, basestring):
        return obj
    elif isinstance(obj, int) or isinstance(obj, float):
        return str(obj)
    elif isinstance(obj, type):
        return obj.__module__ + "." + obj.__name__
    elif isinstance(obj, list):
        string_list = [object_to_dict(item, final_dict, name_dict, depth + 1,
                                      out_dir)
                       for item in obj]
        return "[{}]".format(",".join(string_list))
    elif isfunction(obj):
        return "{}.{}".format(obj.__module__, obj.__name__)
    elif obj in name_dict:
        return "<{}>".format(name_dict[obj])
    elif isinstance(obj, vocabulary.Vocabulary):
        voc_name = "vocabulary_{}".format(obj.__hash__())
        file_name = "{}/{}.pickle".format(out_dir, voc_name)
        if not os.path.isfile(file_name):
            obj.save_to_file(file_name)
        vocabulary_obj = {
            'class': 'vocabulary.from_pickled',
            'path': file_name
        }
        final_dict[voc_name] = vocabulary_obj
        name_dict[obj] = voc_name
        return "<{}>".format(voc_name)
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
                obj_dict[arg] = object_to_dict(obj.__dict__[arg], final_dict,
                                               name_dict, depth + 1, out_dir)
            except KeyError:
                log("Error while generation the run-time configuration.")
                log('Class "{}" in module "{}" is missing attribute "{}"'
                    .format(clazz.__name__, clazz.__module__, arg), 'red')
                exit(1)
        final_dict[name] = obj_dict
        name_dict[obj] = name
        return "<{}>".format(name)


def save_configuration(configuration, out_dir):
    final_dict = {}
    name_dict = {}
    main_dict = {}
    for key, value in configuration.iteritems():
        main_dict[key] = object_to_dict(value, final_dict, name_dict, 0,
                                        out_dir)

    final_dict['main'] = main_dict
    final_dict_to_ini(out_dir+"/run.ini", final_dict)
