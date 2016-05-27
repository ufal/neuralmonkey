"""
This module is responsible for loading training configuration.
"""

import traceback
import collections
import regex as re
from utils import log


OBJECT_NAME = re.compile(r"^\[([a-zA-Z][a-zA-Z0-9_]*)\]$")
OBJECT_REF = re.compile(r"^<([a-zA-Z][a-zA-Z0-9_]*)>$")
KEY_VALUE_PAIR = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*) *= *(.+)$")
INTEGER = re.compile(r"^[0-9]+$")
FLOAT = re.compile(r"^[0-9]*\.[0-9]*(e[+-]?[0-9]+)?$")
LIST = re.compile(r"\[([^]]*)\]")
TUPLE = re.compile(r"\(([^]]+)\)")
CLASS_NAME = re.compile(r"^_*[a-zA-Z][a-zA-Z0-9_]*(\._*[a-zA-Z][a-zA-Z0-9_]*)+$")

def split_on_commas(string):
    """
    This is a clever splitter a bracketed string on commas.
    """
    items = []
    char_buffer = []
    openings = []
    for i, c in enumerate(string):
        if c == ',' and len(openings) == 0:
            items.append("".join(char_buffer))
            char_buffer = []
            continue
        elif c == '(' or c == '[':
            openings.append(c)
        elif c == ')':
            if openings.pop() != '(':
                raise Exception('Invalid bracket end ")", col {}.'.format(i))
        elif c == ']':
            if openings.pop() != '[':
                raise Exception('Invalid bracket end "]", col {}.'.format(i))
        char_buffer.append(c)
    items.append("".join(char_buffer))
    return items



def format_value(string):
    #pylint: disable=too-many-return-statements
    """ Parses value from the INI file: int/float/string/object """
    if string == 'False':
        return False
    elif string == 'True':
        return True
    elif string == 'None':
        return None
    elif INTEGER.match(string):
        return int(string)
    elif FLOAT.match(string):
        return float(string)
    elif CLASS_NAME.match(string):
        class_parts = string.split(".")
        class_name = class_parts[-1]
        module_name = ".".join(class_parts[:-1])
        module = __import__(module_name)
        clazz = getattr(module, class_name)
        return clazz
    elif OBJECT_REF.match(string):
        return "object:"+OBJECT_REF.match(string)[1]
    elif LIST.match(string):
        items = split_on_commas(LIST.match(string)[1])
        values = [format_value(val) for val in items]
        types = [type(val) for val in values]
        #log(values)
        if len(set(types)) > 1:
            raise Exception("List must of a same type, is: {}".format(types))
        return values
    elif TUPLE.match(string):
        #log("Here is tuple: {}".format(string))
        items = split_on_commas(TUPLE.match(string)[1])
        values = [format_value(val) for val in items]
        return tuple(values)
    else:
        return string


def get_config_dicts(config_file):
    """ Parses the INI file into a dictionary """
    config_dicts = dict()

    current_name = None
    for i, line in enumerate(config_file):
        try:
            line = line.strip()
            if not line:
                pass
            elif line.startswith(";"):
                pass
            elif OBJECT_NAME.match(line):
                current_name = OBJECT_NAME.match(line)[1]
                if current_name in config_dicts:
                    raise Exception("Duplicit object key: '{}', line {}.".format(current_name, i))
                config_dicts[current_name] = dict()
            elif KEY_VALUE_PAIR.match(line):
                matched = KEY_VALUE_PAIR.match(line)
                key = matched[1]
                value_string = matched[2]
                if key in config_dicts[current_name]:
                    raise Exception("Duplicit key in '{}' object, line {}.".format(key, i))
                config_dicts[current_name][key] = format_value(value_string)
            else:
                raise Exception("Unknown string: \"{}\"".format(line))
        except Exception as exc:
            log("Syntax error on line {}: {}".format(i, exc.message), color='red')
            exit(1)

    config_file.close()
    return config_dicts


def get_object(value, all_dicts, existing_objects, depth):
    """
    Constructs an object from dict with its arguments. It works recursively.

    Args:

        value: A value that should be resolved (either a singular value or
            object name)

        all_dicts: Raw configuration dictionaries. It is used to find configuration
            of unconstructed objects.

        existing_objects: A dictionary for keeping already constructed objects.

        depth: Current depth of recursion. Used to prevent an infinite recursion.

    """
    if not isinstance(value, basestring) and isinstance(value, collections.Iterable):
        return [get_object(val, all_dicts, existing_objects, depth + 1)
                for val in value]
    if value in existing_objects:
        return existing_objects[value]
    if not isinstance(value, basestring) or not value.startswith("object:"):
        return value

    name = value[7:]
    if name not in all_dicts:
        raise Exception("Object \"{}\" was not defined in the configuration.".format(name))
    this_dict = all_dicts[name]

    if depth > 20:
        raise Exception("Configuration does also object depth more thatn 20.")
    if 'class' not in this_dict:
        raise Exception("Class is not defined for object: {}".format(name))

    clazz = this_dict['class']

    def process_arg(arg):
        """ Resolves potential references to other objects """
        return get_object(arg, all_dicts, existing_objects, depth + 1)

    args = {k: process_arg(arg) for k, arg in this_dict.iteritems() if k != 'class'}

    try:
        result = clazz(**args)
    except Exception as exc:
        log("Failed to create object \"{}\" of class \"{}.{}\": {}"\
                .format(name, clazz.__module__, clazz.__name__, exc.message), color='red')
        traceback.print_exc()
        exit(1)
    existing_objects[value] = result
    return result


def load_config_file(config_file):
    """ Loads the complete configuration of an experiment. """
    config_dicts = get_config_dicts(config_file)
    log("INI file is parsed.")

    # first load the configuration into a dictionary

    if "main" not in config_dicts:
        raise Exception("Configuration does not contain the main block.")

    existing_objects = dict()

    main_config = config_dicts['main']

    configuration = dict()
    for key, value in main_config.iteritems():
        try:
            configuration[key] = get_object(value, config_dicts,
                                            existing_objects, 0)
        except Exception as exc:
            log("Error while loading {}: {}".format(key, exc.message), color='red')
            traceback.print_exc()
            exit(1)

    return configuration
