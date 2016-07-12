#tests: lint

import re
import importlib
import time

from neuralmonkey.config.exceptions import IniError

SECTION_HEADER = re.compile(r"^\[([a-zA-Z][a-zA-Z0-9_]*)\]$")
KEY_VALUE_PAIR = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*) *= *(.+)$")
OBJECT_NAME = re.compile(r"^\[([a-zA-Z][a-zA-Z0-9_]*)\]$")

OBJECT_REF = re.compile(r"^<([a-zA-Z][a-zA-Z0-9_]*)>$")
INTEGER = re.compile(r"^[0-9]+$")
FLOAT = re.compile(r"^[0-9]*\.[0-9]*(e[+-]?[0-9]+)?$")
LIST = re.compile(r"\[([^]]*)\]")
TUPLE = re.compile(r"\(([^]]+)\)")
CLASS_NAME = re.compile(
    r"^_*[a-zA-Z][a-zA-Z0-9_]*(\._*[a-zA-Z][a-zA-Z0-9_]*)+$")


CONSTANTS = {
    'False': False,
    'True': True,
    'None': None
}


## this is a function because of the parse_*
## functions which are not defined yet
def _keyval_parser_dict():
    return {
        INTEGER: int,
        FLOAT: float,
        CLASS_NAME: parse_class_name,
        OBJECT_REF: lambda x: "object:" + OBJECT_REF.match(x).group(1),
        LIST: parse_list,
        TUPLE: parse_tuple
    }


def split_on_commas(string):
    """Splits a bracketed string by commas, preserving any commas
    inside brackets."""

    items = []
    char_buffer = []
    openings = []

    for i, char in enumerate(string):
        if char == ',' and len(openings) == 0:
            if len(char_buffer) > 0:
                items.append("".join(char_buffer))
            char_buffer = []
            continue
        elif char == ' ' and len(char_buffer) == 0:
            continue
        elif char == '(' or char == '[':
            openings.append(char)
        elif char == ')':
            if openings.pop() != '(':
                raise Exception('Invalid bracket end ")", col {}.'.format(i))
        elif char == ']':
            if openings.pop() != '[':
                raise Exception('Invalid bracket end "]", col {}.'.format(i))
        char_buffer.append(char)

    if len(char_buffer) > 0:
        items.append("".join(char_buffer))
    return items


def parse_list(string):
    """ Parses the string recursively as a list """

    matched_content = LIST.match(string).group(1)
    if matched_content == '':
        return []

    items = split_on_commas(matched_content)
    values = [parse_value(val) for val in items]
    types = [type(val) for val in values]

    if len(set(types)) > 1:
        raise Exception("List must of a same type, is: {}".format(types))

    return values


def parse_tuple(string):
    """ Parses the string recursively as a tuple """

    items = split_on_commas(TUPLE.match(string)[1])
    values = [parse_value(val) for val in items]

    return tuple(values)


def parse_class_name(string):
    """ Parse the string as a module or class name.
    Raises Exception when the class (or module) cannot be imported.
    """

    class_parts = string.split(".")
    class_name = class_parts[-1]

    # TODO should we not assume that everything is from neuralmonkey?
    module_name = ".".join(["neuralmonkey"] + class_parts[:-1])

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # if the problem is really importing the module
        if exc.name == module_name:
            raise Exception(("Interpretation '{}' as type name, module '{}' "
                             "does not exist. Did you mean file './{}'? \n{}")
                            .format(string, module_name, string, exc)) from None
        else:
            raise

    try:
        clazz = getattr(module, class_name)
    except AttributeError as exc:
        raise Exception(("Interpretation '{}' as type name, class '{}' "
                         "does not exist. Did you mean file './{}'? \n{}")
                        .format(string, class_name, string, exc))
    return clazz


def parse_value(string):
    """ Parses the value recursively according to the Nerualmonkey grammar.

    Arguments:
        string: the string to be parsed
    """

    if string in CONSTANTS:
        return CONSTANTS[string]

    for matcher, parser in _keyval_parser_dict().items():
        if matcher.match(string):
            return parser(string)

    return string


def preprocess_line(line, time_stamp):
    """ Remove comments and trim whitespaces from a line
    of the configuration file. Also performs expansion of variables.

    Arguments:
        line: Line from a config file to be processed.

    Supported variables:
        $TIME - replace this variable with a current time
    """

    line = line.strip()

    if line.startswith(';'):
        return None

    line = re.sub(r"#.*", "", line)
    if line == "":
        return None

    # expansion
    # TODO do this using **kwargs with dict from names to values
    line = re.sub(r"\$TIME", time_stamp, line)

    return line


def parse_file(config_file):
    """ Parses an INI file into a dictionary """

    parsed_dicts = dict()
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    current_name = None
    global_dict = dict()

    for lineno, line_raw in enumerate(config_file):
        line = preprocess_line(line_raw, time_stamp)
        if not line:
            continue

        ## Two kinds of lines: Section headers and key-value pairs
        if SECTION_HEADER.match(line):
            current_name = OBJECT_NAME.match(line).group(1)

            if current_name in parsed_dicts:
                raise IniError(
                    lineno + 1, "Duplicit section: '{}'".format(current_name))

            parsed_dicts[current_name] = dict()

        elif KEY_VALUE_PAIR.match(line):
            matched = KEY_VALUE_PAIR.match(line)
            key = matched.group(1)
            value_string = matched.group(2)

            selected_dict = global_dict

            if current_name is not None:
                selected_dict = parsed_dicts[current_name]


            if key in selected_dict:
                raise IniError(
                    lineno + 1, "Duplicit key in '{}' section.".format(key))

            try:
                value = parse_value(value_string)
            except IniError as exc:
                raise
            except Exception as exc:
                raise IniError(lineno + 1, "Error", exc) from None

            selected_dict[key] = value

        else:
            raise IniError(lineno + 1,
                           "Unknown string: '{}'".format(line))

    return parsed_dicts
