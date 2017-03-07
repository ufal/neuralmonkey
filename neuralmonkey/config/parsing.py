""" Module responsible for INI parsing """

from collections import OrderedDict
import configparser
import re
import time
# pylint: disable=unused-import
from typing import Any, Dict, Callable, Iterable, IO, List, Tuple, Optional
# pylint: enable=unused-import

from neuralmonkey.config.builder import ClassSymbol
from neuralmonkey.config.exceptions import IniError
from neuralmonkey.logging import log

LINE_NUM = re.compile(r"^(.*) ([0-9]+)$")

OBJECT_REF = re.compile(r"^<([a-zA-Z][a-zA-Z0-9_]*)>$")
INTEGER = re.compile(r"^[0-9]+$")
FLOAT = re.compile(r"^[0-9]*\.[0-9]*(e[+-]?[0-9]+)?$")
LIST = re.compile(r"\[([^]]*)\]")
TUPLE = re.compile(r"\(([^]]+)\)")
STRING = re.compile(r"^\"(.*)\"$")
CLASS_NAME = re.compile(
    r"^_*[a-zA-Z][a-zA-Z0-9_]*(\._*[a-zA-Z][a-zA-Z0-9_]*)+$")


CONSTANTS = {
    'False': False,
    'True': True,
    'None': None
}


# this is a function because of the parse_*
# functions which are not defined yet
def _keyval_parser_dict() -> Dict[Any, Callable]:
    return {
        INTEGER: int,
        FLOAT: float,
        STRING: lambda x: STRING.match(x).group(1),
        CLASS_NAME: _parse_class_name,
        OBJECT_REF: lambda x: "object:" + OBJECT_REF.match(x).group(1),
        LIST: _parse_list,
        TUPLE: _parse_tuple
    }


def _split_on_commas(string: str) -> List[str]:
    """Splits a bracketed string by commas, preserving any commas
    inside brackets."""

    items = []
    char_buffer = []  # type: List[Optional[str]]
    openings = []  # type: List[Optional[str]]

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


def _parse_list(string: str) -> List[Any]:
    """ Parses the string recursively as a list """

    matched_content = LIST.match(string).group(1)
    if matched_content == '':
        return []

    items = _split_on_commas(matched_content)
    values = [_parse_value(val) for val in items]
    types = [type(val) for val in values]

    if len(set(types)) > 1:
        raise Exception("List must of a same type, is: {}".format(types))

    return values


def _parse_tuple(string: str) -> Tuple[Any, ...]:
    """ Parses the string recursively as a tuple """

    items = _split_on_commas(TUPLE.match(string).group(1))
    values = [_parse_value(val) for val in items]

    return tuple(values)


def _parse_class_name(string: str) -> ClassSymbol:
    """ Parse the string as a module or class name.
    """
    return ClassSymbol(string)


def _parse_value(string: str) -> Any:
    """ Parses the value recursively according to the Nerualmonkey grammar.

    Arguments:
        string: the string to be parsed
    """

    if string in CONSTANTS:
        return CONSTANTS[string]

    for matcher, parser in _keyval_parser_dict().items():
        if matcher.match(string):
            return parser(string)

    raise Exception("Cannot parse value: '{}'.".format(string)) from None


def _parse_ini(config_file: Iterable[str], filename: str="") -> Dict[str, Any]:
    """ Parses an INI file into a dictionary """

    line_numbers = (line.strip() + " " + str(i + 1)
                    if line.strip() != "" else ""
                    for i, line in
                    enumerate(config_file))
    config = configparser.ConfigParser()
    config.read_file(line_numbers, source=filename)

    new_config = OrderedDict()  # type: Dict[str, Any]
    for section in config.sections():
        new_config[section] = OrderedDict()

        for key in config[section]:
            match = LINE_NUM.match(config[section][key])
            new_config[section][key] = match.group(2), match.group(1)

    return new_config


def _apply_change(config_dict: Dict[str, Any], setting: str) -> None:
    if '=' not in setting:
        raise Exception('Invalid setting "{}"'.format(setting))
    key, value = (s.strip() for s in setting.split('=', maxsplit=1))

    if '.' in key:
        section, option = key.split('.', maxsplit=1)
    else:
        section = 'main'
        option = key

    if section not in config_dict:
        log("Creating new section '{}'".format(section))
        config_dict[section] = OrderedDict()

    config_dict[section][option] = -1, value  # no line number


def parse_file(config_file: Iterable[str],
               changes: Optional[Iterable[str]]=None) -> Tuple[Dict[str, Any],
                                                               Dict[str, Any]]:
    """ Parses an INI file and creates all values """

    parsed_dicts = OrderedDict()  # type: Dict[str, Any]
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    config = _parse_ini(config_file)

    if changes is not None:
        for change in changes:
            _apply_change(config, change)

    for section in config:
        parsed_dicts[section] = OrderedDict()
        for key, (lineno, value_string) in config[section].items():
            # expansion
            # TODO do this using **kwargs with dict from names to values
            value_string = re.sub(r"\$TIME", time_stamp, value_string)

            try:
                value = _parse_value(value_string)
            except IniError as exc:
                raise
            except Exception as exc:
                raise IniError(
                    lineno, "Cannot parse value: '{}'.".format(value_string),
                    exc) from None

            parsed_dicts[section][key] = value

    # also return the unparsed config dict; need to remove line numbers
    raw_config = OrderedDict([
        (name, OrderedDict([(key, val) for key, (_, val) in section.items()]))
        for name, section in config.items()])

    return raw_config, parsed_dicts


def write_file(config_dict: Dict[str, Any], config_file: IO[str]) -> None:
    config = configparser.ConfigParser()
    config.read_dict(config_dict)
    config.write(config_file, space_around_delimiters=False)
