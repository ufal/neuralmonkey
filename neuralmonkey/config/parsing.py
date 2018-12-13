"""Module responsible for INI parsing."""

from collections import OrderedDict
import configparser
import os
import re
import time
from typing import Any, Dict, Callable, Iterable, IO, List, Tuple, Pattern

from neuralmonkey.config.builder import ClassSymbol, ObjectRef
from neuralmonkey.config.exceptions import ParseError
from neuralmonkey.logging import log

LINE_NUM = re.compile(r"^(.*) ([0-9]+)$")

INTEGER = re.compile(r"^-?[0-9]+$")
FLOAT = re.compile(r"^-?[0-9]*\.[0-9]*(e[+-]?[0-9]+)?$|^-?[0-9]+e[+-]?[0-9]+$")
LIST = re.compile(r"\[([^]]*)\]")
TUPLE = re.compile(r"\(([^)]+)\)")
STRING = re.compile(r'^"(.*)"$')
VAR_REF = re.compile(r"^\$([a-zA-Z][a-zA-Z0-9_]*)$")
OBJECT_REF = re.compile(
    r"^<([a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)*)>$")
CLASS_NAME = re.compile(
    r"^_*[a-zA-Z][a-zA-Z0-9_]*(\._*[a-zA-Z][a-zA-Z0-9_]*)+$")


CONSTANTS = {
    "False": False,
    "True": True,
    "None": None
}


def get_first_match(pattern: Pattern, string: str) -> str:
    """Return the first matching substring.

    Args:
        pattern: The pattern to find.
        string: The string to search.

    Returns:
        The first occurence of the pattern in the string.

    Raises:
        ValueError if the string does not match the pattern.
    """
    match = pattern.match(string)
    if match is None:
        raise ValueError("String '{}' does not match the pattern '{}'"
                         .format(string, pattern.pattern))
    return match.group(1)


# this is a function because of the parse_*
# functions which are not defined yet
def _keyval_parser_dict() -> Dict[Any, Callable]:
    return {
        INTEGER: lambda x, _: int(x),
        FLOAT: lambda x, _: float(x),
        STRING: _parse_string,
        VAR_REF: lambda x, vars_dict: vars_dict[get_first_match(VAR_REF, x)],
        CLASS_NAME: _parse_class_name,
        OBJECT_REF: lambda x, _: ObjectRef(get_first_match(OBJECT_REF, x)),
        LIST: _parse_list,
        TUPLE: _parse_tuple
    }


class VarsDict(OrderedDict, Dict[str, Any]):

    def __missing__(self, key):
        """Try to fetch and parse the variable value from `os.environ`."""
        if key in os.environ:
            try:
                value = _parse_value(os.environ[key], self)
            except ParseError:
                # If we cannot parse it, use it as a string.
                value = os.environ[key]
            log("Variable {}={!r} taken from the environment."
                .format(key, value))
            return value

        raise ParseError("Undefined variable: {}".format(key))


def _split_on_commas(string: str) -> List[str]:
    """Split a bracketed string on commas.

    The commas inside brackets are preserved.
    """

    items = []
    char_buffer = []  # type: List[str]
    openings = []  # type: List[str]

    for i, char in enumerate(string):
        if char == "," and not openings:
            if char_buffer:
                items.append("".join(char_buffer))
            char_buffer = []
            continue
        elif char == " " and not char_buffer:
            continue
        elif char in ("(", "["):
            openings.append(char)
        elif char == ")":
            if openings.pop() != "(":
                raise ParseError("Invalid bracket end ')', col {}.".format(i))
        elif char == "]":
            if openings.pop() != "[":
                raise ParseError("Invalid bracket end ']', col {}.".format(i))
        char_buffer.append(char)

    if char_buffer:
        items.append("".join(char_buffer))
    return items


def _parse_string(string: str, vars_dict: VarsDict) -> str:
    return get_first_match(STRING, string).format_map(vars_dict)


def _parse_list(string: str, vars_dict: VarsDict) -> List[Any]:
    """Parse the string recursively as a list."""

    matched_content = get_first_match(LIST, string)
    if not matched_content:
        return []

    items = _split_on_commas(matched_content)
    values = [_parse_value(val, vars_dict) for val in items]

    return values


def _parse_tuple(string: str, vars_dict: VarsDict) -> Tuple[Any, ...]:
    """Parse the string recursively as a tuple."""

    items = _split_on_commas(get_first_match(TUPLE, string))
    values = [_parse_value(val, vars_dict) for val in items]

    return tuple(values)


def _parse_class_name(string: str, vars_dict: VarsDict) -> ClassSymbol:
    """Parse the string as a module or class name."""
    del vars_dict
    return ClassSymbol(string)


def _parse_value(string: str, vars_dict: VarsDict) -> Any:
    """Parse the value recursively according to the Nerual Monkey grammar.

    Arguments:
        string: the string to be parsed
        vars_dict: a dictionary of variables for substitution
    """
    string = string.strip()

    if string in CONSTANTS:
        return CONSTANTS[string]

    for matcher, parser in _keyval_parser_dict().items():
        if matcher.match(string) is not None:
            return parser(string, vars_dict)

    raise ParseError("Cannot parse value: '{}'.".format(string))


def _parse_ini(config_file: Iterable[str],
               filename: str = "") -> Dict[str, Any]:
    """Parse an INI file into a dictionary."""

    line_numbers = (line.strip() + " " + str(i + 1)
                    if line.strip() else ""
                    for i, line in
                    enumerate(config_file))
    config = configparser.ConfigParser()
    config.read_file(line_numbers, source=filename)

    new_config = OrderedDict()  # type: Dict[str, Any]
    for section in config.sections():
        new_config[section] = OrderedDict()

        for key in config[section]:
            match = LINE_NUM.match(config[section][key])
            assert match is not None

            new_config[section][key] = match.group(2), match.group(1)

    return new_config


def _apply_change(config_dict: Dict[str, Any], setting: str) -> None:
    if "=" not in setting:
        raise ParseError("Invalid setting '{}'".format(setting))
    key, value = (s.strip() for s in setting.split("=", maxsplit=1))

    if "." in key:
        section, option = key.split(".", maxsplit=1)
    else:
        section = "main"
        option = key

    if section not in config_dict:
        log("Creating new section '{}'".format(section))
        config_dict[section] = OrderedDict()

    config_dict[section][option] = -1, value  # no line number


def parse_file(config_file: Iterable[str],
               changes: Iterable[str] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse an INI file and creates all values."""

    parsed_dicts = OrderedDict()  # type: Dict[str, Any]

    config = _parse_ini(config_file)

    if changes is not None:
        for change in changes:
            _apply_change(config, change)

    vars_dict = VarsDict()
    vars_dict["TIME"] = time.strftime("%Y-%m-%d-%H-%M-%S")

    def parse_section(section: str, output_dict: Dict[str, Any]):
        for key, (lineno, value_string) in config[section].items():
            try:
                value = _parse_value(value_string, vars_dict)
            except ParseError as exc:
                exc.set_line(lineno)
                raise

            output_dict[key] = value

    if "vars" in config:
        parse_section("vars", vars_dict)

    for section in config:
        if section != "vars":
            parsed_dicts[section] = OrderedDict()
            parse_section(section, parsed_dicts[section])

    # also return the unparsed config dict; need to remove line numbers
    raw_config = OrderedDict([
        (name, OrderedDict([(key, val) for key, (_, val) in section.items()]))
        for name, section in config.items()])

    return raw_config, parsed_dicts


def write_file(config_dict: Dict[str, Any], config_file: IO[str]) -> None:
    config = configparser.ConfigParser()
    config.read_dict(config_dict)
    config.write(config_file, space_around_delimiters=False)
