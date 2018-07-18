"""Module that contains exceptions handled in config parsing and loading."""

import traceback
from typing import Any


class ParseError(Exception):
    """Parsing exception caused by a syntax error in INI file."""

    def __init__(self, message: str, line: int = None) -> None:
        super().__init__()
        self.message = message
        self.line = line

    def set_line(self, line: int) -> None:
        self.line = line

    def __str__(self) -> str:
        """Convert this exception to string."""
        if self.line is not None:
            return "INI error on line {}: {}".format(self.line, self.message)

        return "INI parsing error: {}".format(self.message)


class ConfigInvalidValueException(Exception):

    def __init__(self, value: Any, message: str) -> None:
        """Create an instance of the exception.

        Arguments:
            value: The invalid value
            message: String that describes the nature of the error
        """
        super().__init__()
        self.value = value
        self.message = message

    def __str__(self) -> str:
        """Convert this exception to string."""
        return "Error in configuration of {}: {}".format(
            self.value, self.message)


class ConfigBuildException(Exception):
    """Exception caused by error in loading the model."""

    def __init__(self, object_name: str,
                 original_exception: Exception) -> None:
        """Create an instance of the exception.

        Arguments:
            object_name: The name of the object that has failed to build
            original_exception: The exception that caused the failure
        """
        super().__init__()
        self.object_name = object_name
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Convert this exception to string."""

        trc = "".join(traceback.format_list(traceback.extract_tb(
            self.original_exception.__traceback__)))
        return "Error while loading '{}': {}\nTraceback: {}".format(
            self.object_name, self.original_exception, trc)
