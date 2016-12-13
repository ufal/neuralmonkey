"""
Module that contains exceptions handled in config parsing and loading
"""
# tests: lint, mypy

import traceback


class IniError(Exception):
    """ Exception caused by error in INI file syntax """

    def __init__(self, line, message, original_exc=None):
        """ Creates an instance of the exception.

        Arguments:
            line: Line number on which the error occured
            message: A string describing the nature of the error
            original_exc (optional): An exception that caused this exception
                                   to be thrown
        """
        super().__init__()
        self.line = line
        self.message = message
        self.original_exc = original_exc

    def __str__(self):
        """ Converts this exception to string """

        msg = "Error on line {}: {}".format(self.line, self.message)
        if self.original_exc is not None:
            trc = "".join(traceback.format_list(traceback.extract_tb(
                self.original_exc.__traceback__)))
            msg += "\nTraceback:{}".format(trc)
        return msg


class ConfigInvalidValueException(Exception):

    def __init__(self, value, message):
        """ Creates an instance of the exception

        Arguments:
            value: The invalid value
            message: String that describes the nature of the error
        """
        super().__init__()
        self.value = value
        self.message = message

    def __str__(self):
        """ Converts this exception to string """
        return "Error in configuration of {}: {}".format(self.value,
                                                         self.message)


class ConfigBuildException(Exception):
    """ Exception caused by error in loading the model """

    def __init__(self, object_name, original_exception):
        """ Creates an instance of the exception

        Arguments:
            object_name: The name of the object that has failed to build
            original_exception: The exception that caused the failure
        """
        super().__init__()
        self.object_name = object_name
        self.original_exception = original_exception

    def __str__(self):
        """ Converts this exception to string"""

        trc = "".join(traceback.format_list(traceback.extract_tb(
            self.original_exception.__traceback__)))
        return "Error while loading '{}': {}\nTraceback: {}".format(
            self.object_name, self.original_exception, trc)
