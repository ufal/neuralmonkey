#tests: lint

import traceback
from argparse import Namespace

from neuralmonkey.logging import log
from neuralmonkey.config.config_loader import load_config_file, build_config

class Configuration(object):
    """
    Loads the configuration file in an analogical way the python's
    argparse.ArgumentParser works.
    """

    def __init__(self):
        self.data_types = {}
        self.defaults = {}
        self.conditions = {}
        self.ignored = set()
        self.config_dict = {}

    #pylint: disable=too-many-arguments
    def add_argument(self, name, arg_type=object, required=False, default=None,
                     cond=None):

        if name in self.data_types:
            raise Exception("Data filed defined multiple times.")
        self.data_types[name] = arg_type
        if not required:
            self.defaults[name] = default
        if cond is not None:
            self.conditions[name] = cond

    def ignore_argument(self, name):
        self.ignored.add(name)

    def load_file(self, path):
        log("Loading INI file: '{}'".format(path), color='blue')

        try:
            arguments = Namespace()
            self.config_dict = load_config_file(path)

            for name, value in self.config_dict.items():
                if name in self.conditions and not self.conditions[name](value):
                    cond_code = self.conditions[name].__code__
                    cond_filename = cond_code.co_filename
                    cond_line_number = cond_code.co_firstlineno
                    raise Exception(
                        "Value of field '{}' does not satisfy "
                        "condition defined at {}:{}."
                        .format(name, cond_filename, cond_line_number))

                setattr(arguments, name, value)
                #arguments.__dict__[name] = value

            for name, value in self.defaults.items():
                if name not in arguments.__dict__:
                    arguments.__dict__[name] = value
            log("INI file loaded.", color='blue')
        #pylint: disable=broad-except
        except Exception as exc:
            log("Failed to load INI file: {}".format(exc), color='red')
            traceback.print_exc()
            exit(1)

        return arguments

    def build_model(self):
        log("Building model based on the config.")
        try:
            model = build_config(self.config_dict, self.ignored)
        except Exception as exc:
            log("Failed to build model: {}".format(exc), color='red')
            traceback.print_exc()
            exit(1)
        log("Model built.")
        self._check_loaded_conf(model)
        model_n = Namespace()
        for name, value in model.items():
            if name in self.conditions and not self.conditions[name](value):
                cond_code = self.conditions[name].__code__
                cond_filename = cond_code.co_filename
                cond_line_number = cond_code.co_firstlineno
                raise Exception(
                        "Value of field '{}' does not satisfy "
                        "condition defined at {}:{}."
                        .format(name, cond_filename, cond_line_number))

            setattr(model_n, name, value)
            #arguments.__dict__[name] = value

        for name, value in self.defaults.items():
            if name not in model_n.__dict__:
                model_n.__dict__[name] = value

        return model_n

    def _check_loaded_conf(self, config_dict):
        """ Checks whether there are unexpected or missing fields """
        expected_fields = set(self.data_types.keys())

        expected_missing = []
        for name in expected_fields:
            if name not in self.defaults:
                expected_missing.append(name)
        if expected_missing:
            raise Exception("Missing mandatory fields: {}"
                            .format(", ".join(expected_missing)))
        unexpected = []
        for name in config_dict:
            if name not in expected_fields:
                unexpected.append(name)
        if unexpected:
            raise Exception("Unexpected fields: {}"
                            .format(", ".join(unexpected)))
