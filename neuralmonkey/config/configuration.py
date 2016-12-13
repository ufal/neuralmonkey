# tests: lint, mypy

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

    # pylint: disable=too-many-arguments
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

    def make_namespace(self, d_obj):
        n_space = Namespace()
        for name, value in d_obj.items():
            if name in self.conditions and not self.conditions[name](value):
                cond_code = self.conditions[name].__code__
                cond_filename = cond_code.co_filename
                cond_line_number = cond_code.co_firstlineno
                raise Exception(
                    "Value of field '{}' does not satisfy "
                    "condition defined at {}:{}."
                    .format(name, cond_filename, cond_line_number))

            setattr(n_space, name, value)

        for name, value in self.defaults.items():
            if name not in n_space.__dict__:
                n_space.__dict__[name] = value
        return n_space

    def load_file(self, path):
        log("Loading INI file: '{}'".format(path), color='blue')

        try:
            self.config_dict = load_config_file(path)
            arguments = self.make_namespace(self.config_dict)
            log("INI file loaded.", color='blue')
        # pylint: disable=broad-except
        except Exception as exc:
            log("Failed to load INI file: {}".format(exc), color='red')
            traceback.print_exc()
            exit(1)

        return arguments

    def build_model(self):
        log("Building model based on the config.")
        try:
            model = build_config(self.config_dict, self.ignored)
        # pylint: disable=broad-except
        except Exception as exc:
            log("Failed to build model: {}".format(exc), color='red')
            traceback.print_exc()
            exit(1)
        log("Model built.")
        self._check_loaded_conf(model)
        return self.make_namespace(model)

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
