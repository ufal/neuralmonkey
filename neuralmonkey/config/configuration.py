import traceback
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Callable, List, Optional

from neuralmonkey.logging import log
from neuralmonkey.config.builder import build_config
from neuralmonkey.config.parsing import parse_file, write_file


class Configuration:
    """Configuration loader.

    Loads the configuration file in an analogical way the python's
    argparse.ArgumentParser works.
    """

    def __init__(self):
        self.names = []
        self.defaults = {}
        self.conditions = {}
        self.ignored = set()
        self.raw_config = OrderedDict()
        self.config_dict = OrderedDict()
        self.objects = None
        self.args = None
        self.model = None

    # pylint: disable=too-many-arguments
    def add_argument(self,
                     name: str,
                     required: bool = False,
                     default: Any = None,
                     cond: Callable[[Any], bool] = None) -> None:

        if name in self.names:
            raise Exception("Data filed defined multiple times.")
        self.names.append(name)

        if not required:
            self.defaults[name] = default
        if cond is not None:
            self.conditions[name] = cond

    def ignore_argument(self, name: str) -> None:
        self.ignored.add(name)

    def make_namespace(self, d_obj) -> Namespace:
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

    def load_file(self, path: str,
                  changes: Optional[List[str]] = None) -> None:
        log("Loading INI file: '{}'".format(path), color="blue")

        try:
            with open(path, "r", encoding="utf-8") as file:
                raw_config, config_dict = parse_file(file, changes)
            log("INI file is parsed.")

            self.raw_config.update(raw_config)
            self.config_dict.update(config_dict)
        # pylint: disable=broad-except
        except Exception as exc:
            log("Failed to load INI file: {}".format(exc), color="red")
            traceback.print_exc()
            exit(1)

        if "main" in self.config_dict:
            self.args = self.make_namespace(self.config_dict["main"])

    def build_model(self, warn_unused=False) -> None:
        log("Building model based on the config.")
        self._check_loaded_conf()
        try:
            model, self.objects = build_config(self.config_dict, self.ignored,
                                               warn_unused)
        # pylint: disable=broad-except
        except Exception as exc:
            log("Failed to build model: {}".format(exc), color="red")
            traceback.print_exc()
            exit(1)
        log("Model built.")
        self.model = self.make_namespace(model)

    def _check_loaded_conf(self) -> None:
        """Check whether there are unexpected or missing fields."""
        expected_missing = []
        for name in self.names:
            if name not in self.args.__dict__:
                expected_missing.append(name)
        if expected_missing:
            raise Exception("Missing mandatory fields: {}"
                            .format(", ".join(expected_missing)))
        unexpected = []
        for name in self.config_dict["main"]:
            if name not in self.names and name not in self.ignored:
                unexpected.append(name)
        if unexpected:
            raise Exception("Unexpected fields: {}"
                            .format(", ".join(unexpected)))

    def save_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            write_file(self.raw_config, file)
