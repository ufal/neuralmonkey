"""Configuration Object Builder.

This module is responsible for instantiating objects
specified by the experiment configuration.
"""

import collections
import importlib
from argparse import Namespace
from inspect import signature, isclass, isfunction, Parameter
from typing import Any, Dict, Set, Tuple

from neuralmonkey.logging import debug, warn
from neuralmonkey.config.exceptions import (ConfigInvalidValueException,
                                            ConfigBuildException)


# pylint:disable=too-few-public-methods
class ClassSymbol:
    """Represents a class (or other callable) in configuration."""

    def __init__(self, string: str) -> None:
        self.clazz = string

    def create(self) -> Any:
        class_parts = self.clazz.split(".")

        class_name = class_parts[-1]

        simple_module_path = ".".join(class_parts[:-1])
        try:
            module = importlib.import_module(simple_module_path)
        except ImportError:
            try:
                if class_parts[0] == "tf":
                    # Due to the architecture of TensorFlow, it must be
                    # imported this way.
                    tensorflow = importlib.import_module("tensorflow")
                    module = tensorflow
                    for i in range(1, len(class_parts) - 1):
                        module = getattr(module, class_parts[i])
                else:
                    module_name = ".".join(["neuralmonkey"] + class_parts[:-1])
                    module = importlib.import_module(module_name)
            except ImportError as exc:
                # if the problem is really importing the module
                if exc.name == module_name:  # type: ignore
                    raise Exception(
                        "Cannot import module {}.".format(module_name))
                else:
                    raise

        try:
            clazz = getattr(module, class_name)
        except AttributeError as exc:
            raise Exception(("Interpretation '{}' as type name, class '{}' "
                             "does not exist. Did you mean file './{}'? \n{}")
                            .format(self.clazz, class_name, self.clazz, exc))
        return clazz


class ObjectRef:
    """Represents a named object or its attribute in configuration."""

    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.name, *self.attr_chain = expression.split(".")
        self._obj = None

    def bind(self, value: Any):
        self._obj = value

    @property
    def target(self) -> Any:
        value = self._obj
        for attr in self.attr_chain:
            value = getattr(value, attr)
        return value


# pylint: disable=too-many-return-statements
def build_object(value: str,
                 all_dicts: Dict[str, Any],
                 existing_objects: Dict[str, Any],
                 depth: int) -> Any:
    """Build an object from config dictionary of its arguments.

    Works recursively.

    Arguments:
        value: Value that should be resolved (either a literal value or
               a config section name)
        all_dicts: Configuration dictionaries used to find configuration
                   of unconstructed objects.
        existing_objects: Dictionary of already constructed objects.
        depth: The current depth of recursion. Used to prevent an infinite
        recursion.
    """
    # TODO detect infinite recursion by other means than depth argument
    # TODO as soon as config is run from an entrypoint, remove the
    # ignore_names feature
    if depth > 20:
        raise AssertionError("Config recursion should not be deeper that 20.")

    debug("Building value on depth {}: {}".format(depth, value), "configBuild")

    # if isinstance(value, str) and value in ignore_names:
    # TODO zapisovani do argumentu
    #   existing_objects[value] = None

    if isinstance(value, tuple):
        return tuple(build_object(val, all_dicts, existing_objects, depth + 1)
                     for val in value)
    if (isinstance(value, collections.Iterable)
            and not isinstance(value, str)):
        return [build_object(val, all_dicts, existing_objects, depth + 1)
                for val in value]

    if isinstance(value, ObjectRef):
        if value.name in existing_objects:
            debug("Skipping already initialized object: {}".format(value.name),
                  "configBuild")
        else:
            existing_objects[value.name] = instantiate_class(
                value.name, all_dicts, existing_objects, depth)
        value.bind(existing_objects[value.name])
        return value.target

    if isinstance(value, ClassSymbol):
        return value.create()

    return value


def instantiate_class(name: str,
                      all_dicts: Dict[str, Any],
                      existing_objects: Dict[str, Any],
                      depth: int) -> Any:
    """Instantiate a class from the configuration.

    Arguments: see help(build_object)
    """
    if name not in all_dicts:
        debug(str(all_dicts), "configBuild")
        raise ConfigInvalidValueException(name, "Undefined object")
    this_dict = all_dicts[name]

    if "class" not in this_dict:
        raise ConfigInvalidValueException(name, "Undefined object type")
    clazz = this_dict["class"].create()

    if not isclass(clazz) and not isfunction(clazz):
        raise ConfigInvalidValueException(
            name, "Cannot instantiate object with '{}'".format(clazz))

    # prepare the arguments for the constructor
    arguments = dict()

    for key, value in this_dict.items():
        if key == "class":
            continue

        arguments[key] = build_object(value, all_dicts, existing_objects,
                                      depth + 1)

    # get a signature of the constructing function
    construct_sig = signature(clazz)

    # if a signature contains a "name" attribute which is not in arguments,
    # replace it with the name of the section
    if "name" in construct_sig.parameters and "name" not in arguments:
        annotation = construct_sig.parameters["name"].annotation

        if annotation == Parameter.empty:
            debug("No type annotation for the 'name' parameter in "
                  "class/function {}. Default value will not be used."
                  .format(this_dict["class"].clazz), "configBuild")
        elif annotation != str:
            debug("Type annotation for the 'name' parameter in class/function "
                  "{} is not 'str'. Default value will not be used."
                  .format(this_dict["class"].clazz), "configBuild")
            debug("Annotation is {}".format(str(annotation)))
        else:
            debug("Using default 'name' for object {}"
                  .format(this_dict["class"].clazz), "configBuild")
            arguments["name"] = name

    try:
        # try to bound the arguments to the signature
        bounded_params = construct_sig.bind(**arguments)
    except TypeError as exc:
        raise ConfigBuildException(clazz, exc)

    debug("Instantiating class {} with arguments {}".format(clazz, arguments),
          "configBuild")

    # call the function with the arguments
    # NOTE: any exception thrown from the body of the constructor is
    # not worth catching here
    obj = clazz(*bounded_params.args, **bounded_params.kwargs)

    debug("Class {} initialized into object {}".format(clazz, obj),
          "configBuild")

    return obj


def build_config(config_dicts: Dict[str, Any],
                 ignore_names: Set[str],
                 warn_unused: bool = False) -> Tuple[Dict[str, Any],
                                                     Dict[str, Any]]:
    """Build the model from the configuration.

    Arguments:
        config_dicts: The parsed configuration file
        ignore_names: A set of names that should be ignored during the loading.
        warn_unused: Emit a warning if there are unused sections.

    Returns:
        A tuple containing a dictionary corresponding to the main section and
        a dictionary mapping section names to objects.
    """
    if "main" not in config_dicts:
        raise Exception("Configuration does not contain the main block.")

    existing_objects = collections.OrderedDict()  # type: Dict[str, Any]

    main_config = config_dicts["main"]
    existing_objects["main"] = Namespace(**main_config)

    configuration = collections.OrderedDict()  # type: Dict[str, Any]
    # TODO ensure tf_manager goes last in a better way
    for key, value in sorted(main_config.items(),
                             key=lambda t: t[0] if t[0] != "tf_manager"
                             else "zzz"):
        if key not in ignore_names:
            try:
                configuration[key] = build_object(
                    value, config_dicts, existing_objects, 0)
            except Exception as exc:
                raise ConfigBuildException(key, exc) from None

    if warn_unused:
        existing_names = set(existing_objects.keys()) | {"main"}
        unused = config_dicts.keys() - existing_names
        if unused:
            warn("Configuration contains unused sections: "
                 + str(unused) + ".")

    return configuration, existing_objects
