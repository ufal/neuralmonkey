"""
This module is responsible for instantiating objects
specified by the experiment configuration
"""
# tests: lint, mypy

import collections
from inspect import signature, isclass, isfunction

import neuralmonkey.config.parsing as parsing
from neuralmonkey.logging import log, debug
from neuralmonkey.config.exceptions import (ConfigInvalidValueException,
                                            ConfigBuildException)


# pylint: disable=too-many-return-statements
def build_object(value, all_dicts, existing_objects, depth):
    """Builds an object from config dictionary of its arguments.
    It works recursively.

    Arguments:
        value: Value that should be resolved (either a literal value or
               a config section name)
        all_dicts: Configuration dictionaries used to find configuration
                   of unconstructed objects.
        existing_objects: Dictionary of already constructed objects.
        ignore_names: Set of names that shoud be ignored.
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
    elif (isinstance(value, collections.Iterable) and
          not isinstance(value, str)):
        return [build_object(val, all_dicts, existing_objects, depth + 1)
                for val in value]

    if value in existing_objects:
        debug("Skipping already initialized value: {}".format(value),
              "configBuild")

        return existing_objects[value]

    if isinstance(value, str):
        # either a string or a reference to an object
        if not value.startswith("object:"):
            return value

        obj = instantiate_class(value[7:], all_dicts, existing_objects, depth)
        existing_objects[value] = obj
        return obj

    return value


def instantiate_class(name, all_dicts, existing_objects, depth):
    """ Instantiate a class from the configuration

    Arguments: see help(build_object)
    """
    if name not in all_dicts:
        debug(all_dicts, "configBuild")
        raise ConfigInvalidValueException(name, "Undefined object")
    this_dict = all_dicts[name]

    if 'class' not in this_dict:
        raise ConfigInvalidValueException(name, "Undefined object type")
    clazz = this_dict['class']

    if not isclass(clazz) and not isfunction(clazz):
        raise ConfigInvalidValueException(
            name, "Cannot instantiate object with '{}'".format(clazz))

    # prepare the arguments for the constructor
    arguments = dict()

    for key, value in this_dict.items():
        if key == 'class':
            continue

        arguments[key] = build_object(value, all_dicts, existing_objects,
                                      depth + 1)

    # get a signature of the constructing function
    construct_sig = signature(clazz)

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


def load_config_file(config_file):
    """ Loads the configuration

    Arguments:
        config_file: The configuration file
    """
    with open(config_file, 'r', encoding='utf-8') as file:
        config_dicts = parsing.parse_file(file)
    log("INI file is parsed.")
    return config_dicts


def build_config(config_dicts, ignore_names):
    """ Builds the model from the configuration

    Arguments:
        config_dicts: The parsed configuration file
        ignore_names: A set of names that should be ignored during the loading.
    """
    if "main" not in config_dicts:
        raise Exception("Configuration does not contain the main block.")

    existing_objects = dict()

    main_config = config_dicts['main']

    configuration = dict()
    # TODO ensure tf_manager goes last in a better way
    for key, value in sorted(main_config.items(),
                             key=lambda t: t[0] if t[0] != 'tf_manager'
                             else 'zzz'):
        if key not in ignore_names:
            try:
                configuration[key] = build_object(
                    value, config_dicts, existing_objects, 0)
            except Exception as exc:
                raise ConfigBuildException(key, exc) from None

    return configuration
