
"""
This module servers as a library of API checks used as assertions during
constructing the computational graph.
"""

# tests: lint

from neuralmonkey.logging import log, debug

class CheckingException(Exception):
    pass

def check_dataset_and_coders(dataset, runners):
    #pylint: disable=protected-access

    data_list = []

    for runner in runners:
        for c in runner.all_coders:
            if hasattr(c, "data_id"):
                data_list.append((c.data_id, c))
            elif hasattr(c, "data_ids"):
                data_list.extend([(d, c) for d in c.data_ids])
            else:
                log(("Warning: Coder: {} does not have"
                     "a data attribute").format(c))

    debug("Found series: {}".format(str(data_list)), "checking")
    missing = []

    for (serie, coder) in data_list:
        if not dataset.has_series(serie):
            log("dataset {} does not have serie {}".format(dataset.name, serie))
            missing.append((coder, serie))

    if len(missing) > 0:
        formated = ["{} ({}, {}.{})" .format(name, cod.name,
                                             cod.__class__.__module__,
                                             cod.__class__.__name__)
                    for name, cod in missing]

        raise CheckingException("Dataset '{}' is mising series {}:"
                                .format(dataset.name, ", ".join(formated)))

def missing_attributes(obj, attributes):
    return [attr for attr in attributes is not hasattr(obj, attributes)]


def type_to_str(type_obj):
    return "{}.{}".format(type_obj.__module__, type_obj.__name__)


def assert_type(obj, name, value, expected_type, can_be_none=False):
    if value is None and can_be_none:
        return
    if not isinstance(value, expected_type):
        caller_type_str = type_to_str(type(obj))
        exptected_str = type_to_str(expected_type)
        real_type_str = type_to_str(type(value))
        raise CheckingException(
            'Value of "{}" in "{}"should be "{}" but was "{}" {}'.format(
                name, caller_type_str, exptected_str, real_type_str, value))
