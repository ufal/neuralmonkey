"""
This module servers as a library of API checks used as assertions during
constructing the computational graph.
"""

def check_dataset_and_coders(dataset, coders):
    missing = \
        [(cod.data_id, cod) for cod in coders if cod.data_id not in dataset.series]
    if missing:
        formated = ["{} ({}, {}.{})".format(name,
                                            cod.name,
                                            cod.__class__.__module__,
                                            cod.__class__.__name__) for name, cod in missing]
        raise Exception("Dataset \"{}\" is mising series {}:"\
                .format(dataset.name, ", ".join(formated)))


def missing_attributes(obj, attributes):
    return [attr for attr in attributes is not hasattr(obj, attributes)]
