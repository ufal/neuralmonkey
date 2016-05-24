from argparse import Namespace

from config_loader import load_config_file

class Configuration(object):
    """
    Loads the configuration file in an analogical way the python's
    argparse.ArgumentParser words.
    """

    def __init__(self):
        self.data_types = {}
        self.defaults = {}
        self.conditions = {}

    def add_argument(self, name, arg_type=object, required=False, default=None, cond=None):
        if name in self.data_types:
            raise Exception("Data filed defined multiple times.")
        self.data_types[name] = arg_type
        if not required:
            self.defaults[name] = default
        if cond is not None:
            self.conditions[name] = cond

    def load_file(self, path):
        arguments = Namespace()#type('', (object,), {})()

        config_dict = load_config_file(path)

        self._check_loaded_conf(config_dict)

        for name, value in config_dict.iteritems():
            if name in self.conditions and not self.conditions[name](value):
                cond_code = self.conditions[name].func_code
                cond_filename = cond_code.co_filename
                cond_line_number = cond_code.co_firstlineno
                raise Exception(
                    "Value of field \"{}\" does not satisfy condition defined at {}:{}."\
                        .format(name, cond_filename, cond_line_number))

            setattr(arguments, name, value)
            #arguments.__dict__[name] = value

        for name, value in self.defaults.iteritems():
            if name not in arguments.__dict__:
                arguments.__dict__[name] = value

        return arguments

    def _check_loaded_conf(self, config_dict):
        """ Checks whether there are unexpected or missing fields """
        expected_fields = set(self.data_types.keys())

        expected_missing = []
        for name in expected_fields:
            if name not in self.defaults:
                expected_missing.append(name)
        if expected_missing:
            raise Exception("Missing mandatory fileds: {}".format(", ".join(expected_missing)))

        unexpected = []
        for name in config_dict:
            if name not in expected_fields:
                unexpected.append(name)
        if unexpected:
            raise Exception("Unexpected fields: {}".format(", ".join(unexpected)))



