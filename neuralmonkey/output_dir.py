import os
from shutil import copyfile

import tensorflow as tf

from neuralmonkey.logging import Logging, log, log_print, debug
from neuralmonkey.saving import Saving

class OutputDirectory(object):

    def __init__(path, overwrite=False):
        self.path = path
        self.overwrite = overwrite

        ## TODO nicer exceptions

        if os.path.isdir(path) and os.path.exists(self.ini_file):
            if overwrite:
                log("Experiment directory '{}' exists, "
                    "overwriting enabled, proceeding."
                    .format(path))
            else:
                log("Experiment directory '{}' exists, "
                    "overwriting disabled."
                    .format(path), color="red")
                raise Exception("Cannot create output dir")

        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except Exception as exc:
                log("Failed to create experiment directory: {}. Exception: {}"
                    .format(path, exc), color="red")
                raise Exception("Cannot create output dir")

        self.cont_index = 0

        while (os.path.exists(self.log_file)
               or os.path.exists(self.ini_file)
               or os.path.exists(self.commit_file)
               or os.path.exists(self.diff_file)
               or os.path.exists(self.var_prefix)
               or os.path.exists("{}.0".format(self.var_prefix))):
            self.cont_index += 1

        copyfile(sys.argv[1], ini_file)
        Logging.set_log_file(log_file)
        Logging.print_header(args.name)

        os.system("git log -1 --format=%H > {}".format(git_commit_file))
        os.system("git --no-pager diff --color=always > {}"
                  .format(git_diff_file))





    def get_filename(filename):
        if self.cont_index == 0:
            return os.path.join(self.path, filename)
        else:
            return os.path.join(self.path,
                                "{}.cont-{}".format(filename, self.cont_index))



    @property
    def ini_file(self):
        return self.get_filename("experiment.ini")

    @property
    def log_file(self):
        return self.get_filename("experiment.log")

    @property
    def commit_file(self):
        return self.get_filename("git_commit")

    @property
    def diff_file(self):
        return self.get_filename("git_diff")

    @property
    def var_prefix(self):
        return self.get_filename("variables.data")

    @property
    def link_best_vars(self):
        return "{}.best".format(self.var_prefix)
