#tests: lint

import os
import sys
from shutil import copyfile

from neuralmonkey.logging import Logging, log
from neuralmonkey.exceptions import OutputDirectoryException

class OutputDirectory(object):
    """Helper class for manipulating file names inside the output directory"""

    def __init__(self, path, overwrite=False):
        """Creates an instance of output directory helper object.
        This constructor does not do any changes to the file system.
        The initialization of the directory is done by the populate method.

        Arguments:
            path: Path to the directory
            overwrite: If True, do not crash when the directory exists,
                       instead establish a non-conflicting postfix to all of
                       the created files inside the existing directory.
        """
        self.path = path
        self.overwrite = overwrite

        if os.path.isdir(path):
            if overwrite:
                log("Experiment directory '{}' exists, "
                    "overwriting enabled, proceeding."
                    .format(path))
            else:
                log("Experiment directory '{}' exists, "
                    "overwriting disabled."
                    .format(path), color="red")
                raise OutputDirectoryException("Cannot create output dir")

        self.cont_index = 0

        ## TODO use glob to check for the other variable files
        while (os.path.exists(self.log_file)
               or os.path.exists(self.ini_file)
               or os.path.exists(self.commit_file)
               or os.path.exists(self.diff_file)
               or os.path.exists(self.var_prefix)
               or os.path.exists(self.link_best_vars)
               or os.path.exists(self.get_variable_file(0))):
            self.cont_index += 1


    def populate(self, original_ini_file=sys.argv[2]):
        """Create the output directory on the file system and populates it
        with the experiment context files.

        Arguments:
            original_ini_file: The INI file to copy. This can be used if
                               sys.argv has changed or if you want to use
                               another file
        """
        if not os.path.isdir(self.path):
            try:
                os.mkdir(self.path)
            except Exception as exc:
                log("Failed to create experiment directory: {}. Exception: {}"
                    .format(self.path, exc), color="red")
                raise OutputDirectoryException("Cannot create output dir")

        copyfile(original_ini_file, self.ini_file)
        Logging.set_log_file(self.log_file)

        os.system("git log -1 --format=%H > {}".format(self.commit_file))
        os.system("git --no-pager diff --color=always > {}"
                  .format(self.diff_file))


    def _get_filename(self, filename):
        """Returns path to a file (non-conflicting)"""
        if self.cont_index == 0:
            return os.path.join(self.path, filename)
        else:
            return os.path.join(self.path,
                                "{}.cont-{}".format(filename, self.cont_index))


    def get_variable_file(self, index):
        """Returns file name for a variable file, given index

        Arguments:
            index: The index of the variable file.
        """
        return "{}.{}".format(self.var_prefix, index)


    @property
    def ini_file(self):
        """File where the configuration of the experiment is stored"""
        return self._get_filename("experiment.ini")


    @property
    def log_file(self):
        """File with logs"""
        return self._get_filename("experiment.log")


    @property
    def commit_file(self):
        """File with hash to the commit in which the experiment was run."""
        return self._get_filename("git_commit")


    @property
    def diff_file(self):
        """This file contains the diff of working copy vs. the current HEAD"""
        return self._get_filename("git_diff")


    @property
    def var_prefix(self):
        """Prefix of saved variable files"""
        return self._get_filename("variables.data")


    @property
    def link_best_vars(self):
        """Link to the best variable file"""
        return "{}.best".format(self.var_prefix)
