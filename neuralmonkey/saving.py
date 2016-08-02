# tests: lint, mypy

import os
import tensorflow as tf

class Saving(object):


    def __init__(vars_prefix, link_best_vars, max_to_keep, minimize=False):
        self.vars_prefix = vars_prefix
        self.link_best_vars = link_best_vars
        self.max_to_keep = max_to_keep

        if max_to_keep < 1:
            raise Exception('save_n_best_vars must be greater than zero')

        if max_to_keep == 1:
            self.variables_files = [vars_prefix]
        elif max_to_keep > 1:
            self.variables_files = ['{}.{}'.format(vars_prefix, i)
                                    for i in range(max_to_keep)]

        if os.path.islink(link_best_vars):
            # if overwriting output dir
            os.unlink(link_best_vars)
        os.symlink(os.path.basename(self.variables_files[0]), link_best_vars)

        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        log("Saving initialization done.")




    def load_best_parameters(self, session):
        if os.path.islink(self.link_best_vars):
            self.saver.restore(session, self.link_best_vars)

    def save_parameters(self, session, score):
