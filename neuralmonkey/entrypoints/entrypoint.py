#tests: lint

import tensorflow as tf

from neuralmonkey.logging import log

class EntryPoint(object):

    def __init__(self, tfconfig):
        self.tfconfig = tfconfig


    def execute(self, *args):
        """ Execute the entry point

        Arguments:
            args: additional arguments from command line
        """
        raise NotImplementedError("Cannot call directly - abstract method")


    def create_session(self, random_init=True):
        log("Initializing the TensorFlow session.")
        session = tf.Session(config=self.tfconfig)

        if random_init:
            session.run(tf.initialize_all_variables())
            log("Random variable initialization done.")


    def create_session_from_variables(self, variables_file):
        session = self.create_session()

        log("Loading variables from file {}.".format(variables_file))
        saver = tf.train.Saver()
        saver.restore(session, variables_file)
        log("Variables restored.")

        return session
