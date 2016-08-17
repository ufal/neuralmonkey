#tests: lint

import time
import codecs
from termcolor import colored

class Logging(object):

    log_file = None # type: file

    ## 'all' and 'none' are special symbols,
    ## others are filtered according the labels
    debug_enabled = ['all']


    @staticmethod
    def set_log_file(path):
        """Sets up the file where the logging will be done."""
        Logging.log_file = codecs.open(path, 'w', 'utf-8', buffering=0)

    @staticmethod
    def log_print(text):
        """Prints a string both to console and
        a log file is it is defined.
        """
        if Logging.log_file is not None:
            if not isinstance(text, str):
                text = str(text)
            Logging.log_file.write(text+"\n")
            Logging.log_file.flush()

        print(text)

    @staticmethod
    def log(message, color='yellow'):
        """Logs message with a colored timestamp."""
        log_print("{}: {}".format(colored(
            time.strftime("%Y-%m-%d %H:%M:%S"), color), message))

    @staticmethod
    def print_header(title):
        """Prints the title of the experiment and
        the set of arguments it uses.
        """
        log_print(colored("".join("=" for _ in range(80)), 'green'))
        log_print(colored(title.upper(), 'green'))
        log_print(colored("".join("=" for _ in range(80)), 'green'))
        log_print("Launched at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

        log_print("")

    @staticmethod
    def debug(message, label=None):
        if 'none' in Logging.debug_enabled:
            return

        if (label not in Logging.debug_enabled and
                'all' not in Logging.debug_enabled):
            return

        if label:
            prefix = "DEBUG ({}):".format(label)
        else:
            prefix = "DEBUG:"

        log_print("{}{}".format(colored(prefix, color="cyan"), message))


# pylint: disable=invalid-name
# we want these helper functions to have this exact name
log = Logging.log
log_print = Logging.log_print
debug = Logging.debug
