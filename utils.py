import time
import codecs
from termcolor import colored

# tests: mypy

# TODO make this a class!

log_file = None # type: file

def set_log_file(path):
    """ Sets up the file where the logging will be done. """
    log_file = codecs.open(path, 'w', 'utf-8', buffering=0)


def log_print(text):
    """ Prints a string both to console and a log file is it si defined. """
    if log_file:
        if not isinstance(text, basestring):
            text = str(text)
        log_file.write(text+"\n")
        log_file.flush()

    encoded = text.encode('utf8', 'replace')
    print encoded


def log(message, color='yellow'):
    """ Logs message with a colored timestamp. """
    log_print("{}: {}".format(colored(time.strftime("%Y-%m-%d %H:%M:%S"), color), message))


def print_header(title):
    """
    Prints the title of the experiment and the set of arguments it uses.
    """
    log_print(colored("".join("=" for _ in range(80)), 'green'))
    log_print(colored(title.upper(), 'green'))
    log_print(colored("".join("=" for _ in range(80)), 'green'))
    log_print("Launched at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    log_print("")
