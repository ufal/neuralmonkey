import os
import time
from termcolor import colored


def log(message, color='yellow'):
    print "{}: {}".format(colored(time.strftime("%Y-%m-%d %H:%M:%S"), color), message)


def print_header(title):
    """
    Prints the title of the experiment and the set of arguments it uses.
    """
    print colored("".join("=" for _ in range(80)), 'green')
    print colored(title.upper(), 'green')
    print colored("".join("=" for _ in range(80)), 'green')
    print "Launched at {}".format(time.strftime("%Y-%m-%d %H:%M:%S"))

    print ""

    # TODO print the complete configuration (unroll objects to the first level)
