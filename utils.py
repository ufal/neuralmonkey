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
#    for arg in vars(args):
#        value = getattr(args, arg)
#        if isinstance(value, file):
#            value_str = value.name
#        else:
#            value_str = str(value)
#        dots_count = 78 - len(arg) - len(value_str)
#        print "{} {} {}".format(arg, "".join(['.' for _ in range(dots_count)]), value_str)
#    print ""
#
#    os.system("echo last commit: `git log -1 --format=%H`")
#    os.system("git --no-pager diff --color=always")
#    print ""
