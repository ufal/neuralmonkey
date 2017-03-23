import time
import codecs
import sys
import os

# pylint: disable=unused-import
from typing import Any, Optional
from termcolor import colored


class Logging(object):

    log_file = None  # type: Optional[Any]

    # 'all' and 'none' are special symbols,
    # others are filtered according the labels
    debug_enabled = [
        os.environ.get("NEURALMONKEY_DEBUG_ENABLE", "none")]  # type: List[str]
    debug_disabled = [
        os.environ.get("NEURALMONKEY_DEBUG_DISABLE", "")]  # type: List[str]
    strict_mode = os.environ.get("NEURALMONKEY_STRICT")  # type: str

    @staticmethod
    def _get_time() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def set_log_file(path: str) -> None:
        """Sets up the file where the logging will be done."""
        Logging.log_file = codecs.open(path, 'w', 'utf-8', buffering=0)

    @staticmethod
    def log_print(text: str) -> None:
        """Prints a string both to console and
        a log file is it is defined.
        """
        if Logging.log_file is not None:
            if not isinstance(text, str):
                text = str(text)
            Logging.log_file.write(text + "\n")
            Logging.log_file.flush()

        print(text, file=sys.stderr)

    @staticmethod
    def log(message: str, color: str='yellow') -> None:
        """Logs message with a colored timestamp."""
        log_print("{}: {}".format(colored(
            Logging._get_time(), color), message))

    @staticmethod
    def warn(message: str) -> None:
        """Logs a warning."""
        log_print(colored("{}: Warning! {}".format(
            Logging._get_time(), message), color='red'))
        if Logging.strict_mode:
            raise Exception(
                "Encountered a warning in strict mode: " + message)

    @staticmethod
    def print_header(title: str) -> None:
        """Prints the title of the experiment and
        the set of arguments it uses.
        """
        log_print(colored("".join("=" for _ in range(80)), 'green'))
        log_print(colored(title.upper(), 'green'))
        log_print(colored("".join("=" for _ in range(80)), 'green'))
        log_print("Launched at {}".format(Logging._get_time()))

        log_print("")

    @staticmethod
    def debug(message: str, label: Optional[str]=None):
        if 'none' in Logging.debug_enabled:
            return

        if (label not in Logging.debug_enabled and
                'all' not in Logging.debug_enabled):
            return

        if label in Logging.debug_disabled:
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
warn = Logging.warn
