import time
import sys
import os

# pylint: disable=unused-import
from typing import Any, List
# pylint: enable=unused-import

from termcolor import colored


class Logging:

    log_file = None  # type: Any

    # 'all' and 'none' are special symbols,
    # others are filtered according the labels
    debug_enabled_for = [
        os.environ.get("NEURALMONKEY_DEBUG_ENABLE", "none")]  # type: List[str]
    debug_disabled_for = [
        os.environ.get("NEURALMONKEY_DEBUG_DISABLE", "")]  # type: List[str]
    strict_mode = os.environ.get("NEURALMONKEY_STRICT", "")  # type: str

    @staticmethod
    def _get_time() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def set_log_file(path: str) -> None:
        """Set up the file where the logging will be done."""
        if Logging.log_file is not None and not Logging.log_file.closed:
            Logging.log_file.close()
        Logging.log_file = open(path, "w", encoding="utf-8", buffering=1)

    @staticmethod
    def log_print(text: str) -> None:
        """Print a string both to console and a log file is it is defined."""
        if Logging.log_file is not None:
            if not isinstance(text, str):
                text = str(text)
            Logging.log_file.write(text + "\n")
            Logging.log_file.flush()

        print(text, file=sys.stderr)

    @staticmethod
    def log(message: str, color: str = "yellow") -> None:
        """Log a message with a colored timestamp."""
        log_print("{}: {}".format(colored(
            Logging._get_time(), color), message))

    @staticmethod
    def notice(message: str) -> None:
        """Log a notice with a colored timestamp."""
        log_print("{}: {}".format(colored(
            Logging._get_time(), "red"), message))

    @staticmethod
    def warn(message: str) -> None:
        """Log a warning."""
        log_print(colored("{}: Warning! {}".format(
            Logging._get_time(), message), color="red"))
        if Logging.strict_mode:
            raise Exception(
                "Encountered a warning in strict mode: " + message)

    @staticmethod
    def print_header(title: str, path: str) -> None:
        """Print the title of the experiment and a set of arguments it uses."""
        log_print(colored("".join("=" for _ in range(80)), "green"))
        log_print(colored(title.upper(), "green"))
        log_print(colored("".join("=" for _ in range(80)), "green"))
        log_print("Launched at {}".format(Logging._get_time()))
        log_print("Experiment directory: {}".format(path))

        log_print("")

    @staticmethod
    def debug(message: str, label: str = None):
        if not debug_enabled(label):
            return

        if label:
            prefix = "{}: DEBUG ({}): ".format(Logging._get_time(), label)
        else:
            prefix = "{}: DEBUG: ".format(Logging._get_time())

        log_print("{}{}".format(colored(prefix, color="cyan"), message))

    @staticmethod
    def debug_enabled(label: str = None):
        if "none" in Logging.debug_enabled_for:
            return False

        if label is None:
            return True

        if (label in Logging.debug_disabled_for
                or ("all" not in Logging.debug_enabled_for
                    and label not in Logging.debug_enabled_for)):
            return False

        return True


# pylint: disable=invalid-name
# we want these helper functions to have this exact name
log = Logging.log
log_print = Logging.log_print
debug = Logging.debug
warn = Logging.warn
notice = Logging.notice
debug_enabled = Logging.debug_enabled
