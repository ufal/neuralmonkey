from typing import Iterator, List, Any, Callable

from neuralmonkey.logging import log

# pylint: disable=invalid-name
# Writer: function that gets file and the data
Writer = Callable[[str, Any], None]
# pylint: enable=invalid-name


def tokenized_text_writer(encoding: str = "utf-8") -> Writer:

    def writer(path: str, data: Iterator[List[str]]) -> None:
        with open(path, "w", encoding=encoding) as f_out:
            for sentence in data:
                f_out.write(" ".join(sentence) + "\n")

        import pudb;pu.db
        log("Result saved as tokenized text in '{}'".format(path))

    return writer


def text_writer(encoding: str = "utf-8") -> Writer:

    def writer(path: str, data: Iterator[Any]) -> None:
        with open(path, "w", encoding=encoding) as f_out:
            for sentence in data:
                f_out.write(str(sentence) + "\n")
        log("Result saved as plain text in '{}'".format(path))

    return writer


# pylint: disable=invalid-name
UtfPlainTextWriter = tokenized_text_writer()
