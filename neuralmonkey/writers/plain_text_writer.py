from typing import Iterator, List, Any, Callable

from neuralmonkey.logging import log
from neuralmonkey.readers.plain_text_reader import _ALNUM_CHARSET

# pylint: disable=invalid-name
# Writer: function that gets file and the data
Writer = Callable[[str, Any], None]
# pylint: enable=invalid-name


def text_writer(encoding: str = "utf-8") -> Writer:

    def writer(path: str, data: Iterator[Any]) -> None:
        with open(path, "w", encoding=encoding) as f_out:
            for sentence in data:
                f_out.write(str(sentence) + "\n")
        log("Result saved as plain text in '{}'".format(path))

    return writer


def tokenized_text_writer(encoding: str = "utf-8") -> Writer:

    def writer(path: str, data: Iterator[List[str]]) -> None:
        writer = text_writer(encoding)
        writer(path, (" ".join(s) for s in data))

    return writer


def t2t_tokenized_text_writer(encoding: str = "utf-8") -> Writer:
    """Get a writer that is reversed to the readers.t2t_tokenized_text_reader.

    Code in the detokenize method is inspired by tensor2tensor tokenizer:
    https://github.com/tensorflow/tensor2tensor/blob/v1.5.5/tensor2tensor/data_generators/text_encoder.py
    """
    def detokenize(data: Iterator[List[str]]) -> Iterator[str]:
        for sentence in data:
            is_alnum = [t[0] in _ALNUM_CHARSET for t in sentence]
            ret = []
            for i, token in enumerate(sentence):
                if i > 0 and is_alnum[i - 1] and is_alnum[i]:
                    ret.append(" ")
                ret.append(token)
            yield "".join(ret)

    def writer(path: str, data: Iterator[List[str]]) -> None:
        writer = text_writer(encoding)
        writer(path, detokenize(data))

    return writer


# pylint: disable=invalid-name
UtfPlainTextWriter = tokenized_text_writer()
T2TWriter = t2t_tokenized_text_writer()
# pylint: enable=invalid-name
