from typing import Any, Callable, Generator, List
from random import randint


def preprocess_char_based(sentence: List[str]) -> List[str]:
    return list(" ".join(sentence))


def preprocess_add_noise(sentence: List[str]) -> List[str]:
    sent = sentence[:]
    length = len(sentence)
    if length > 1:
        for _ in range(length // 2):
            swap = randint(0, length - 2)
            sent[swap] = sent[swap + 1]
            sent[swap + 1] = sent[swap]
    return sent


# TODO refactor post-processors to work on sentence level
def postprocess_char_based(sentences: List[List[str]]) -> List[List[str]]:
    result = []

    for sentence in sentences:
        joined = "".join(sentence)
        tokenized = joined.split(" ")
        result.append(tokenized)

    return result


def untruecase(
        sentences: List[List[str]]) -> Generator[List[str], None, None]:
    for sentence in sentences:
        if sentence:
            yield [sentence[0].capitalize()] + sentence[1:]
        else:
            yield []


def pipeline(processors: List[Callable]) -> Callable:
    """Concatenate processors."""

    def process(data: Any) -> Any:
        for processor in processors:
            data = processor(data)
        return data

    return process
