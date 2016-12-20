from typing import Generator, List
# tests: lint, mypy


def preprocess_char_based(sentence: List[str]) -> List[str]:
    return list(" ".join(sentence))


def postprocess_char_based(sentence: List[str]) -> List[str]:
    joined = "".join(sentence)
    tokenized = joined.split(" ")
    return tokenized


def untruecase(
        sentences: List[List[str]]) -> Generator[List[str], None, None]:
    for sentence in sentences:
        if sentence:
            yield [sentence[0].capitalize()] + sentence[1:]
        else:
            yield []
