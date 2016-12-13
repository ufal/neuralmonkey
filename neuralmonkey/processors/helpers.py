from typing import Generator, List
# tests: lint, mypy


def preprocess_char_based(sequences: List[List[str]]) -> List[List[str]]:
    return [list(" ".join(sequence)) for sequence in sequences]


def postprocess_char_based(sequences: List[List[str]]) -> List[List[str]]:
    return [["".join(sqc)] for sqc in sequences]


def untruecase(
        sentences: List[List[str]]) -> Generator[List[str], None, None]:
    for sentence in sentences:
        if sentence:
            yield [sentence[0].capitalize()] + sentence[1:]
        else:
            yield []
