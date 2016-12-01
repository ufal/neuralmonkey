# tests: lint

def preprocess_char_based(sequences):
    return [list(sequence) for sequence in sequences]


def postprocess_char_based(sequences):
    return [["".join(sqc)] for sqc in sequences]


def untruecase(sentences):
    for sentence in sentences:
        if sentence:
            yield [sentence[0].capitalize()] + sentence[1:]
        else:
            yield []
