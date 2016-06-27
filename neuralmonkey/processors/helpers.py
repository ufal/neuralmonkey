def preprocess_char_based(sequence):
    return list(sequence)


def postprocess_char_based(sequences, _):
    return [["".join(sqc)] for sqc in sequences]


def untruecase(sentence):
    if sentence:
        return [sentence[0].capitalize()] + sentence[1:]
    else:
        return []
