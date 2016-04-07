import regex as re

def untruecase(sentence):
    if sentence:
        return [sentence[0].capitalize()] + sentence[1:]
    else:
        return []

class GermanPreprocessor(object):
    def __init__(self, compounding, contracting, pronouns):
        pass

    def __call__(self, sentence):
        pass

class GermanPostprocessor(object):
    def __init__(self, compounding, contracting, pronouns):
        pass

    def __call__(self, sentence):
        pass
