import regex as re

def untruecase(sentence):
    if sentence:
        return [sentence[0].capitalize()] + sentence[1:]
    else:
        return []


## Now starts the language specific code for geman

contractions = ["am", "ans", "beim", "im", "ins", "vom", "zum", "zur"]
contractions_set = set(contractions)
uncontracted_forms = [["an", "dem"], ["an", "das"], ["bei", "dem"], ["in", "dem"],
                ["in", "das"], ["von", "dem"], ["zu", "dem"], ["zu", "der"]]
uncontract = {c: un for c, un in zip(contractions, uncontracted_forms)}

contract = {}
for cont, (prep, article) in zip(contractions, uncontracted_forms):
    if not article in contract:
        contract[article] = {}
    contract[article][prep] = cont


class GermanPreprocessor(object):
    def __init__(self, compounding=True, contracting=True, pronouns=True):
        self.compounding = compounding
        self.contracting = contracting
        self.pronouns = pronouns

    def __call__(self, sentence):
        result = []

        for word in sentence:
            if self.contracting and word in contractions_set:
                result.extend(uncontract[word])
            else:
                result.append[word]

        return result


class GermanPostprocessor(object):
    def __init__(self, compounding=True, contracting=True, pronouns=True):
        self.compounding = compounding
        self.contracting = contracting
        self.pronouns = pronouns

    def __call__(self, sentence):
        result = []

        for word in sentence:
            if self.contracting and word in contract \
                    and result and result[-1] in contract[word]:
                result[-1] = contract[word][result[-1]]
            else:
                result.append(word)

        return result
