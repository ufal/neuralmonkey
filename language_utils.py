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


ein_type_pronouns = re.compile("^(ein|[mdsk]ein|ihr|unser|euer|Ihr)(e|es|er|em|en)$")
der_type_pronouns = re.compile("^(dies|welch|jed|all)(e|es|er|em|en)$")



class GermanPreprocessor(object):
    def __init__(self, compounding=True, contracting=True, pronouns=True):
        self.compounding = compounding
        self.contracting = contracting
        self.pronouns = pronouns

    def __call__(self, sentence):
        result = []

        for word in sentence:
            if self.pronouns:
                ein_match = ein_type_pronouns.match(word)
                der_match = der_type_pronouns.match(word)

            if self.contracting and word in contractions_set:
                result.extend(uncontract[word])
            elif self.pronouns and ein_match:
                result.append(ein_match[1])
                result.append("<<"+ein_match[2])
            elif self.pronouns and der_match:
                result.append(der_match[1])
                result.append("<<"+der_match[2])
            elif self.compounding and word.find(">><<") > -1:
                compound_parts = word.split(">><<")
                result.append(compound_parts[0])
                for w in compound_parts[:1]:
                    result.append(">><<")
                    result.append(w.capitalize())
            else:
                result.append(word)

        return result


class GermanPostprocessor(object):
    def __init__(self, compounding=True, contracting=True, pronouns=True):
        self.compounding = compounding
        self.contracting = contracting
        self.pronouns = pronouns

    def __call__(self, sentence):
        result = []

        compound = False
        for word in sentence:
            if self.contracting and word in contract \
                    and result and result[-1] in contract[word]:
                result[-1] = contract[word][result[-1]]
            elif self.pronouns and word.startswith("<<"):
                if result:
                    result[-1] += word[2:]
            elif self.compounding and result and word == ">><<":
                compound = True
            elif self.compounding and compound:
                # TODO inserting 's'
                result[-1] += word.lower()
            else:
                result.append(word)

        result[0] = result[0].capitalize()

        return result
