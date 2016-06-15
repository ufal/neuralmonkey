#!/usr/bin/env python

import codecs
import collections
import regex as re

from readers.plain_text_reader import PlainTextFileReader
from utils import log

from lib.subword_nmt.apply_bpe import BPE, encode

class BPEPreprocessor(object):
    """ Wrapper class for Byte-Pair-Encoding from Edinburgh """

    def __init__(self, **kwargs):

        if "merge_file" not in kwargs:
            raise Exception("No merge file for BPE preprocessor")

        log("Initializing BPE preprocessor")

        separator = kwargs.get("separator", "@@")
        merge_file = kwargs["merge_file"]

        with codecs.open(merge_file, "r", "utf-8") as f_data:
            self.bpe = BPE(f_data, separator)


    def __call__(self, sentence):
        # type: List[str] -> List[str]
        """ Adapted code from BPE.segment """

        output = []
        for word in sentence:

            ## Hack. TODO: inspect why there are empty sentences
            if len(word) == 0:
                output.append(word)
                continue

            new_word = encode(word, self.bpe.bpe_codes)

            for item in new_word[:-1]:
                output.append(item + self.bpe.separator)
            output.append(new_word[-1])

        return output




# class BPEPreprocessor_old(object):
#     """ Wrapper class for Byte-Pair-Encoding from Edinburgh """

#     def __init__(self, **kwargs):
#         if "data" not in kwargs:
#             raise Exception("No data for BPE preprocessor")

#         log("Initializing BPE preprocessor")

#         self.vocab = dict()

#         if isinstance(kwargs["data"], list):
#             for f in kwargs["data"]:
#                 self._add_file(f)
#         elif isinstance(kwargs["data"], str):
#             self._add_file(kwargs["data"])
#         else:
#             raise Exception("Unsupported type for data cfg.")

#         self.num_merges = kwargs.get('num_merges', 50000)
#         self.merges = [] # list of regexes

#         for i in range(self.num_merges):
#             bigram_vocab = self._get_stats()
#             best = max(bigram_vocab, key=bigram_vocab.get)

#             if bigram_vocab[best] < 2:
#                 log("No mergeable bigrams in vocabulary")
#                 break

#             self._merge_bigram_in_vocab(best)

#         log("BPE preprocessor initialized")


#     def _get_stats(self):
#         bigram_vocab = collections.defaultdict(int) # type: Dict[(str,str), int]
#         for word, freq in self.vocab.iteritems():
#             symbols = word.split(" ")
#             for i in range(len(symbols) - 1):
#                 bigram_vocab[symbols[i],symbols[i+1]] += freq
#         return bigram_vocab


#     def _merge_bigram_in_vocab(self, pair):
#         vocab_merged = {}
#         bigram = re.escape(" ".join(pair))
#         ## matches exactly the bigram, surrounded by spaces
#         p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

#         joined_bigram = "".join(pair)
#         self.merges.append((p, joined_bigram))

#         for word in self.vocab:
#             word_merged = p.sub(joined_bigram, word)
#             vocab_merged[word_merged] = self.vocab[word]

#         self.vocab = vocab_merged


#     def _add_file(self, path):
#         reader = PlainTextFileReader(path)
#         for line in reader.read():
#             for token in line:
#                 symbols = " ".join(list(token) + [u"</w>"])

#                 if symbols not in self.vocab:
#                     self.vocab[symbols] = 0

#                 self.vocab[symbols] += 1


#     def __call__(self, tokens):
#         # type: List[str] -> List[str]
#         tokens_processed = []
#         for token in tokens:
#             symbols = " ".join(list(token) + [u"</w>"])

#             for pattern, replacement in self.merges:
#                 symbols = pattern.sub(replacement, symbols)

#             tokens_processed += symbols.split(" ")

#         return tokens_processed


class BPEPostprocessor(object):

    def __init__(self, **kwargs):
        self.separator = kwargs.get("separator", "@@")

        esc = re.escape(self.separator)
        self.pattern = re.compile(esc + r" ")

    def __call__(self, decoded_sentences, dataset):
        # type: List[List[str]] -> List[List[str]]
        return [self.decode(s) for s in decoded_sentences]


    def decode(self, sentence):
        # type: List[str] -> List[str]

        joined = " ".join(sentence)
        decoded = self.pattern.sub("", joined)
        splitted = decoded.split(" ")

        return splitted



    # def _call2__(self, decoded_sentences, dataset):
    #     # type: List[List[str]] -> List[List[str]]

    #     pp_batch = []

    #     for sentence in decoded_sentences:
    #         word = ""
    #         pp_sentence = []

    #         for token in sentence:
    #             if token.endswith(self.separator):
    #                 word += token[:-(len(self.separator))]
    #             else:
    #                 word += token
    #                 pp_sentence.append(word)
    #                 word = ""

    #         pp_batch.append(pp_sentence)

    #     return pp_batch

    # def _foo():
    #     postprocessed = []

    #     for sentence in decoded_sentences:
    #         joined = "".join(sentence)
    #         replaced = joined.replace(" ","").replace("</w>"," ").rstrip()
    #         splitted = replaced.split(" ")

    #         postprocessed.append(splitted)

    #     return postprocessed
