"""A simple de-tokenizer for MT post-processing."""

import re
import sys

__author__ = "Ondřej Dušek"
__date__ = "2013"

DEFAULT_ENCODING = 'UTF-8'


class Detokenizer(object):
    """A simple de-tokenizer class."""

    # Moses special characters de-escaping
    ESCAPES = [('&bar;', '|'),
               ('&lt;', '<'),
               ('&gt;', '>'),
               ('&bra;', '['),
               ('&ket;', ']'),
               ('&amp;', '&')]  # should go last to prevent double de-escaping

    # Contractions for different languages
    CONTRACTIONS = {'en': r'^\p{Alpha}+(\'(ll|ve|re|[dsm])|n\'t)$',
                    'fr': r'^([cjtmnsdl]|qu)\'\p{Alpha}+$',
                    'es': r'^[dl]\'\p{Alpha}+$',
                    'it': r'^\p{Alpha}*(l\'\p{Alpha}+|[cv]\'è)$',
                    'cs': r'^\p{Alpha}+[-–](mail|li)$', }

    def __init__(self,
                 language: str,
                 moses_deescape: bool = True,
                 capitalize_sents: bool = False):
        """Constructor (pre-compile all needed regexes)."""
        self.moses_deescape = moses_deescape
        self.language = language
        self.capitalize_sents = capitalize_sents

        # compile regexes
        self.__currency_or_init_punct = re.compile(r'^[\p{Sc}\(\[\{\¿\¡]+$')
        self.__noprespace_punct = re.compile(r'^[\,\.\?\!\:\;\\\%\}\]\)]+$')
        self.__cjk_chars = re.compile(r'[\u1100-\u11FF\u2E80-\uA4CF\uA840-\uA87F'
                                 + r'\uAC00-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F'
                                 + r'\uFF65-\uFFDC]')
        self.__final_punct = re.compile(r'([\.!?])([\'\"\)\]\p{Pf}\%])*$')
        # language-specific regexes
        self.__fr_prespace_punct = re.compile(r'^[\?\!\:\;\\\%]$')
        self.__contract = None
        if self.language in self.CONTRACTIONS:
            self.__contract = re.compile(self.CONTRACTIONS[self.language],
                                    re.IGNORECASE)

    def __call__(self, words: List[str]) -> List[str]:
        """\
        Detokenize the given text using current settings.
        """
        # paste text back, omitting spaces where needed
        text = ''
        pre_spc = ' '
        quote_count = {'\'': 0, '"': 0, '`': 0}
        for pos, word in enumerate(words):
            # remove spaces in between CJK chars
            if self.__cjk_chars.match(text[-1:]) and \
                    self.__cjk_chars.match(word[:1]):
                text += word
                pre_spc = ' '
            # no space after currency and initial punctuation
            elif self.__currency_or_init_punct.match(word):
                text += pre_spc + word
                pre_spc = ''
            # no space before commas etc. (exclude some punctuation for French)
            elif self.__noprespace_punct.match(word) and \
                    (self.language != 'fr' or not
                     self.__fr_prespace_punct.match(word)):
                text += word
                pre_spc = ' '
            # contractions with comma or hyphen
            elif word in "'-–" and pos > 0 and pos < len(words) - 1 \
                    and self.__contract is not None \
                    and self.__contract.match(''.join(words[pos - 1:pos + 2])):
                text += word
                pre_spc = ''
            # handle quoting
            elif word in '\'"„“”‚‘’`':
                # detect opening and closing quotes by counting
                # the appropriate quote types
                quote_type = word
                if quote_type in '„“”':
                    quote_type = '"'
                elif quote_type in '‚‘’':
                    quote_type = '\''
                # exceptions for true Unicode quotes in Czech & German
                if self.language in ['cs', 'de'] and word in '„‚':
                    quote_count[quote_type] = 0
                elif self.language in ['cs', 'de'] and word in '“‘':
                    quote_count[quote_type] = 1
                # special case: possessives in English ("Jones'" etc.)
                if self.language == 'en' and text.endswith('s'):
                    text += word
                    pre_spc = ' '
                # really a quotation mark
                else:
                    # opening quote
                    if quote_count[quote_type] % 2 == 0:
                        text += pre_spc + word
                        pre_spc = ''
                    # closing quote
                    else:
                        text += word
                        pre_spc = ' '
                    quote_count[quote_type] += 1
            # keep spaces around normal words
            else:
                text += pre_spc + word
                pre_spc = ' '
        # de-escape chars that are special to Moses
        if self.moses_deescape:
            for char, repl in self.ESCAPES:
                text = text.replace(char, repl)
        # strip leading/trailing space
        text = text.strip()
        # capitalize, if the sentence ends with a final punctuation
        if self.capitalize_sents and self.__final_punct.search(text):
            text = text[0].upper() + text[1:]
        return text.split(' ')
