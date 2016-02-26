#!/usr/bin/env python

import sys, codecs, argparse
from nltk import word_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", help="Lanuage the text is in.")
    args = parser.parse_args()

    if args.language not in set(['english', 'german']):
        raise Exception("Language must be 'englis' or 'german', not '{}'.".format(args.language))

    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    for line in sys.stdin:
        print ' '.join(word_tokenize(line, language=args.language))

