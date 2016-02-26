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

    if args.language == "german":
        import javabridge
        javabridge.start_vm(class_path=["tf/jwordsplitter/target/jwordsplitter-4.2-SNAPSHOT.jar"])
        java_instance = javabridge.make_instance("de/danielnaber/jwordsplitter/GermanWordSplitter", "(Z)V", True)
        decompounder = javabridge.JWrapper(java_instance)

    for line in sys.stdin:
        tokenized = word_tokenize(line, language=args.language)

        if args.language == "german":
            for i, token in enumerate(tokenized):
                decompounded = decompounder.splitWord(token)
                if decompounded.size() >= 2:
                    tokenized[i] = \
                        "#".join([decompounded.get(j) for j in range(decompounded.size())])

        print ' '.join(tokenized)
    javabridge.kill_vm()
