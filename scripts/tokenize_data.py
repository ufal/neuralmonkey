#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, codecs, argparse
from nltk import word_tokenize
import javabridge
import gc
import regex as re
from termcolor import cprint

def get_decompounder():
    """
    Restarts the JVM with the decompounder. It is necessary once in a while.
    """
    javabridge.start_vm(class_path=["tf/jwordsplitter/target/jwordsplitter-4.2-SNAPSHOT.jar"])
    java_instance = javabridge.make_instance("de/danielnaber/jwordsplitter/GermanWordSplitter", "(Z)V", False)
    decompounder = javabridge.JWrapper(java_instance)
    return decompounder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizes and decompounds the text on stdin.")
    parser.add_argument("--language", required=True, help="Lanuage the text is in.")
    args = parser.parse_args()

    if args.language not in set(['english', 'german']):
        raise Exception("Language must be 'english' or 'german', not '{}'.".format(args.language))

    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr)

    try:
        if args.language == "german":
            decompounder = get_decompounder()
            decompounded_count = 0

        for ln, line in enumerate(sys.stdin):
            line = re.sub(r"[[:space:]]+", " ", line.rstrip())
            line = re.sub(r"^[[:space:]]+", "", line)
            line = re.sub(r"''", "\"", line)
            line = re.sub(r"``", "\"", line)
            line = re.sub(r"-([[:punct:]\$])", "\g<1>", line)
            line = re.sub(r"([[:punct:]\$])-", "\g<1>", line)
            line = re.sub(r"^[[:space:]]*-[[:space:]]", "", line)
            line = re.sub(r"([[:alpha:]0-9ß])-([ [:punct:]])", "\g<1>\g<2>", line, re.UNICODE)
            line = re.sub(r"([ [:punct:]])-([[:alpha:]0-9ß])", "\g<1>\g<2>", line, re.UNICODE)
            line = re.sub(r" - ", " – ", line)
            line = re.sub(r"– -", "–", line)

            def normalize_quotes(token):
                token = re.sub(r"-$", '', token)
                token = re.sub(r"``", '\u201c', token)
                token = re.sub(r"''", '\u201d', token)
                return token

            tokenized = [normalize_quotes(t) for t in word_tokenize(line, language=args.language)]

            if args.language == "german":
                for i, token in enumerate(tokenized):
                    decompounded_count += 1
                    decompounded = decompounder.splitWord(token)
                    if decompounded.size() >= 2:
                        tokenized[i] = \
                            "#".join([decompounded.get(j) for j in range(decompounded.size())])
                    del decompounded

                    if token.endswith("s") and not tokenized[i].endswith("s"):
                        tokenized[i] += "s"

                    # we need to manually garbage collect because of Java Heap Space
                    if decompounded_count % 150 == 0:
                        gc.collect()

            tokenized_string = ' '.join(tokenized)

            # Now put special character for spaces introduced by the tokenizer
            original_i = 0
            tokenized_chars_result = []
            for tokenized_i, char in enumerate(tokenized_string):
                #print u"pair '{}' ({}) and '{}' ({})".format(char, ord(char), line[original_i], ord(line[original_i]))
                if char == line[original_i] or (char == " " and ord(line[original_i]) == 160):
                    tokenized_chars_result.append(char)
                    original_i += 1
                    #print u"same characters {}".format(char)
                elif line[original_i] == '"' and (char == '\u201c' or char == '\u201d'):
                    original_i += 1
                    #print u"quotation mark {}".format(char)
                elif char == " ":
                    tokenized_chars_result.append("@")
                    #print "space added by tokenizer"
                elif char == "#":
                    if line[original_i] == "-":
                        tokenized_chars_result.append("-")
                        original_i += 1
                    else:
                        if args.language == 'german' and \
                                (line[original_i] == "s" or line[original_i] == "S") \
                                and line[original_i + 1] == tokenized_string[tokenized_i + 1]:
                            original_i += 1
                            tokenized_chars_result.append("$")
                            #print "decompounded with inserted s"
                        if args.language == 'german' and line[original_i] == "s" and line[original_i + 1] == "-":
                            original_i += 2
                            tokenized_chars_result.append("-")
                        else:
                            #print "decompompounded"
                            tokenized_chars_result.append("#")
                else:
                    #print ""
                    #print "Error on line {}".format(ln)
                    #cprint(u"tokenized on index {}: \"{}\", original on index {}: \"{}\""\
                    #        .format(tokenized_i, char, original_i, line[original_i]), 'yellow')
                    #cprint(line, 'red')
                    #cprint(tokenized_string, 'red')
                    #javabridge.kill_vm()
                    #exit()
                    tokenized_chars_result = list("<ERROR>")
                    break

            print("".join(tokenized_chars_result))
    except:
        javabridge.kill_vm()
        exit(1)
    finally:
        javabridge.kill_vm()
