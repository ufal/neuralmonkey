#!/usr/bin/env python3

import sys
import codecs
import javabridge

from tokenize_data import get_decompounder

def main():
    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr)

    try:
        decompounder = get_decompounder()
        for line in sys.stdin:
            tokens = []
            for token in line.rstrip().split(" "):
                if not token:
                    continue
                if token[0].isupper():
                    decompounded = decompounder.splitWord(token)
                    if decompounded.size() >= 2:
                        parts = [decompounded.get(j)
                                 for j in range(decompounded.size())]
                        parts_with_hyphens = ['-' if not p else p
                                              for p in parts]
                        tokens.append(">><<".join(parts_with_hyphens))
                        del decompounded
                    else:
                        tokens.append(token)
                else:
                    tokens.append(token)
            print(" ".join(tokens))
#    except:
#        javabridge.kill_vm()
#        exit(1)
    finally:
        javabridge.kill_vm()

if __name__ == "__main__":
    main()
