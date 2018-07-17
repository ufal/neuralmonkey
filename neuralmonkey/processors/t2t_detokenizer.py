import sys
import unicodedata
from typing import List

from neuralmonkey.readers.plain_text_reader import T2TReader

# This set contains all letter and number characters.
alphanumeric_charset = set(
  chr(i) for i in range(sys.maxunicode)
  if (unicodedata.category(chr(i)).startswith("L")
      or unicodedata.category(chr(i)).startswith("N")))


def decode(tokens: List[str]) -> List[str]:
  """Decode a list of tokens to a unicode string.
  Args:
    tokens: a list of Unicode strings
  Returns:
    a unicode string
  """
  token_is_alnum = [t[0] in alphanumeric_charset for t in tokens]
  ret = []
  for i, token in enumerate(tokens):
    if not token_is_alnum[i]:
      if token[0] == " ":
        token = token[1:]
      if token[-1] == " ":
        token = token[:-1]

    if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
      ret.append(" ")
    ret.append(token)
  return "".join(ret)


if __name__ == "__main__":

  for line in T2TReader(["/dev/stdin"]):
    print(decode(line))
