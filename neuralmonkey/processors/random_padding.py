from typing import List, Callable
import random


def random_padding(padding_symbol: str,
                   min_add: int,
                   max_add: int) -> Callable[[List[str]], List[str]]:

    rnd = random.Random()

    def process(sentence: List[str]) -> List[str]:
        rnd.seed(" ".join(sentence))
        add = rnd.randint(min_add, max_add)
        return sentence + [padding_symbol] * add

    return process
