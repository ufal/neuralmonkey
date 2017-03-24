import pyter


# pylint: disable=too-few-public-methods
class TEREvalutator(object):
    """Compute TER using the pyter library."""
    def __init__(self, name: str="TER") -> None:
        self.name = name

    def __call__(self, decoded, references) -> float:
        ter_sum = 0.
        count = 0
        for hyp, ref in zip(decoded, references):
            count += 1
            if ref and hyp:
                ter_sum += pyter.ter(hyp, ref)
            elif not ref and not hyp:
                ter_sum += 0.
            else:
                ter_sum += 1.
        return ter_sum / count


TER = TEREvalutator()
