import pyter


# pylint: disable=too-few-public-methods
class TEREvalutator(object):
    """Compute TER using the pyter library."""
    def __init__(self, name="TER"):
        self.name = name

    def __call__(self, decoded, references):
        ter_sum = 0
        for hyp, ref in zip(decoded, references):
            if ref and hyp:
                ter_sum += pyter.ter(hyp, ref)
            elif not ref and not hyp:
                ter_sum += 0.
            else:
                ter_sum += 1.
        return ter_sum / len(decoded)


TER = TEREvalutator()
