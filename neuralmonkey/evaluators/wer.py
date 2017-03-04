import pyter


# pylint: disable=too-few-public-methods
class WEREvaluator(object):
    """Compute WER using the pyter library."""
    def __init__(self, name="WER"):
        self.name = name

    def __call__(self, decoded, references):
        dist_sum = 0
        length_sum = 0
        for hyp, ref in zip(decoded, references):
            length_sum += len(ref)
            if ref and hyp:
                dist_sum += pyter.edit_distance(hyp, ref)
            elif not ref and not hyp:
                dist_sum += 0
            else:
                dist_sum += len(ref)
        return dist_sum / length_sum


WER = WEREvaluator()
