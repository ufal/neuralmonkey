from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BLEUEvaluator(object):
    bleu_smoothing = SmoothingFunction(epsilon=0.01).method1

    def __init__(self, n=4, deduplicate=False, name=None):
        self.n = n
        self.weights = [1.0/n for _ in range(n)]
        self.deduplicate = deduplicate

        if name is not None:
            self.name = name
        else:
            self.name = "BLEU-{}".format(n)
            if self.deduplicate:
                self.name += "-dedup"



    def __call__(self, decoded, references):
        # type: (List[List[str]], List[List[str]]) -> float
        listed_references = [[s] for s in references]

        if self.deduplicate:
            decoded = BLEUEvaluator._deduplicate_sentences(decoded)

        return 100 * corpus_bleu(listed_references, decoded, self.weights,
                                 BLEUEvaluator.bleu_smoothing)


    @staticmethod
    def _deduplicate_sentences(sentences):
        # type: List[List[str]] -> List[List[str]]
        deduplicated_sentences = []

        for sentence in sentences:
            last_w = None
            dedup_snt = []

            for word in sentence:
                if word != last_w:
                    dedup_snt.append(word)
                    last_w = word

            deduplicated_sentences.append(dedup_snt)

        return deduplicated_sentences


    @staticmethod
    def compare_scores(score1, score2):
        # type: (float, float) -> int
        # the bigger the better
        return (score1 > score2) - (score1 < score2)
