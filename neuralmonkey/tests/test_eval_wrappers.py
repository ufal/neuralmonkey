# test evaluation metric wrappers

import unittest
import os.path

from neuralmonkey.evaluators.multeval import MultEvalWrapper
from neuralmonkey.evaluators.beer import BeerWrapper
from neuralmonkey.evaluators.gleu import GLEUEvaluator
from neuralmonkey.evaluators.f1_bio import F1Evaluator

REF = ["I", "like", "tulips", "."]
HYP = ["I", "hate", "flowers", "and", "stones", "."]

# 4 common, |bio| = 7, |bio_ref| = 6
BIO = "BBOBOOOBIIOOOOOBIBIIIIB"
BIO_REF = "BIOBOOOBIIBIOOOBIBIIIIO"

MULTEVAL = "scripts/multeval-0.5.1/multeval.sh"
BEER = "scripts/beer_2.0/beer"


class TestExternalEvaluators(unittest.TestCase):

    def test_multeval_bleu(self):
        if os.path.exists(MULTEVAL):
            multeval = MultEvalWrapper(MULTEVAL, metric="bleu")
            bleu = multeval([HYP], [REF])
            max_bleu = multeval([REF], [REF], )
            min_bleu = multeval([], [REF])
            self.assertEqual(max_bleu, 1.0)
            self.assertEqual(min_bleu, 0.042)  # smoothing
            self.assertAlmostEqual(bleu, 0.097)
        else:
            print("MultEval not installed, cannot be found here: {}".
                  format(MULTEVAL))

    def test_multeval_ter(self):
        if os.path.exists(MULTEVAL):
            multeval = MultEvalWrapper(MULTEVAL, metric="ter")
            ter = multeval([HYP], [REF])
            min_ter = multeval([REF], [REF])
            max_ter = multeval([], [REF])
            self.assertEqual(min_ter, 0.0)
            self.assertEqual(max_ter, 1.0)
            self.assertAlmostEqual(ter, 1.0)
        else:
            print("MultEval not installed, cannot be found here: {}".
                  format(MULTEVAL))

    def test_multeval_meteor(self):
        if os.path.exists(MULTEVAL):
            multeval = MultEvalWrapper(MULTEVAL, metric="meteor")
            meteor = multeval([HYP], [REF])
            max_meteor = multeval([REF], [REF])
            min_meteor = multeval([], [REF])
            self.assertEqual(max_meteor, 1.0)
            self.assertAlmostEqual(min_meteor, 0.0)
            self.assertAlmostEqual(meteor, 0.093)
        else:
            print("MultEval not installed, cannot be found here: {}".
                  format(MULTEVAL))

    def test_beer(self):
        if os.path.exists(BEER):
            beer_evaluator = BeerWrapper(BEER)
            beer = beer_evaluator([HYP], [REF])
            max_beer = beer_evaluator([REF], [REF])
            min_beer = beer_evaluator([], [REF])
            self.assertAlmostEqual(beer, 0.1120231)
            self.assertAlmostEqual(max_beer, 0.4744488)
            self.assertEqual(min_beer, 0)
        else:
            print("BEER not installed, cannot be found here: {}".format(BEER))

    def test_gleu(self):
        gleu_evaluator = GLEUEvaluator()
        gleu = gleu_evaluator([HYP], [REF])
        max_gleu = gleu_evaluator([REF], [REF])
        min_gleu = gleu_evaluator([], [REF])
        self.assertEqual(min_gleu, 0.0)
        self.assertEqual(max_gleu, 1.0)
        self.assertAlmostEqual(gleu, 0.1111111)

    def test_f1(self):
        f1_evaluator = F1Evaluator()
        f1val = f1_evaluator([BIO], [BIO_REF])
        self.assertAlmostEqual(f1val, 8.0/13.0)


if __name__ == "__main__":
    unittest.main()
