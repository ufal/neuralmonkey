import numpy as np
import tensorflow as tf
from collections import defaultdict
from neuralmonkey.logging import log
from neuralmonkey.model.sequence import EmbeddedSequence


class Word2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, sequence: EmbeddedSequence, eval_filename: str):
        self.vocabulary = sequence.vocabulary
        self._emb = sequence.embedding_matrix
        self._build_eval_graph()
        self._read_analogies(eval_filename)

    def _read_analogies(self, eval_filename):
        """Reads through the analogy question file.

        Returns:
          questions: a [n, 4] numpy array containing the analogy question's
                     word ids.
          questions_skipped: questions skipped due to unknown words.
        """
        questions = []
        self.question_types = []
        questions_skipped = 0
        quest_name = "NONAME"
        with open(eval_filename, "r") as analogy_f:
            for line in analogy_f:
                if line.startswith(":"):
                    quest_name = line.strip()
                    continue

                words = line.strip().split(" ")

                if len(words) != 4:
                    raise Exception("Following question do not have 4 words: "
                                    "{}".format(line.strip()))

                ids = []
                for w in words:
                    if w in self.vocabulary:
                        ids.append(self.vocabulary.get_word_index(w))
                    else:
                        ids.append(None)

                if None in ids:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
                    self.question_types.append(quest_name)

        log("Questions in total: {}, question skipped {}".format(
            len(questions), questions_skipped))
        self._analogy_questions = np.array(questions, dtype=np.int32)

    def _build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        nemb = tf.nn.l2_normalize(self._emb, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000,
                                                 len(self.vocabulary)))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

    def _predict(self, analogy, session):
        """Predict the top 4 answers for analogy questions."""
        idx, = session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx

    def eval(self, session):
        """Evaluate analogy questions and reports accuracy."""

        # How many questions we get right at precision@1.

        correct = 0
        correct_by_type = defaultdict(int)
        total_by_type = defaultdict(int)
        total = self._analogy_questions.shape[0]

        start = 0
        index = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub, session)
            start = limit
            for question in range(sub.shape[0]):
                total_by_type[self.question_types[index]] += 1
                for j in range(4):
                    if idx[question, j] == sub[question, 3]:
                        # We predicted correctly. E.g., [italy, rome, france].
                        correct += 1
                        correct_by_type[self.question_types[index]] += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break
                index += 1

        out = "Total word2vec eval %4.1f%%" % (correct * 100.0 / total)
        for type in sorted(total_by_type):
            out += "\t {} {:.1f}%".format(type, correct_by_type[type] * 100.0 /
                                          total_by_type[type])
        log(out)

        return out
