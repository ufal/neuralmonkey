import random
import numpy as np

class Vocabulary(object):
    def __init__(self, tokenized_text=None):
        self.word_to_index = {}
        self.index_to_word = []
        self.word_count = {}

        self.add_tokenized_text(["<pad>", "<s>", "</s>", "<unk>"])

        if tokenized_text:
            self.add_tokenized_text(tokenized_text)

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)
            self.word_count[word] = 0
        self.word_count[word] += 1

    def add_tokenized_text(self, tokenized_text):
        for word in tokenized_text:
            self.add_word(word)

    def get_train_word_index(self, word):
        if word not in self.word_count or self.word_count[word] <= 1:
            if random.random > 0.5:
                return self.word_to_index["<unk>"]
            else:
                return self.word_to_index[word]
        else:
            return self.word_to_index[word]

    def get_word_index(self, word):
        if word not in self.word_count:
            return self.word_to_index["<unk>"]
        else:
            return self.word_to_index[word]

    def __len__(self):
        return len(self.index_to_word)

    def sentences_to_tensor(self, sentences, max_len, train=False):
        """
        Generates the tensor representation for the provided sentences.

        Args:

            sentences: List of sentences as lists of tokens.
            max_len: Maximum lengh of a sentence toward which they will be
              padded to.
            train: Flag whehter this is for training purposes.

        """

        word_indices = [np.zeros([len(sentences)], dtype=np.int) for _ in range(max_len + 2)]
        weights = [np.zeros([len(sentences)]) for _ in range(max_len + 1)]

        word_indices[0] += self.get_word_index("<s>")

        for i in range(max_len + 1):
            for j, sent in enumerate(sentences):
                if i < len(sent):
                    word_indices[i + 1][j] = self.get_train_word_index(sent[i]) if train \
                                         else self.get_word_index(sent[i])
                    weights[i][j] = 1.0
                elif i == len(sent):
                    word_indices[i + 1][j] = self.get_word_index("</s>")
                    weights[i][j] = 1.0

        return word_indices, weights

    def vectors_to_sentences(self, vectors):
        sentences = [[] for _ in range(vectors[0].shape[0])]

        for vec in vectors:
            for sentence, word_i in zip(sentences, vec):
                if sentence and sentence[-1] != "</s>":
                    sentence.append(self.index_to_word[word_i])

        return [s[:-1] for s in sentences]



