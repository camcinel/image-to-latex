import pandas as pd
import os


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab_path):
        self.idx2word = pd.read_pickle(os.path.join(vocab_path, 'dict_id2word.pkl'))
        self.word2idx = {word: idx for idx, word in self.idx2word.items()}

    def __call__(self, word):
        if not word in self.word2idx:
            return 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def load_vocab(vocab_path):
    return Vocabulary(vocab_path)