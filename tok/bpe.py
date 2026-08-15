import pickle
from collections import Counter
from itertools import pairwise

import bpe_native
from tqdm import tqdm

BASE_VOCAB_SIZE = 256
TERMINATOR = 256


class BPE:
    def __init__(self, vocab):
        self.vocab = vocab

    def train(self, docs: list[str]):
        self.pairs = bpe_native.train_native(docs, self.vocab)

        self.token_to_pair = dict(
            map(lambda t: (t[0] + BASE_VOCAB_SIZE + 1, t[1]), enumerate(self.pairs))
        )
        self.pair_to_token = dict(
            map(lambda t: (t[1], t[0] + BASE_VOCAB_SIZE + 1), enumerate(self.pairs))
        )

    def forward(self, doc):
        tokens = self.forward_continue(doc)
        tokens.append(TERMINATOR)
        return tokens

    def forward_continue(self, doc):
        tokens = []
        utf = list(doc.encode("utf-8"))
        for byte in utf:
            tokens.append(byte)
            while len(tokens) > 1:
                a = tokens[-2]
                b = tokens[-1]
                if (a, b) in self.pair_to_token:
                    tokens[-2] = self.pair_to_token[(a, b)]
                    tokens.pop()
                else:
                    break
        return tokens

    def backward_str(self, tokens) -> str:
        utf = []
        for tok in tokens:
            b = self.backward_utf(tok)
            if b is None:
                break
            utf += b
        return bytes(utf).decode("utf-8", "replace")

    def backward_one(self, token) -> str | None:
        b = self.backward_utf(token)
        if b is None:
            return None
        return bytes(b).decode("utf-8", "replace")

    def backward_utf(self, token) -> list[int] | None:
        if token == TERMINATOR:
            return None
        expanded = [token]
        dirty = True
        while dirty:
            dirty = False
            new_expanded = []
            for tok in expanded:
                insane_int_conversion = int(tok)
                if insane_int_conversion in self.token_to_pair:
                    a, b = self.token_to_pair[insane_int_conversion]
                    new_expanded.append(a)
                    new_expanded.append(b)
                    dirty = True
                else:
                    new_expanded.append(insane_int_conversion)
            expanded = new_expanded
        return expanded

    def load(self, file) -> bool:
        try:
            dat = pickle.Unpickler(file).load()
        except Exception as e:
            print("failed to load bpe tokenizer file: ", e)
            return False
        if self.vocab != dat["vocab"]:
            print("vocab diverges")
            return False
        self.vocab = dat["vocab"]
        self.pairs = dat["pairs"]
        self.token_to_pair = dict(
            map(lambda t: (t[0] + BASE_VOCAB_SIZE + 1, t[1]), enumerate(self.pairs))
        )
        self.pair_to_token = dict(
            map(lambda t: (t[1], t[0] + BASE_VOCAB_SIZE + 1), enumerate(self.pairs))
        )

        return True

    def save(self, file):
        dat = {"vocab": self.vocab, "pairs": self.pairs}
        pickle.Pickler(file).dump(dat)
        pass
