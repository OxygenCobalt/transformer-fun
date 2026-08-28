import pickle

import bpe_native
from .tok import Tokenizer

BASE_VOCAB_SIZE = 256
TERMINATOR = 256


class BPE(Tokenizer):
    def __init__(self, vocab: int):
        self._vocab_size = vocab
        self._codec = None

    def _set_pairs(self, pairs):
        codec = bpe_native._BpeCodec(pairs)
        self._pairs = pairs
        self._token_to_pair = {
            token: pair
            for token, pair in enumerate(pairs, start=BASE_VOCAB_SIZE + 1)
        }
        self._codec = codec

    def _require_codec(self):
        if self._codec is None:
            raise RuntimeError("BPE must be trained or loaded before use")
        return self._codec

    def train(self, docs: list[str]) -> None:
        self._set_pairs(bpe_native._train(docs, self._vocab_size))

    def vocab(self) -> int:
        return self._vocab_size

    def tokenize(self, docs: list[str]) -> list[int]:
        return self._require_codec().encode(docs)

    def stringify_one(self, token: int) -> str | None:
        self._require_codec()
        if token == TERMINATOR:
            return None

        expanded = [token]
        utf = []
        while expanded:
            current = int(expanded.pop())
            pair = self._token_to_pair.get(current)
            if pair is None:
                utf.append(current)
            else:
                left, right = pair
                expanded.extend((right, left))
        return bytes(utf).decode("utf-8", "replace")

    def load(self, path: str) -> bool:
        try:
            dat = None
            with open(path, "rb") as file:
                dat = pickle.Unpickler(file).load()
        except Exception as e:
            print("failed to load bpe tokenizer file: ", e)
            return False
        if self._vocab_size != dat["vocab"]:
            print("vocab diverges")
            return False
        self._set_pairs(dat["pairs"])
        return True

    def save(self, path: str) -> None:
        with open(path, "xb") as file:
            dat = {"vocab": self._vocab_size, "pairs": self._pairs}
            pickle.Pickler(file).dump(dat)

    def id(self) -> str:
        return "bpe"
