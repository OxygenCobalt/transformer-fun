import pickle

import bpe_native
from tok.tok import Tokenizer


class BPE(Tokenizer):
    def __init__(self, vocab: int):
        self._vocab_size = vocab
        self._codec = None

    def _set_pairs(self, pairs):
        codec = bpe_native._BpeCodec(pairs)
        self._pairs = pairs
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

    def load(self, file) -> bool:
        try:
            dat = pickle.Unpickler(file).load()
        except Exception as e:
            print("failed to load bpe tokenizer file: ", e)
            return False
        if self._vocab_size != dat["vocab"]:
            print("vocab diverges")
            return False
        self._set_pairs(dat["pairs"])

        return True

    def save(self, file) -> None:
        dat = {"vocab": self._vocab_size, "pairs": self._pairs}
        pickle.Pickler(file).dump(dat)
