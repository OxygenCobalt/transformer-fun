import pickle
from typing import Self, Union

from tqdm import tqdm


class _BPETrie:
    def __init__(self, parent, token: int | None = None, seq: list = []):
        self.token = token
        self.seq = seq
        self.children: dict[int, _BPETrie] = {}
        self.parent = parent

    def insert(self, utf: list[int], token: int):
        cur = self
        for c in utf:
            if c in cur.children:
                cur = cur.children[c]
            else:
                new = _BPETrie(cur)
                cur.children[c] = new
                cur = new
        cur.token = token
        cur.seq = utf
        return cur

    def walk(self, seq: list[int], start: int):
        cur = self
        i = start
        while i < len(seq):
            c = seq[i]
            if c in cur.children:
                cur = cur.children[c]
                i += 1
            else:
                break
        while cur.token is None:
            cur = cur.parent
            i -= 1
        return (cur, i)


class BPE:
    def __init__(self, bow, vocab):
        self.vocab = vocab

    def train(self, bow):
        prog = tqdm(total=self.vocab)
        prog.desc = "bpe.train"
        utf = list(bow.encode("utf-8"))
        self.ctot = _BPETrie(None)
        self.ttoc: dict[int, _BPETrie] = {}
        for i, ch in enumerate(sorted(list(set(utf)))):
            self.ttoc[i] = self.ctot.insert([ch], i)
            prog.update(1)
        tokens = []
        for c in utf:
            token = self.ctot.children[c].token
            if token is None:
                raise RuntimeError("invalid shallow trie somehow")
            tokens.append(token)

        def pairs(tokens: list[int]) -> dict[tuple[int, int], list[int]]:
            pairs: dict[tuple[int, int], list[int]] = {}
            i = 0
            while i < len(tokens) - 1:
                now = tokens[i]
                later = tokens[i + 1]
                pair = (now, later)
                if pair in pairs:
                    pairs[pair].append(i)
                else:
                    pairs[pair] = [i]
                i += 1
            return pairs

        while len(self.ttoc) < self.vocab:
            p = pairs(tokens)
            if not p:
                break
            common_pair, common_pair_idxs = max(p.items(), key=lambda e: len(e[1]))
            common_now, common_later = common_pair
            new_token = len(self.ttoc)
            self.ttoc[new_token] = self.ttoc[common_now].insert(
                [common_later], new_token
            )
            new_tokens = []
            i = 0
            idxs = set(common_pair_idxs)
            while i < len(tokens):
                if i in idxs:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            prog.update(1)

        prog.close()

    def forward(self, s):
        tokens = []
        slice = list(s.encode("utf-8"))
        i = 0
        while i < len(s):
            node, ni = self.ctot.walk(slice, i)
            if i == ni:
                raise RuntimeError(
                    "ctot invalid! cannot find sequence for ",
                    slice[:10],
                    "..., somehow the root neighbors are ",
                    node.children.keys(),
                    " and all i got is ",
                    list(slice[0].encode("utf-8")),
                )
            tokens.append(node.token)
            i = ni
        return tokens

    def backward(self, s):
        utf = []
        for tok in s:
            utf += self.ttoc[tok].seq
        return bytes(utf).decode("utf-8", "replace")

    def load(self, file) -> bool:
        try:
            dat = pickle.Unpickler(file).load()
        except:
            return False
        self.vocab = dat["vocab"]
        self.ctot = dat["ctot"]
        self.ttoc = dat["ttoc"]
        return True

    def save(self, file):
        dat = {"vocab": self.vocab, "ctot": self.ctot, "ttoc": self.ttoc}
        pickle.Pickler(file).dump(dat)
