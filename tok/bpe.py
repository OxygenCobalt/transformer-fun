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
        utf = list(bow.encode("utf-8"))
        table: dict[Union[int, tuple[int, int]], int] = {}
        for i, ch in enumerate(sorted(list(set(utf)))):
            table[ch] = i
            prog.update(1)
        tokens = list(map(lambda c: table[c], utf))

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

        while len(table) < self.vocab:
            p = pairs(tokens)
            if not p:
                break
            common = max(p.items(), key=lambda e: len(e[1]))
            target, idxs = common
            idxs = set(idxs)
            new_token = len(table)
            table[target] = new_token
            new_tokens = []
            i = 0
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

        self.ctot = _BPETrie(None)
        self.ttoc: dict[int, _BPETrie] = {}

        flip = dict(map(lambda i: (i[1], i[0]), table.items()))
        for token, seq in tqdm(flip.items()):
            if isinstance(seq, int):
                self.ttoc[token] = self.ctot.insert([seq], token)
            else:
                seq = [seq[0], seq[1]]
                base = {}
                while True:
                    dirty = False
                    for i, s in enumerate(seq):
                        if i in base:
                            continue
                        u = flip[s]
                        if isinstance(u, int):
                            seq[i] = u
                            base[i] = True
                        else:
                            seq[i] = u[1]
                            seq.insert(i, u[0])
                            dirty = True
                    if not dirty:
                        break
                self.ttoc[token] = self.ctot.insert(seq, token)

    def forward(self, s):
        tokens = []
        slice = list(s.encode("utf-8"))
        prog = tqdm(total=len(slice))
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
            prog.update(ni - i)
            i = ni
        prog.close()
        return tokens

    def backward(self, s):
        utf = []
        for tok in s:
            utf += self.ttoc[tok].seq
        return bytes(utf).decode("utf-8", "replace")
