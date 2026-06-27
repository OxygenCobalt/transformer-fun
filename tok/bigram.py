class Bigram:
    def __init__(self):
        pass

    def train(self, docs: list[str]):
        chars = sorted(list(set("".join(docs))))
        self.vocab = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def forward(self, s):
        return [self.stoi[c] for c in s]

    def backward(self, l):
        return "".join([self.itos[i] for i in l])
