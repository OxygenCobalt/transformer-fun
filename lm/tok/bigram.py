class Bigram:
    def __init__(self):
        self.vocab_size = 0
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.chars: list[str] = []

    def train(self, docs: list[str]) -> None:
        self.chars = sorted(list(set("".join(docs))))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def vocab(self) -> int:
        return self.vocab_size

    def tokenize(self, docs: list[str]) -> list[int]:
        tokens = []
        for doc in docs:
            tokens += [self.stoi[c] for c in doc]
        return tokens

    def backward(self, tokens: list[int]) -> str:
        return "".join([self.itos[i] for i in tokens])

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as file:
                self.chars = list(file.read().decode("utf-8"))
        except Exception as e:
            print("failed to load bpe tokenizer file: ", e)
            return False
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        return True

    def save(self, path: str) -> None:
        with open(path, "xb") as file:
            file.write("".join(self.chars).encode("utf-8"))

    def id(self) -> str:
        return "bigram"
