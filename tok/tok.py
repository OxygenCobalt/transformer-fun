from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def vocab(self) -> int:
        pass

    @abstractmethod
    def train(self, docs: list[str]) -> None:
        pass

    @abstractmethod
    def tokenize(self, docs: list[str]) -> list[int]:
        pass

    @abstractmethod
    def load(self, file) -> bool:
        pass

    @abstractmethod
    def save(self, file) -> None:
        pass
