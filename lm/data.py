import torch
from torch import Tensor
from pydantic import BaseModel
from .tok.tok import Tokenizer
import pandas
import tqdm

class Dataset(BaseModel):
    id: str
    train: list[str]
    test: list[str]
    validation: list[str]

    def prepare(self, tokenizer: Tokenizer, device: str) -> dict[str, Tensor]:
        def tokenize_docs(docs: list[str], desc: str) -> list[int]:
            tokens: list[int] = []
            tokenizer_batch_size = 4096
            with tqdm.tqdm(total=len(docs), desc=desc) as progress:
                for start in range(0, len(docs), tokenizer_batch_size):
                    batch = docs[start : start + tokenizer_batch_size]
                    tokens.extend(tokenizer.tokenize(batch))
                    progress.update(len(batch))
            return tokens

        train = tokenize_docs(self.train, "train")
        test = tokenize_docs(self.test, "test")
        validation = tokenize_docs(self.validation, "validation")

        train = torch.tensor(train, dtype=torch.long).to(device)
        test = torch.tensor(test, dtype=torch.long).to(device)
        validation = torch.tensor(validation, dtype=torch.long).to(device)
        return {
            "train": train,
            "test": test,
            "validation": validation
        }


def wikitext_103_raw_v1(path: str) -> Dataset:
    train_data = pandas.concat(
        [
            pandas.read_parquet(f"./{path}/train-00000-of-00002.parquet"),
            pandas.read_parquet(f"./{path}/train-00001-of-00002.parquet"),
        ]
    )
    train_docs: list[str] = [text for text in train_data["text"]]
    test_data = pandas.concat(
        [
            pandas.read_parquet(f"./{path}/test-00000-of-00001.parquet"),
        ]
    )
    test_docs: list[str] = [text for text in test_data["text"]]

    validation_data = pandas.concat(
        [
            pandas.read_parquet(f"./{path}/validation-00000-of-00001.parquet"),
        ]
    )
    validation_docs: list[str] = [text for text in validation_data["text"]]
    return Dataset(
        id="wikitext-103-raw-v1",
        train=train_docs,
        test=test_docs,
        validation=validation_docs
    )
