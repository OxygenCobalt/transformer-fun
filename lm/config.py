from pydantic import BaseModel, ConfigDict

from .tok.tok import Tokenizer

class Experiments(BaseModel):
    eval_offsets: bool

class Hyperparams(BaseModel):
    n_layer: int
    embed_size: int
    heads: int
    block_size: int
    ffn_ratio: float
    dropout: float
    learning_rate: float
    batch_size: int

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hyperparams: Hyperparams
    tokenizer: Tokenizer
    experiments: Experiments
    positions: str
    gating: str
    device: str
