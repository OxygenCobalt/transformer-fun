from pydantic import BaseModel, ConfigDict

from .tok.tok import Tokenizer

class Experiments(BaseModel):
    eval_offsets: bool

class Hyperparams(BaseModel):
    n_layer: int
    embed_size: int
    q_heads: int
    kv_heads: int
    block_size: int
    ffn_ratio: float
    dropout: float
    batch_size: int

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hyperparams: Hyperparams
    tokenizer: Tokenizer
    experiments: Experiments
    optimizer: str
    optimizer_hyperparams: dict
    positions: str
    gating: str
    device: str
