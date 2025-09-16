from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    patch_size: int
    d_model: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    dropout: float
    image_size: int
    in_channels: int
    out_dim: int