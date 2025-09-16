import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from configs.model_config import ModelConfig

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0., attn_bias=False):
        super().__init__()

        assert d_model % n_heads == 0, "Number of heads must divide d_model"

        self.d_model = d_model
        self.n_heads = n_heads
        
        self.d_k = d_model // n_heads
        hasOut = not (d_model == (self.d_k) and self.n_heads == 1)

        self.w_q = nn.Linear(d_model, self.n_heads * self.d_k, bias=attn_bias)
        self.w_k = nn.Linear(d_model, self.n_heads * self.d_k, bias=attn_bias)
        self.w_v = nn.Linear(d_model, self.n_heads * self.d_k, bias=attn_bias)

        self.w_o = nn.Sequential(
            nn.Linear(self.n_heads * self.d_k, d_model),
            nn.Dropout(dropout),
        ) if hasOut else nn.Identity()

        self.linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):

        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        query = rearrange(query, "b t (h d) -> b h t d", h=self.n_heads)
        key = rearrange(key, "b t (h d) -> b h t d", h=self.n_heads)
        value = rearrange(value, "b t (h d) -> b h t d", h=self.n_heads)

        attention = torch.einsum("b h i d, b h j d -> b h i j", query, key)
        attention = attention / (self.d_k**0.5)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("b h i j, b h j d -> b h i d", attention, value)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.w_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float =0.):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int, in_chanels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.in_chanels = in_chanels
        
        self.ln1 = nn.LayerNorm(patch_size * patch_size * in_chanels)
        self.projection = nn.Linear(patch_size * patch_size * in_chanels, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.tensor):
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        patches = self.ln1(patches)
        x = self.projection(patches)
        x = self.ln2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

    def forward(self, x: torch.tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class ViT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert config.image_size % config.patch_size == 0, "Image size must be divisible by patch size"

        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.hidden_dim = config.hidden_dim
        self.dropout_rate = config.dropout
        self.n_blocks = config.n_blocks

        self.patch_embedding = PatchEmbedding(self.patch_size, self.d_model, self.in_channels)

        # pos embedding
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.dropout = nn.Dropout(self.dropout_rate)

        self.blocks = nn.ModuleList([AttentionBlock(self.d_model, self.n_heads, self.hidden_dim, self.dropout_rate) for _ in range(self.n_blocks)])

        self.ln = nn.LayerNorm(self.d_model)
        self.mlp_head = nn.Linear(self.d_model, config.out_dim)

    def forward(self, x: torch.tensor):
        x = self.patch_embedding(x)

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        x = self.dropout_rate(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x
    
