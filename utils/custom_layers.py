import torch
from torch import nn


# Equation 1
class PatchEmbeddingLayer(nn.Module):
    def __init__(
            self, input_shape: int = 3, embedding_dim: int = 768, patch_size: int = 16
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_shape,
                embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            nn.Flatten(start_dim=2, end_dim=-1),
        )
        self.patch_size = patch_size

    def forward(self, x) -> torch.Tensor:
        image_size = x.shape[-1]
        assert (
            not image_size % self.patch_size
        ), f"Image size must be divisible by patch size, image shape: {image_size}, patch_size: {self.patch_size}"
        return self.block(x).permute(
            0, 2, 1
        )  # (batch_size, num_patches, embedding_dimensions)


# Equation 2
class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            num_heads: int = 12,
            attention_dropout: float = 0.0,
            device: torch.device = "cpu",
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, device=device)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            device=device,
            batch_first=True,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.layer_norm(x)
        x, _ = self.multi_head_attention(query=x, key=x, value=x)
        return x


# Equation 3
class MLPLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            mlp_size: int = 3072,
            dropout: float = 0.1,
            device: torch.device = "cpu",
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.mlp(self.layer_norm(x))


# Equation 2 and 3 together
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            num_heads: int = 12,
            attention_dropout: float = 0.0,
            dropout: float = 0.1,
            mlp_size: int = 3072,
            device: torch.device = "cpu",
    ):
        super().__init__()
        self.MSA_block = MultiHeadSelfAttentionLayer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            device=device,
        )
        self.MLP_block = MLPLayer(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=dropout,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.MSA_block(x) + x
        x = self.MLP_block(x) + x
        return x
