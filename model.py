import torch
from torch import nn

from utils.custom_layers import PatchEmbeddingLayer, TransformerEncoder


class Vit(nn.Module):
    def __init__(
            self,
            patch_size: int = 16,
            embedding_dim: int = 768,
            image_size: int = 224,
            mlp_dropout: int = 0.1,
            in_channels: int = 3,
            num_heads: int = 12,
            attention_dropout: float = 0.0,
            mlp_size: int = 3072,
            num_classes: int = 10,
            num_transformer_layers: int = 12,
            device: torch.device = "cuda",
    ):
        super().__init__()

        # Make an assertion that the image size is compatible with the patch size
        assert (
                image_size % patch_size == 0
        ), f"Image size must be divisible by patch size, image: {image_size}, patch_size: {patch_size}"

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        num_patches = (image_size ** 2) // (patch_size ** 2)

        self.class_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        self.positional_embedding = nn.Parameter(
            torch.randn((1, num_patches + 1, embedding_dim)), requires_grad=True
        )

        self.embedding_dropout = nn.Dropout(p=mlp_dropout)

        self.patch_embedding_layer = PatchEmbeddingLayer(
            input_shape=in_channels, embedding_dim=embedding_dim, patch_size=patch_size
        )

        self.encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    dropout=mlp_dropout,
                    mlp_size=mlp_size,
                    device=device,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        # Get the batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to the batch size
        class_token = self.class_token.expand(
            batch_size, -1, -1
        )  # '-1' means to infer the dimensions

        # Equation 1
        x = self.patch_embedding_layer(x)

        # Concat class token embedding and patch embedding
        x = torch.cat(
            (class_token, x), dim=1
        )  # (batch_size, number_of_patches, embedding_dim)
        # Add position embedding to class token and patch embedding
        x = self.positional_embedding + x

        # Apply dropout to patch embedding
        x = self.embedding_dropout(x)

        # Pass position and patch embedding to transformer encoder (Equation 2 and 3)
        x = self.encoder(x)

        # Pass 0th index logit to classifier (Equation 4)
        x = self.classifier(x[:, 0])

        return x
