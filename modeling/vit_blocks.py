import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        Splitting an image into patches and linearly projecting these flattened 
        patches as a single convolution operation.

        Why this works:
        1. Convolutional layers are highly optimized in PyTorch
        2. The Conv2D does two things at once:
            - Projects the patches linearly (output channels = embed_dim)
              Each kernel filter acts as a learnable projection matrix that maps the 
              flattened patch (patch_size * patch_size * input_channels) to the embedding 
              space.
              This is mathematically equivalent to a linear layer applied to flattened 
              patches but more streamlined.

            - Splits the image into patches (kernel_size=patch_size, stride=patch_size) 
              It automatically slides over the image and extracts non-overlapping patches
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, num_patches, embed_dim)
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False):
        """
        Multi-head self-attention mechanism.

        How this works:
        1. QKV projection: Projects the input into three different linear spaces (query, key, value)
        2. Attention calculation: 
            - Scaled dot-product attention: 
                - Computes the similarity between query and key vectors, \alpha = q x k^T
                - Scales the dot product by the square root of the key dimension, \alpha = \alpha / \sqrt(d_k)
                - Applies softmax to get attention weights
            - Weighted sum: 
                - Multiplies the attention weights with the value vectors, \alpha x v
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, dim) where N is sequence length
        Returns:
            Tensor of shape (B, N, dim)
        """
        B, N, C = x.shape
        
        """
        1. Splitting into heads
            To do multi-head attention, we must divide that 768 dimension into 12 
            separate, smaller heads (where each head gets size 768 / 12 = 64).
        Original shape: (B, N, 3 * C), Reshape to: (B, N, 3, num_heads, head_dim)
        """
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        """
        2. Grouping by head
            Following "multi-head" mechanism, we group the Q, K, V projections 
            by head. This means we have 3 sets of 12 smaller projections (each 
            with 64 dimensions).

            Also, PyTorch's @ operator performs matrix multiplication on the 
            last two dimensions of a tensor. Hence, we permute to: 
            (3, B, num_heads, N, head_dim)
        """
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. Attention Calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) @ v
        
        # 4. Concatenate Heads
        out = attn.transpose(1, 2).reshape(B, N, C)

        # 5. Final Projection
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        """
        A single Vision Transformer block.
        From the original paper: https://arxiv.org/abs/2010.11929,
           "transformer encoder consists of alternating layers of multiheaded self-
            attention and MLP blocks. LayerNorm (LN) is applied before every block,
            and residual connections after every block. The MLP contains two layers
            with a GELU non-linearity."

        What and Why?
        1. Layer Normalization: 
            - Ensures the input has zero mean and unit variance
            - Improves numerical stability and allows for better learning
        2. Self-Attention: 
            - Computes attention weights for each position
            - Aggregates the weighted values to produce the final output
        3. Feed-Forward Network: 
            - Projects the output through a series of linear layers
            - Non-linear activation (GELU) introduces non-linearity
        4. Residual Connections: 
            - Adds the original input to the output of the feed-forward network
            - Helps with gradient flow and allows for deeper networks
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input sequence of shape (B, N, dim)
        """
        # Following equations 1-4 from ViT paper: https://arxiv.org/abs/2010.11929
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
