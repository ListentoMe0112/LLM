from __future__ import annotations
import torch
import torch.nn as nn
from einops import einsum, reduce,rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor
import numpy.typing as npt

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import List, Tuple, Dict

import numpy.typing as npt
import torch
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.param = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(
            self.param,
            mean=0.0,
            std=2 / (self.in_features + self.out_features),
            a=-3,
            b=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.param, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters
        Args:
            num_embeddings (_type_): int Size of the vocabulary
            embedding_dim (_type_):  int sieze of d_model
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=2.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors
        """
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:

        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.param = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = reduce(x**2, "... d -> ... 1", "mean")
        rms += self.eps
        rms = rms.sqrt()
        result = x / rms * self.param
        return result.to(in_dtype)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, input_features) -> torch.Tensor:
        w1x = einsum(input_features, self.w1, "... d_model, d_ff d_model -> ... d_ff")

        siluw1x = torch.sigmoid(w1x) * w1x

        w3x = einsum(input_features, self.w3, "... d_model, d_ff d_model -> ... d_ff")

        inner = siluw1x * w3x

        ret = einsum(inner, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return ret


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        half_dim = d_k // 2
        i = torch.arange(max_seq_len, dtype=torch.float32)
        i = rearrange(i, "seq -> seq 1")# [seq_len, 1]
        k = torch.arange(half_dim, dtype=torch.float32)    
        k = rearrange(k, "half_dim ->  1 half_dim ") # [1, dim/2]

        # Compute inverse frequency as in RoPE paper
        inv_k = 1.0 / (theta ** (2 * k / d_k))  # [1, dim/2]

        # Compute angle = position * frequency
        angle = i * inv_k # [seq_len, dim/2]

        sin_angles = torch.sin(angle)
        cos_angles = torch.cos(angle)
        self.register_buffer("sin", sin_angles, persistent=False)
        self.register_buffer("cos", cos_angles, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
        Note that you should tolerate x with an arbitrary number of batch dimensions. 
        You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension. 
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
        """
        *batch_dims, seq_len, dim = x.shape
        assert dim % 2 == 0, "Embedding dimension must be even"
        half_dim = dim // 2

        x1 = x[..., 0:dim:2]  # even dims
        x2 = x[..., 1:dim:2]  # odd dims

        # sin/cos: [max_seq_len, dim/2] → index by token_positions → [..., seq_len, dim/2]
        sin_pos = self.sin[token_positions]  # [..., seq_len, dim/2]
        cos_pos = self.cos[token_positions]  # same shape

        # Make suresin/cos shape matches x1/x2
        # sin_pos = sin_pos.to(x1.device)
        # cos_pos = cos_pos.to(x1.device)
        print("====================================")
        print(x.shape, cos_pos.shape)
        print("====================================")

        rotated_even = x1 * cos_pos - x2 * sin_pos
        rotated_odd  = x1 * sin_pos + x2 * cos_pos

        x_out = torch.empty_like(x)
        x_out[..., 0:dim:2] = rotated_even
        x_out[..., 1:dim:2] = rotated_odd

        return x_out

def softmax(x : torch.Tensor, dim: int):
    x_max = torch.max(x, dim = dim, keepdim=True)
    x = x - x_max[0]
    x_exp = torch.exp(x)
    numerator = torch.sum(x_exp, dim = dim, keepdim=True)
    return x_exp / numerator 
    
def dot_product_attention(q: Float[Tensor, " ... queries d_k"],
    k: Float[Tensor, " ... keys d_k"],
    v: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    inv_d_k = torch.sqrt(torch.tensor(1 / k.shape[-1], dtype = torch.float32))
    pre_softmax_score = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
    pre_softmax_score *= inv_d_k

    if mask != None:
        mask = mask.to(torch.float32)
        mask = ((1 - mask) * 1e9)
        pre_softmax_score = pre_softmax_score - mask

    score = softmax(pre_softmax_score, dim = -1)
    attention = einsum(score, v, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k =  self.d_model // self.num_heads

    def forward(self, Q, K, V):
        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2]))
        mask = repeat(mask, "s1 s2 -> b h s1 s2", b = Q.shape[0], h = self.num_heads)
        o = dot_product_attention(Q, K, V, mask)
        return o
