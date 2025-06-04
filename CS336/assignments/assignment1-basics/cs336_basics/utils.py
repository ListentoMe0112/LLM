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
import math

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rms_norm1 = RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.theta = theta
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rope = RoPE(self.theta, self.d_k, self.max_seq_len) 
        self.rms_norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)


    def forward(self, in_features):
        norm_features = self.rms_norm1(in_features)
        token_positions = torch.arange(in_features.shape[-2])
        token_positions = repeat(token_positions, "seq -> b seq", b = in_features.shape[0])
        Q = self.q_proj(norm_features)
        K = self.k_proj(norm_features)
        V = self.v_proj(norm_features)
        Q = rearrange(Q, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = self.num_heads)
        K = rearrange(K, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = self.num_heads)
        V = rearrange(V, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = self.num_heads)


        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        O =  self.mha(Q, K, V)
        O  = rearrange(O, "... num_head seq_len d_k-> ... seq_len (num_head d_k)")
        O = self.o_proj(O)
        in_features = in_features + O

        norm_features = self.rms_norm2(in_features)

        ret = self.ffn(norm_features) + in_features
        return ret 

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, theta, max_seq_len, d_ff, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.dff  = d_ff
        self.num_layes = num_layers
        
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, theta, max_seq_len, d_ff))
        self.final_rms_norm = RMSNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self, in_indices):
        in_features = self.embedding(in_indices)
        for transformer_block in self.transformer_blocks:
            in_features = transformer_block(in_features)
        in_features = self.final_rms_norm(in_features)
        in_features = self.final_linear(in_features)
        return in_features

def cross_entropy(o: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss ℓi = -log softmax(oi)[xi+1]
    Arguments:
        o: Tensor of shape [..., vocab_size] – logits
        x_next: Tensor of shape [...] – target indices
    Returns:
        Scalar – mean cross-entropy loss over the batch
    """

    row_indices = torch.arange(o.shape[0])
    # Step 1: subtract max for numerical stability
    o_max = o.max(dim=-1, keepdim=True).values
    o_stable = o - o_max

    # Step 2: compute logsumexp
    logsumexp = torch.log(torch.sum(torch.exp(o_stable), dim=-1))

    # Step 3: gather the logit at the target index
    target_logit = o_stable[row_indices, x_next] 

    # Step 4: compute negative log likelihood
    loss = logsumexp - target_logit

    # Step 5: return mean loss
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    # opt = opt_class(
    #     model.parameters(),
    #     lr=1e-3,
    #     weight_decay=0.01,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    # )
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {
            "lr" : lr,
            "beta1" : betas[0], 
            "beta2" : betas[1],
            "eps" : eps, 
            "lam" : weight_decay 
        }
        super().__init__(params=params, defaults = defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lam = group["lam"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                grad = p.grad.data 
                t = state.get("t", torch.ones_like(grad)) 
                m = state.get("m", torch.zeros_like(grad))  #β1m + (1−β1)g
                m = beta1 * m + (1-beta1) * grad
                v = state.get("v", torch.zeros_like(grad))  #β2v + (1−β2)g2
                v = beta2 * v + (1-beta2) * (grad ** 2)
                alpha_t = lr * torch.sqrt(1 - (beta2 ** t)) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps) 
                p.data -= lr * lam * p.data
                state["t"] = t + 1 
                state["m"] = m 
                state["v"] = v 
        return loss

def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        return t / Tw * alpha_max
    if t >= Tw and t <= Tc:
        return alpha_min + 0.5 * (1 + math.cos((t - Tw)/(Tc - Tw) * torch.pi)) * (alpha_max - alpha_min)
    if t > Tc:
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = list(parameters)
    total_norm = 0.0

    # Calculate the global norm
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    # Apply clipping if needed
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
    total_len = len(dataset)
    x = []
    y = []
    for i in range(batch_size):
        start_idx = torch.randint(low=1, high=total_len - context_length + 1, size=(1,))
        x.append(torch.tensor(dataset[start_idx -1 : start_idx -1 + context_length], device = device))
        y.append(torch.tensor(dataset[start_idx : start_idx + context_length], device = device))
    # x = torch.concat(x, dim=0)    
    # y = torch.concat(y, dim=0)
    x = rearrange(x, "batch_size len -> batch_size len", batch_size=batch_size)
    y = rearrange(y, "batch_size len -> batch_size len", batch_size=batch_size)
    return (x,y)
        
def save_checkpoint(model : torch.nn.Module, optimizer : torch.optim.Optimizer, iteration : int , out : str):
    """
        should dump all the state from the first three parameters into the file-like object out. 
        You can use the state_dict method of both the model and the optimizer to get their relevant 
        states and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path 
        or a file-like object here). A typical choice is to have obj be a dictionary, 
        but you can use whatever format you want as long as you can load your checkpoint later.

        This function expects the following parameters:
           model: torch.nn.Module
           optimizer: torch.optim.Optimizer
           iteration: int
           out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    save_dict = {}
    save_dict["model_state_dict"] = model.state_dict()
    save_dict["optimizer_state_dict"] = optimizer.state_dict()
    save_dict["iteration"] = iteration
    torch.save(save_dict, out)
    return

def load_checkpoint(src:str, model : torch.nn.Module , optimizer:torch.optim.Optimizer):
    save_dict = torch.load(src)
    model.load_state_dict(save_dict["model_state_dict"])
    optimizer.load_state_dict(save_dict["optimizer_state_dict"])
    return save_dict["iteration"]

