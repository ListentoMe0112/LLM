from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from typing import List, Tuple, Dict

import numpy.typing as npt
import torch
from torch import Tensor 
import torch.nn.functional as F
import multiprocessing
import regex as re
from collections import defaultdict
from cs336_basics import tokenizer
from cs336_basics import utils 

from einops import einsum, reduce,rearrange, repeat


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    linear = utils.Linear(d_in, d_out)
    with torch.no_grad():
        linear.param.data.copy_(weights)
    return linear(in_features)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = utils.Embedding(vocab_size, d_model)
    with torch.no_grad():
        embedding.embedding_matrix.copy_(weights)
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    ffn = utils.FFN(d_model, d_ff)
    with torch.no_grad():
        ffn.w1.copy_(w1_weight) 
        ffn.w2.copy_(w2_weight) 
        ffn.w3.copy_(w3_weight) 
    return ffn(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # print('--------------------------')
    # print(mask.shape) # (4, 12 16)
    # print(Q.shape) # (4, 12, 64)
    # print(K.shape) # (4, 12, 16)
    # print('--------------------------')
    return utils.dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    Q = einsum(in_features, q_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
    K = einsum(in_features, k_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
    V = einsum(in_features, v_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")

    Q = rearrange(Q, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)
    K = rearrange(K, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)
    V = rearrange(V, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)

    multi_head_attention = utils.MultiHeadAttention(d_model, num_heads)
    O = multi_head_attention(Q, K, V)

    O  = rearrange(O, "... num_head seq_len d_k-> ... seq_len (num_head d_k)")
    output = einsum(O, o_proj_weight, "... seq_len d_v, d_model d_v -> ... seq_len d_model")
    
    return output

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # Q = einsum(in_features, q_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
    # K = einsum(in_features, k_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
    # V = einsum(in_features, v_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")

    # Q = rearrange(Q, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)
    # K = rearrange(K, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)
    # V = rearrange(V, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)

    # multi_head_attention = utils.MultiHeadAttention(d_model, num_heads)
    # O = multi_head_attention(Q, K, V)

    # O  = rearrange(O, "... num_head seq_len d_k-> ... seq_len (num_head d_k)")
    # output = einsum(O, o_proj_weight, "... seq_len d_v, d_model d_v -> ... seq_len d_model")
    # 
    # return output


    multi_head_attention = utils.MultiHeadAttention(d_model, num_heads)
    Q = einsum(in_features, q_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
    K = einsum(in_features, k_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
    V = einsum(in_features, v_proj_weight, "... seq_len d_in, d_k d_in -> ... seq_len d_k")

    Q = rearrange(Q, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)
    K = rearrange(K, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)
    V = rearrange(V, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = num_heads)

    rope = utils.RoPE(theta, Q.shape[-1], max_seq_len) 
    Q = rope(Q, token_positions)
    K = rope(K, token_positions)
    O =  multi_head_attention(Q, K, V)
    O  = rearrange(O, "... num_head seq_len d_k-> ... seq_len (num_head d_k)")
    O = einsum(O, o_proj_weight, "... seq_len d_v, d_model d_v -> ... seq_len d_model")
    return O


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = utils.RoPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.
    
    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = utils.TransformerBlock(d_model, num_heads, theta, max_seq_len, d_ff)
    with torch.no_grad():
        """
            attn.q_proj.weight
            attn.k_proj.weight
            attn.v_proj.weight
            attn.output_proj.weight
            ln1.weight
            ffn.w1.weight
        """
        transformer_block.q_proj.param.copy_(weights["attn.q_proj.weight"])
        transformer_block.k_proj.param.copy_(weights["attn.k_proj.weight"])
        transformer_block.v_proj.param.copy_(weights["attn.v_proj.weight"])
        transformer_block.o_proj.param.copy_(weights["attn.output_proj.weight"])
        transformer_block.rms_norm1.param.copy_(weights["ln1.weight"])
        transformer_block.ffn.w1.copy_(weights["ffn.w1.weight"])
        transformer_block.ffn.w2.copy_(weights["ffn.w2.weight"])
        transformer_block.ffn.w3.copy_(weights["ffn.w3.weight"])
        transformer_block.rms_norm2.param.copy_(weights["ln2.weight"])
    return transformer_block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = utils.TransformerLanguageModel(vocab_size, d_model, num_heads, rope_theta, context_length, d_ff, num_layers)
    with torch.no_grad():
        transformer_lm.embedding.embedding_matrix.copy_(weights["token_embeddings.weight"])
        for i in range(num_layers):
            pre = f"layers.{i}."
            transformer_lm.transformer_blocks[i].q_proj.param.copy_(weights[pre + "attn.q_proj.weight"])
            transformer_lm.transformer_blocks[i].k_proj.param.copy_(weights[pre + "attn.k_proj.weight"])
            transformer_lm.transformer_blocks[i].v_proj.param.copy_(weights[pre + "attn.v_proj.weight"])
            transformer_lm.transformer_blocks[i].o_proj.param.copy_(weights[pre + "attn.output_proj.weight"])
            transformer_lm.transformer_blocks[i].rms_norm1.param.copy_(weights[pre + "ln1.weight"])
            transformer_lm.transformer_blocks[i].ffn.w1.copy_(weights[pre + "ffn.w1.weight"])
            transformer_lm.transformer_blocks[i].ffn.w2.copy_(weights[pre + "ffn.w2.weight"])
            transformer_lm.transformer_blocks[i].ffn.w3.copy_(weights[pre + "ffn.w3.weight"])
            transformer_lm.transformer_blocks[i].rms_norm2.param.copy_(weights[pre + "ln2.weight"])
        transformer_lm.final_rms_norm.param.copy_(weights["ln_final.weight"])
        transformer_lm.final_linear.param.copy_(weights["lm_head.weight"])
    return transformer_lm(in_indices)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = utils.RMSNorm(d_model, eps)
    with torch.no_grad():
        rmsnorm.param.copy_(weights)
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return torch.sigmoid(in_features) * in_features


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return utils.get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return utils.softmax(in_features,dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return utils.cross_entropy(inputs, targets)


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    return utils.gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return utils.AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return utils.learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    utils.save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    return utils.load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return tokenizer.Tokenizer(vocab = vocab, merges = merges, special_tokens = special_tokens)


def split_corpus_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    Split the corpus into parts based on special tokens. The resulting parts should not contain any special tokens.
    Special tokens should be stripped out, and we return the text between those tokens.
    """
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)  # Create a pattern to match any special token
    split_indices = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    # Split the text at the special tokens, keeping them as separators
    parts = re.split(f"({pattern})", text)

    # Reassemble into separate documents (chunks), skipping the special tokens
    documents = []
    current_doc = []
    for part in parts:
        if part in special_tokens:
            if current_doc:  # Submit the current document
                documents.append("".join(current_doc))
                current_doc = []
        else:
            current_doc.append(part)

    # Add the last document if there is remaining content
    if current_doc:
        documents.append("".join(current_doc))

    return documents


def worker_wrapper(file_path, start, end, special_tokens):
    """Wrapper function to ensure proper resource cleanup"""
    try:
        return worker(file_path, start, end, special_tokens)
    finally:
        # Explicitly clean up resources
        import gc
        gc.collect()

def worker(file_path, start, end, special_tokens):
    """Process a file chunk and return token frequencies"""
    shared_dict = {}
    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            docs = split_corpus_by_special_tokens(chunk, special_tokens)
            for doc in docs:
                pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for match in re.finditer(pat, doc):
                    token_str = match.group()
                    token_bytes = token_str.encode("utf-8")
                    key = tuple(bytes([b]) for b in token_bytes)
                    shared_dict[key] = shared_dict.get(key, 0) + 1
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {str(e)}")
    return shared_dict

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Step 1: Vocabulary Initialization
    vocab = {}
    special_bytes = [token.encode("utf-8") for token in special_tokens]
    for idx, token_bytes in enumerate(special_bytes):
        vocab[idx] = token_bytes

    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # Step 2: Pre-tokenization
    def find_chunk_boundaries(
        file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    pre_token_dict = {}
    pre_token_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 256, "<|endoftext|>".encode("utf-8"))

    i = 0
    # Use a process pool with limited size to avoid too many open files
    max_processes = min(16, len(boundaries) - 1)  # Limit to 16 processes
    with multiprocessing.Pool(processes=max_processes) as pool:
        # Create arguments for each worker
        args_list = [(input_path, start, end, special_tokens) 
                     for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # Map tasks to workers and collect results
        results = pool.starmap(worker_wrapper, args_list)
        
        # Process results
        for result in results:
            pre_token_list.append(result)

    for tmp_token_list in pre_token_list:
        for k, v in tmp_token_list.items():
            if k not in pre_token_dict:
                pre_token_dict[k] = 0
            pre_token_dict[k] += v

    # Step 3: Merge
    def get_initial_stats(
        pre_token_dict: Dict[tuple[bytes], int],
    ) -> Dict[Tuple[bytes, bytes], int]:
        stats = defaultdict(int)
        for tokens, freq in pre_token_dict.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                stats[pair] += freq
        return stats

    def merge(
        pre_token_dict: dict[tuple[bytes], int],
        pair: Tuple[bytes, bytes],
        stats: Dict[Tuple[bytes, bytes], int],
    ):
        a, b = pair
        merged_token = a + b
        changes_stat = defaultdict(int)
        changes_vocab = defaultdict(int)
        for tokens, freq in pre_token_dict.items():
            new_word_tokens: List = []
            i = 0
            modified = False
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == (a, b):
                    left = tokens[i - 1] if i > 0 else None
                    right = tokens[i + 2] if (i + 2) < len(tokens) else None

                    if left:
                        old_pair = (left, a)
                        changes_stat[old_pair] -= freq
                        new_pair = (left, merged_token)
                        changes_stat[new_pair] += freq

                    if right:
                        old_pair = (b, right)
                        changes_stat[old_pair] -= freq
                        new_pair = (merged_token, right)
                        changes_stat[new_pair] += freq

                    changes_stat[(a, b)] -= freq
                    new_word_tokens.append(merged_token)
                    modified = True
                    i += 2
                else:
                    new_word_tokens.append(tokens[i])
                    i += 1
            if modified:
                changes_vocab[tokens] = 0
                changes_vocab[tuple(new_word_tokens)] = freq

        # Second pass: apply vocab changes
        for new_tokens, freq in changes_vocab.items():
            if freq == 0:
                del pre_token_dict[new_tokens]
            else:
                pre_token_dict[new_tokens] = freq

        # Update stats
        for p, delta in changes_stat.items():
            stats[p] += delta
            if stats[p] <= 0:
                del stats[p]

    def get_most_frequent_pair(
        stats: Dict[Tuple[bytes, bytes], int],
    ) -> Tuple[bytes, bytes]:
        """Select the pair with the highest frequency, breaking ties lexicographically."""
        # Key: (frequency, pair), so pairs with higher frequency come first.
        # For ties, lexicographically greater pairs are chosen.
        if len(stats) > 1:
            return max(stats.keys(), key=lambda x: (stats[x], x))
        else :
            return list(stats.keys())[0]

    merges: list[tuple[bytes, bytes]] = []
    init_stats = get_initial_stats(pre_token_dict)
    while len(vocab) < vocab_size:
        pair_merge = get_most_frequent_pair(init_stats)
        merge(pre_token_dict, pair_merge, init_stats)
        merges.append(pair_merge)
        vocab[len(vocab)] = pair_merge[0] + pair_merge[1]

    return vocab, merges
